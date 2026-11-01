"""Student encoder — step 4/4: evaluation report (GPU recommended for the model/teacher rows).

Rows
  e5-small      reference: e5-small queries vs the index's e5-small doc vectors (zipge 0.1's 37MB browser model)
  teacher       teacher queries vs the student doc vectors (= server-side embedding ceiling)
  student       the exported assets scored through common.StudentRuntime (= exactly what ships)
Sets (empty ones are skipped)
  dev / test    your hand-written queries the keyword tier cannot catch (build_queries.mjs --eval)
  title         title -> its own article (semantic only; the keyword tier covers these in the app)
  bodyseg       one held-out body segment per article -> that article
  oov           queries around words absent from the corpus (--oov) — the extension lexicon's job
Writes <work>/reports/eval-<UTC>.md / .json

Run: python pipeline/student/evaluate.py [--work data/student --index data/index] [--skip-models]
"""
import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(HERE))
from common import (ASSET_FILES, LEX_META, StudentRuntime, bootstrap_ci, load_doc_vectors,  # noqa: E402
                    load_json, metrics, mrr_of, r10_of, ranks, write_json)

ALL_SETS = ["dev", "test", "title", "bodyseg", "oov"]


def encode_with(model_id, prefix, ev, sets):
    import torch
    from sentence_transformers import SentenceTransformer
    m = SentenceTransformer(model_id, device="cuda" if torch.cuda.is_available() else "cpu")
    return {k: m.encode([prefix + x["text"] for x in ev[k]], normalize_embeddings=True, convert_to_numpy=True)
            for k in sets}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", default=str(ROOT / "data" / "student"))
    ap.add_argument("--index", default=str(ROOT / "data" / "index"),
                    help="search index dir (doc_meta + e5-small doc vectors)")
    ap.add_argument("--student", help="student asset dir (default: --index)")
    ap.add_argument("--teacher-query-prefix", default="")
    ap.add_argument("--model-mode", default="intfloat/multilingual-e5-small")
    ap.add_argument("--skip-models", action="store_true", help="no GPU: student rows only")
    ap.add_argument("--holdout", default="", help="holdout.json written by train.py --holdout-frac (new-article split)")
    a = ap.parse_args()
    work, index = Path(a.work), Path(a.index)

    ev = load_json(work / "eval_sets.json")
    sets = [k for k in ALL_SETS if ev.get(k)]
    student_dir = Path(a.student or a.index)
    lex_dir = student_dir if (student_dir / LEX_META).exists() else index
    rt = StudentRuntime(student_dir, lexicon_dir=lex_dir)
    rows, agree = {}, None

    if not a.skip_models:
        D_small = load_doc_vectors(index, "doc_vectors_i8.bin", "vector_scales.json")
        Qs = encode_with(a.model_mode, "query: ", ev, sets)
        rows["참고: e5-small (zipge 0.1의 37MB 브라우저 모델)"] = {k: ranks(Qs[k], np.ones(len(ev[k]), bool), ev[k], D_small) for k in sets}
        Qt = encode_with(rt.meta["teacher"], a.teacher_query_prefix, ev, sets)
        rows[f"teacher {rt.meta['teacher']} (서버 임베딩 상한)"] = {
            k: ranks(Qt[k], np.ones(len(ev[k]), bool), ev[k], rt.docs) for k in sets}

    student, coverage = {}, {}
    for k in sets:
        Q, valid, _ = rt.encode(ev[k])
        student[k] = ranks(Q, valid, ev[k], rt.docs)
        coverage[k] = float(np.mean(valid)) * 100
        if k == "dev" and not a.skip_models:
            ts = np.argsort(-(Qt["dev"] @ rt.docs.T), 1)[:, :10]
            ss = np.argsort(-(Q @ rt.docs.T), 1)[:, :10]
            agree = 100.0 * np.mean([len(set(x) & set(y)) / 10 for x, y, v in zip(ts, ss, valid) if v])
    size = sum((student_dir / f).stat().st_size for f in ASSET_FILES.values()) / 1e6
    if rt.lex is not None:  # ablation: the same student with the extension lexicon switched off
        rows["학생 — 확장 어휘 끔"] = {k: ranks(*rt.encode(ev[k], use_lexicon=False)[:2], ev[k], rt.docs) for k in sets}
    lex_note = f" + 확장 어휘 {len(rt.lex)}단어" if rt.lex is not None else ""
    rows[f"학생 (배포 포맷, {rt.meta['members']}멤버, {size:.1f}MB{lex_note})"] = student

    # ---- report
    M = {name: {k: metrics(r[k]) for k in sets} for name, r in rows.items()}
    ci = {name: {"test_mrr": bootstrap_ci(r["test"], mrr_of), "test_r10": bootstrap_ci(r["test"], r10_of)}
          for name, r in rows.items()} if "test" in sets else {}
    f = lambda d: f"{d['r5']:.1f} / {d['r10']:.1f} / {d['mrr']:.3f}"
    head = ["방식"] + [f"{k} R@5/R@10/MRR" for k in sets] + (["test MRR 95% CI"] if ci else [])
    lines = [
        f"# 학생 인코더 평가 — {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
        "",
        f"문서 {rt.docs.shape[0]}편 · 셋 크기 " + " · ".join(f"{k} {len(ev[k])}" for k in sets),
        "",
        "| " + " | ".join(head) + " |",
        "|" + "---|" * len(head),
    ]
    for name, m in M.items():
        cells = [name] + [f(m[k]) for k in sets]
        if ci:
            c = ci[name]["test_mrr"]
            cells.append(f"{c[0]:.3f}–{c[1]:.3f}")
        lines.append("| " + " | ".join(cells) + " |")
    lines += ["", "학생 슬롯 커버리지(형태소가 하나라도 잡힌 쿼리 %): " + " · ".join(f"{k} {v:.0f}%" for k, v in coverage.items())]
    if agree is not None:
        lines += ["", f"학생–teacher top-10 일치율(dev): {agree:.1f}%"]

    if a.holdout:
        hold = set(load_json(a.holdout))
        lines += ["", f"## 신규 글 일반화 (학습에서 숨긴 문서 {len(hold)}편)", "",
                  "| 방식 | 신규 글 대상 R@10/MRR | 기존 글 대상 R@10/MRR |", "|---|---|---|"]
        for name, r in rows.items():
            new, old = [], []
            for k in (x for x in ("dev", "test", "bodyseg") if x in sets):
                for it, rk in zip(ev[k], r[k]):
                    if all(x in hold for x in it["rows"]):
                        new.append(rk)
                    elif not any(x in hold for x in it["rows"]):
                        old.append(rk)
            mn, mo = metrics(new), metrics(old)
            lines.append(f"| {name} | {mn['r10']:.1f} / {mn['mrr']:.3f} (n={mn['n']}) | {mo['r10']:.1f} / {mo['mrr']:.3f} (n={mo['n']}) |")

    report = "\n".join(lines) + "\n"
    print(report)
    rep_dir = work / "reports"
    rep_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M")
    (rep_dir / f"eval-{stamp}.md").write_text(report, encoding="utf-8")
    write_json(rep_dir / f"eval-{stamp}.json", {"metrics": M, "ci": ci, "agree_top10_dev": agree,
                                                "student_meta": {k: v for k, v in rt.meta.items()
                                                                 if k not in ("terms", "doc_ids")}})
    print(f"report -> {rep_dir}/eval-{stamp}.md")


if __name__ == "__main__":
    main()
