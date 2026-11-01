"""Extension lexicon — step B: embed, filter, fix the student input basis, shard (GPU).

Run rarely (when the sources or the domain change), after embed_teacher.py:
    python pipeline/student/build_lexicon.py [--work data/student --out data/index]

1. Candidates (build_lexicon_candidates.mjs) minus the core vocabulary.
2. Teacher vectors; keep words whose best cosine to any corpus document is >= --aff
   (words the corpus cannot answer would only add noise and bytes).
3. Fix the student input space: PCA 384 fitted on core vocabulary + kept lexicon, saved as
   <work>/lexicon/basis.npz with its sha1. embed_teacher.py reuses this basis on every later
   retrain, so the shipped lexicon shards stay valid until the lexicon itself is rebuilt.
4. Outputs: <work>/vocab_input.npz (core vocab in the fixed basis — overwrites embed_teacher's),
   <work>/lexicon/lexicon.npz (training/eval copy), and the packs + student_lex_meta.json in --out.
"""
import argparse
import hashlib
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(HERE))
from common import LEX_SHARDS, load_json, quantize_rows, write_json, write_lexicon_shards  # noqa: E402

INPUT_DIM = 384


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", default=str(ROOT / "data" / "student"))
    ap.add_argument("--out", default=str(ROOT / "data" / "index"), help="index dir that receives the packs")
    ap.add_argument("--candidates", help="default <work>/lexicon/candidates.json")
    ap.add_argument("--teacher", default="BAAI/bge-m3")
    ap.add_argument("--query-prefix", default="")
    ap.add_argument("--aff", type=float, default=0.37,
                    help="keep words whose max cosine to a corpus document is >= this (core-vocab p1~p5 = 0.34~0.38)")
    ap.add_argument("--shards", type=int, default=LEX_SHARDS)
    a = ap.parse_args()
    work = Path(a.work)
    lexdir = work / "lexicon"
    lexdir.mkdir(parents=True, exist_ok=True)

    import torch
    from sentence_transformers import SentenceTransformer
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    t0 = time.time()

    core = load_json(work / "vocab.json")["terms"]
    coreset = set(core)
    cj = load_json(Path(a.candidates) if a.candidates else lexdir / "candidates.json")
    cands = sorted(({w for w, _ in cj["sub"]} | set(cj["single"]) | set(cj["compound"])) - coreset)

    model = SentenceTransformer(a.teacher, device=dev)
    if dev == "cuda":
        model.half()

    def embed(terms):
        return model.encode([a.query_prefix + t for t in terms], batch_size=1024, normalize_embeddings=True,
                            convert_to_numpy=True, show_progress_bar=False).astype(np.float32)

    Ec, Ex = embed(core), embed(cands)
    D = np.load(work / "teacher_docs.npy").astype(np.float32)
    aff = np.concatenate([(Ex[i:i + 65536] @ D.T).max(1) for i in range(0, len(Ex), 65536)])
    keep = aff >= a.aff
    terms = [t for t, k in zip(cands, keep) if k]
    Ek = Ex[keep]
    print(f"candidates {len(cands)} -> affinity >= {a.aff}: {len(terms)} words ({time.time() - t0:.0f}s)")

    # fixed input basis
    X = np.concatenate([Ec, Ek])
    mu = X.mean(0)
    _, S, Vt = np.linalg.svd(X - mu, full_matrices=False)
    comps = Vt[:INPUT_DIM].astype(np.float32)
    explained = float((S[:INPUT_DIM] ** 2).sum() / (S ** 2).sum())
    basis_sha1 = hashlib.sha1(mu.astype("<f4").tobytes() + comps.astype("<f4").tobytes()).hexdigest()
    np.savez(lexdir / "basis.npz", mu=mu.astype(np.float32), comps=comps, explained=explained, basis_sha1=basis_sha1)

    qc, sc = quantize_rows((Ec - mu) @ comps.T)
    np.savez(work / "vocab_input.npz", q=qc, scale=sc, explained=explained, basis_sha1=basis_sha1)
    ql, sl = quantize_rows((Ek - mu) @ comps.T)
    np.savez(lexdir / "lexicon.npz", terms=np.array(terms), q=ql, scale=sl, basis_sha1=basis_sha1)

    size = write_lexicon_shards(a.out, terms, ql, sl, basis_sha1=basis_sha1, n_shards=a.shards,
                                extra={"teacher": a.teacher, "aff": a.aff, "sources": cj.get("sources", {})})
    write_json(lexdir / "build.json", {"candidates": len(cands), "kept": len(terms), "aff": a.aff,
                                       "explained": explained, "basis_sha1": basis_sha1, "bytes": size})
    print(f"basis PCA {INPUT_DIM} explained {explained * 100:.1f}% sha1 {basis_sha1[:12]} · shards {a.shards} "
          f"{size / 1e6:.1f}MB (~{size / a.shards / 1024:.0f}KB each) · {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
