"""Student encoder — step 2/4: teacher embeddings (GPU; default teacher BAAI/bge-m3).

Outputs (work dir):
  teacher_docs.npy     float32 [N, out_dim]  doc vectors in doc_meta row order — chunked and
                       mean-pooled exactly like pipeline/embed_articles.py
  vocab_input.npz      q int8 [V, 384], scale float32 [V], explained float — student input:
                       teacher term vectors -> PCA 384 -> per-row int8 (the shipped format)
  teacher_q_train.npy  float16 [n_train, out_dim]  teacher embeddings of train_queries.json

Run: python pipeline/student/embed_teacher.py [--articles data/articles --index data/index --work data/student]
"""
import argparse
import re
import sys
import time
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
from embed_articles import chunk_text, clean_markdown  # noqa: E402  same chunking as the search index
from common import load_json, quantize_rows  # noqa: E402

INPUT_DIM = 384


def article_chunks(articles_dir):
    out = {}
    for f in Path(articles_dir).glob("*.json"):
        d = load_json(f)
        paras = [clean_markdown(p) for p in re.split(r"(?:\r?\n){2,}", d["content"] or "")]
        paras = [p for p in paras if p]
        out[str(d["id"])] = (d["title"], chunk_text("\n\n".join(paras)) if paras else [""])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--articles", default=str(ROOT / "data" / "articles"))
    ap.add_argument("--index", default=str(ROOT / "data" / "index"))
    ap.add_argument("--work", default=str(ROOT / "data" / "student"))
    ap.add_argument("--teacher", default="BAAI/bge-m3")
    ap.add_argument("--query-prefix", default="", help='e5 models need "query: "')
    ap.add_argument("--passage-prefix", default="", help='e5 models need "passage: "')
    ap.add_argument("--batch", type=int, default=32)
    a = ap.parse_args()
    work = Path(a.work)

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("WARNING: no CUDA — teacher embedding on CPU will be slow", file=sys.stderr)
    t0 = time.time()
    model = SentenceTransformer(a.teacher, device=device)

    # 1) doc vectors aligned to the index row order
    meta = load_json(Path(a.index) / "doc_meta.json")
    arts = article_chunks(a.articles)
    texts, owner, missing = [], [], 0
    for row, m in enumerate(meta):
        if str(m["id"]) not in arts:
            missing += 1
        title, chunks = arts.get(str(m["id"]), (m["title"], [""]))
        for c in chunks:
            texts.append(f"{a.passage_prefix}{title}\n{c}")
            owner.append(row)
    C = model.encode(texts, batch_size=a.batch, normalize_embeddings=True, convert_to_numpy=True).astype(np.float32)
    owner = np.asarray(owner)
    D = np.zeros((len(meta), C.shape[1]), np.float32)
    for r in range(len(meta)):
        v = C[owner == r].mean(0)
        D[r] = v / np.linalg.norm(v)
    np.save(work / "teacher_docs.npy", D)
    print(f"docs: {len(meta)} rows, {len(texts)} chunks, missing articles {missing} -> {D.shape}")

    # 2) student input: teacher term vectors -> PCA 384 -> int8 (what the browser receives).
    #    With an extension lexicon (build_lexicon.py) the basis is fixed and reused, so the
    #    shipped lexicon shards stay in the same input space across retrains.
    terms = load_json(work / "vocab.json")["terms"]
    T = model.encode([a.query_prefix + t for t in terms], batch_size=256, normalize_embeddings=True,
                     convert_to_numpy=True).astype(np.float32)
    basis_path = work / "lexicon" / "basis.npz"
    if basis_path.exists():
        b = np.load(basis_path)
        mu, comps, explained, basis_sha1 = b["mu"], b["comps"], float(b["explained"]), str(b["basis_sha1"])
        how = f"fixed lexicon basis {basis_sha1[:12]}"
    else:
        mu = T.mean(0)
        _, S, Vt = np.linalg.svd(T - mu, full_matrices=False)
        comps = Vt[:INPUT_DIM]
        explained = float((S[:INPUT_DIM] ** 2).sum() / (S ** 2).sum())
        basis_sha1, how = "", "fitted on the core vocabulary (no lexicon)"
    q, scale = quantize_rows((T - mu) @ comps.T)
    np.savez(work / "vocab_input.npz", q=q, scale=scale, explained=explained, basis_sha1=basis_sha1)
    print(f"vocab: {len(terms)} terms -> PCA {INPUT_DIM} ({how}, explained {explained * 100:.1f}%) -> int8")

    # 3) teacher targets for the training queries
    train = load_json(work / "train_queries.json")
    Q = model.encode([a.query_prefix + x["text"] for x in train], batch_size=256, normalize_embeddings=True,
                     convert_to_numpy=True)
    np.save(work / "teacher_q_train.npy", Q.astype(np.float16))
    print(f"train queries: {len(train)} encoded · total {time.time() - t0:.0f}s ({a.teacher}, {device})")


if __name__ == "__main__":
    main()
