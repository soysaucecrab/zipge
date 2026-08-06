"""Semantic-tier benchmark (T1 small / T2 large).

Mirrors production: the embedded text is the T0-corrected query
(corrections.json from bench_t0.ts). Doc vectors are the production-style
title+content embeddings. Usage: bench_semantic.py small|large
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parent.parent
BENCH = ROOT / "data" / "bench"

MODELS = {"small": "intfloat/multilingual-e5-small", "large": "intfloat/multilingual-e5-large"}


def main():
    alias = sys.argv[1]
    emb = np.load(ROOT / "data" / "embeddings" / f"doc_embeddings_{alias}.npz", allow_pickle=True)
    ids, D = list(emb["ids"]), emb["vectors"]
    idx_of = {i: p for p, i in enumerate(ids)}

    queries = json.loads((BENCH / "queries.json").read_text(encoding="utf-8"))
    corrections = json.loads((BENCH / "corrections.json").read_text(encoding="utf-8"))

    model = SentenceTransformer(MODELS[alias])
    texts = []
    for q in queries:
        key = q["article"] + "|" + q["variant"]
        texts.append("query: " + corrections.get(key, q["text"]))
    Q = np.asarray(model.encode(texts, batch_size=64, normalize_embeddings=True,
                                show_progress_bar=False), dtype=np.float32)

    stats = defaultdict(lambda: {"r10": 0, "rr": 0.0, "n": 0})
    S = Q @ D.T
    for qi, q in enumerate(queries):
        t = idx_of[q["article"]]
        rank = 1 + int((S[qi] > S[qi, t]).sum())
        v = stats[q["variant"]]
        v["n"] += 1
        v["r10"] += rank <= 10
        v["rr"] += 1.0 / rank
    for variant, v in stats.items():
        print(f"{alias} {variant:6s} Recall@10: {v['r10']/v['n']:.3f}  MRR: {v['rr']/v['n']:.3f}  (n={v['n']})")


if __name__ == "__main__":
    main()
