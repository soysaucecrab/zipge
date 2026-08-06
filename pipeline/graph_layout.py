"""Compute a 2D layout for the document graph (for visualization).

t-SNE projection of full-graph node2vec embeddings: walk co-occurrence
communities become spatial clusters. Output: data/graph/layout.json
"""
import json
from pathlib import Path

import numpy as np
from sklearn.manifold import TSNE

from related_search import build_adj, load_graph, train_node2vec

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "graph" / "layout.json"


def main():
    nodes, edges = load_graph()
    idx = {n["id"]: i for i, n in enumerate(nodes)}
    n = len(nodes)
    A = build_adj(nodes, edges, idx)
    Z = train_node2vec(A, n)

    pos = TSNE(n_components=2, perplexity=25, init="pca", random_state=42,
               max_iter=1500).fit_transform(Z)
    pos -= pos.mean(axis=0)
    pos /= np.abs(pos).max()

    deg = np.asarray((A > 0).sum(axis=1)).ravel()
    out = {
        "nodes": [
            {"id": nd["id"], "type": nd["type"], "label": nd["label"],
             "x": round(float(pos[i, 0]), 4), "y": round(float(pos[i, 1]), 4),
             "deg": int(deg[i])}
            for i, nd in enumerate(nodes)
        ],
        "edges": [{"s": idx[e["src"]], "t": idx[e["dst"]], "type": e["type"],
                   "w": e["weight"]} for e in edges],
    }
    OUT.write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
    print(f"layout: {n} nodes, {len(out['edges'])} edges -> {OUT} ({OUT.stat().st_size//1024}KB)")


if __name__ == "__main__":
    main()
