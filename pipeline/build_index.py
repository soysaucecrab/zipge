"""Build static search indexes for serving.

Outputs (data/index/):
- doc_meta.json      : per-article metadata, ordered to match vector rows
- doc_vectors_i8.bin : int8-quantized e5 doc vectors (row-major), per-vector scale
- vector_scales.json : dequantization scales + dims
- related.json       : article id -> top-10 related via full-graph BFS
- tags.json          : tag -> {count, article ids} for tag browsing/search
"""
import json
from pathlib import Path

import numpy as np

from related_search import build_adj, load_graph, make_bfs, train_node2vec

ROOT = Path(__file__).resolve().parent.parent
EMB = ROOT / "data" / "embeddings"
ARTICLES = ROOT / "data" / "articles"
OUT = ROOT / "data" / "index"

TOP_K_RELATED = 10
SEARCH_MODEL_ALIAS = "small"  # e5-small won the size/quality comparison
EMB_FILE = "doc_embeddings_small.npz"
MODEL_ID = "intfloat/multilingual-e5-small"


def main():
    OUT.mkdir(parents=True, exist_ok=True)

    # --- article metadata, aligned with embedding order ---
    emb = np.load(EMB / EMB_FILE, allow_pickle=True)
    ids, V = list(emb["ids"]), emb["vectors"]
    docs = {}
    for f in ARTICLES.glob("*.json"):
        d = json.loads(f.read_text(encoding="utf-8"))
        docs[d["id"]] = d
    meta = []
    for aid in ids:
        d = docs[aid]
        meta.append({
            "id": d["id"],
            "slug": d["slug"],
            "title": d["title"],
            "board": (d["board"] or {}).get("slug"),
            "boardName": (d["board"] or {}).get("name"),
            "category": d.get("category"),
            "tags": d["tags"],
            "authors": [{"name": c["name"], "role": c["role"]} for c in d["contributors"]],
            "created": d["created"][:10],
            "thumbnail": d.get("thumbnail") or None,
            "preview": d.get("preview") or None,
            "views": d.get("views") or 0,
        })
    (OUT / "doc_meta.json").write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")

    # --- int8 quantized vectors ---
    scales = np.abs(V).max(axis=1) / 127.0
    Q = np.clip(np.round(V / scales[:, None]), -127, 127).astype(np.int8)
    (OUT / "doc_vectors_i8.bin").write_bytes(Q.tobytes())
    (OUT / "vector_scales.json").write_text(json.dumps({
        "model": MODEL_ID,
        "dim": int(V.shape[1]),
        "count": int(V.shape[0]),
        "scales": [round(float(s), 8) for s in scales],
    }), encoding="utf-8")

    # quantization quality check: top-10 overlap vs float32
    Vq = Q.astype(np.float32) * scales[:, None]
    Vq /= np.linalg.norm(Vq, axis=1, keepdims=True)
    rng = np.random.default_rng(0)
    overlap = []
    for i in rng.choice(len(ids), 100, replace=False):
        t_f = set(np.argsort(-(V @ V[i]))[1:11])
        t_q = set(np.argsort(-(Vq @ Vq[i]))[1:11])
        overlap.append(len(t_f & t_q) / 10)
    print(f"int8 top-10 overlap vs f32: {np.mean(overlap):.3f}")

    # --- tag index ---
    tag_map = {}
    for m in meta:
        for t in m["tags"]:
            tag_map.setdefault(t, []).append(m["id"])
    tags_out = {t: {"count": len(v), "articles": v}
                for t, v in sorted(tag_map.items(), key=lambda x: -len(x[1]))}
    (OUT / "tags.json").write_text(json.dumps(tags_out, ensure_ascii=False), encoding="utf-8")
    print(f"tags: {len(tags_out)}, tagged articles: {len({a for v in tag_map.values() for a in v})}")

    # --- related articles via full-graph BFS ---
    nodes, edges = load_graph()
    idx = {n["id"]: i for i, n in enumerate(nodes)}
    A = build_adj(nodes, edges, idx)
    Z = train_node2vec(A, len(nodes))
    bfs = make_bfs(A, Z, nodes)

    related = {}
    for aid in ids:
        res = bfs(idx[f"a:{aid}"])
        top = sorted(res.items(), key=lambda x: -x[1])[:TOP_K_RELATED]
        related[aid] = [
            {"id": nodes[v]["id"][2:], "slug": docs[nodes[v]["id"][2:]]["slug"], "score": round(s, 4)}
            for v, s in top
        ]
    (OUT / "related.json").write_text(json.dumps(related, ensure_ascii=False), encoding="utf-8")

    # self-contained variant for server-side rendering (no extra meta lookup)
    related_links = {
        aid: [{
            "id": r["id"], "slug": r["slug"], "title": docs[r["id"]]["title"],
            "boardSlug": (docs[r["id"]]["board"] or {}).get("slug"), "score": r["score"],
        } for r in items]
        for aid, items in related.items()
    }
    (OUT / "related_links.json").write_text(json.dumps(related_links, ensure_ascii=False), encoding="utf-8")

    sizes = {f.name: f"{f.stat().st_size/1024:.0f}KB" for f in sorted(OUT.iterdir())}
    n_rel = [len(v) for v in related.values()]
    print(f"related: {len(related)} articles, min/mean top-k: {min(n_rel)}/{np.mean(n_rel):.1f}")
    print("files:", sizes)


if __name__ == "__main__":
    main()
