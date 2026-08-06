"""Similarity-weighted BFS (beam search) over the document graph.

Traversal: expand graph neighbors breadth-first (attribute nodes are
pass-through hops), score candidates by node2vec cosine to the source with
per-depth decay, keep top-`beam` frontier per level.

Evaluated on the same held-out protocol as baseline_related.py, then run on
the full graph for spot checks.
"""
import json
import random
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import sparse

ROOT = Path(__file__).resolve().parent.parent
GRAPH = ROOT / "data" / "graph"

SEED = 42
TEST_FRAC = 0.15
EDGE_WEIGHTS = {"links_to": 1.0, "knn": None, "in_series": 1.0, "authored_by": 0.5,
                "tagged": 0.5, "in_category": 0.3, "in_board": 0.05}
N2V_DIM, N2V_WALKS, N2V_LEN = 128, 10, 80
BEAM, MAX_DEPTH, DECAY = 128, 4, 0.9


def load_graph():
    nodes = json.loads((GRAPH / "nodes.json").read_text(encoding="utf-8"))
    edges = [json.loads(l) for l in (GRAPH / "edges.jsonl").open(encoding="utf-8")]
    return nodes, edges


def build_adj(nodes, edges, idx, skip_pairs=None):
    w = defaultdict(float)
    for e in edges:
        if e["type"] == "links_to":
            if skip_pairs and tuple(sorted((e["src"], e["dst"]))) in skip_pairs:
                continue
            wt = 1.0
        else:
            wt = e["weight"] if EDGE_WEIGHTS[e["type"]] is None else EDGE_WEIGHTS[e["type"]]
        a, b = idx[e["src"]], idx[e["dst"]]
        w[(a, b)] = max(w[(a, b)], wt)
        w[(b, a)] = max(w[(b, a)], wt)
    rows, cols, vals = zip(*[(a, b, v) for (a, b), v in w.items()])
    return sparse.csr_matrix((vals, (rows, cols)), shape=(len(nodes), len(nodes)))


def train_node2vec(A, n):
    from gensim.models import Word2Vec

    nbrs, cums = [], []
    for i in range(n):
        s, e = A.indptr[i], A.indptr[i + 1]
        nbrs.append(A.indices[s:e])
        cums.append(np.cumsum(A.data[s:e]))
    walks = []
    wrng = np.random.default_rng(SEED)
    order = np.arange(n)
    for _ in range(N2V_WALKS):
        wrng.shuffle(order)
        for start in order:
            walk, cur = [str(start)], start
            for _ in range(N2V_LEN - 1):
                c = cums[cur]
                if len(c) == 0:
                    break
                cur = nbrs[cur][np.searchsorted(c, wrng.random() * c[-1])]
                walk.append(str(cur))
            walks.append(walk)
    w2v = Word2Vec(walks, vector_size=N2V_DIM, window=10, min_count=0, sg=1,
                   workers=4, epochs=5, seed=SEED)
    Z = np.stack([w2v.wv[str(i)] for i in range(n)])
    return Z / np.linalg.norm(Z, axis=1, keepdims=True)


def make_bfs(A, Z, nodes):
    nbr_of = [A.indices[A.indptr[i]:A.indptr[i + 1]] for i in range(len(nodes))]
    is_article = np.array([n["type"] == "article" for n in nodes])

    def related_bfs(src_i, beam=BEAM, max_depth=MAX_DEPTH, decay=DECAY):
        """Return {article_index: score} reachable from src via beam BFS."""
        sims = Z @ Z[src_i]
        best = {}
        frontier = [src_i]
        visited = {src_i}
        for d in range(max_depth):
            cand = []
            for u in frontier:
                for v in nbr_of[u]:
                    if v in visited:
                        continue
                    visited.add(v)
                    s = float(sims[v]) * (decay ** d)
                    cand.append((v, s))
                    if is_article[v]:
                        best[v] = s
            cand.sort(key=lambda x: -x[1])
            frontier = [v for v, _ in cand[:beam]]
        return best

    return related_bfs


def main():
    t0 = time.time()
    rng = random.Random(SEED)
    nodes, edges = load_graph()
    idx = {n["id"]: i for i, n in enumerate(nodes)}
    labels = {n["id"]: n["label"] for n in nodes}
    art_ids = [n["id"] for n in nodes if n["type"] == "article"]
    pos_of = {a: p for p, a in enumerate(art_ids)}

    # --- eval on held-out split (identical to baseline) ---
    link_pairs = sorted({tuple(sorted((e["src"], e["dst"]))) for e in edges if e["type"] == "links_to"})
    rng.shuffle(link_pairs)
    n_test = int(len(link_pairs) * TEST_FRAC)
    test_pairs, train_links = set(link_pairs[:n_test]), set(link_pairs[n_test:])

    A = build_adj(nodes, edges, idx, skip_pairs=test_pairs)
    Z = train_node2vec(A, len(nodes))
    print(f"node2vec trained ({time.time()-t0:.0f}s)")
    bfs = make_bfs(A, Z, nodes)

    train_nb = defaultdict(set)
    for a, b in train_links:
        train_nb[a].add(b)
        train_nb[b].add(a)
    tests = [(a, b) for a, b in sorted(test_pairs)] + [(b, a) for a, b in sorted(test_pairs)]

    cache = {}
    recalls, rrs, unreached = [], [], 0
    for src, dst in tests:
        if src not in cache:
            cache[src] = bfs(idx[src])
        scores = cache[src]
        excl = {idx[src]} | {idx[nb] for nb in train_nb[src]}
        if idx[dst] not in scores:
            unreached += 1
            recalls.append(False)
            rrs.append(0.0)
            continue
        sd = scores[idx[dst]]
        rank = 1 + sum(1 for v, s in scores.items() if v not in excl and s > sd)
        recalls.append(rank <= 10)
        rrs.append(1.0 / rank)
    print(f"bfs         Recall@10: {np.mean(recalls):.3f}  MRR: {np.mean(rrs):.3f}  "
          f"(n={len(tests)}, unreached={unreached})")

    # --- full graph for production-style spot check ---
    Af = build_adj(nodes, edges, idx)
    Zf = train_node2vec(Af, len(nodes))
    bfs_f = make_bfs(Af, Zf, nodes)
    print(f"\n--- spot check (bfs, full graph, {time.time()-t0:.0f}s) ---")
    for sid in ["a:14", "a:212", "a:510"]:
        res = bfs_f(idx[sid])
        top = sorted(res.items(), key=lambda x: -x[1])[:5]
        print(f"* {labels[sid][:50]}")
        for v, s in top:
            print(f"    {s:.4f}  {labels[nodes[v]['id']][:60]}")


if __name__ == "__main__":
    main()
