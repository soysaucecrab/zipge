"""Build a heterogeneous document graph from exported articles + embeddings.

Nodes: articles + attribute nodes (author/tag/category/series/board).
Edges: article->attribute (bipartite), article->article in-body links (uid and
slug based), and text-kNN edges from doc embeddings.
Output: data/graph/nodes.json, data/graph/edges.jsonl
"""
import json
import re
import sqlite3
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
ARTICLES = ROOT / "data" / "articles"
EMB = ROOT / "data" / "embeddings"
OUT = ROOT / "data" / "graph"

KNN_K = 10
KNN_MIN_SIM = 0.82

LINK_RE = re.compile(r"https?://(?:www\.)?miseskorea\.org/([^\s)\"'<>\]]*)")
UID_RE = re.compile(r"[?&]uid=(\d+)")


def main():
    docs = {}
    for f in sorted(ARTICLES.glob("*.json")):
        d = json.loads(f.read_text(encoding="utf-8"))
        docs[d["id"]] = d
    by_slug = {d["slug"]: i for i, d in docs.items() if d["slug"]}

    nodes, edges = {}, []

    def add_node(nid, ntype, label):
        nodes.setdefault(nid, {"id": nid, "type": ntype, "label": label})

    def add_edge(src, dst, etype, weight=1.0):
        edges.append({"src": src, "dst": dst, "type": etype, "weight": round(weight, 4)})

    for aid, d in docs.items():
        add_node(f"a:{aid}", "article", d["title"])

    # --- attribute edges ---
    for aid, d in docs.items():
        a = f"a:{aid}"
        for c in d["contributors"]:
            nid = f"author:{c['slug'] or c['name']}"
            add_node(nid, "author", c["name"])
            add_edge(a, nid, "authored_by")
        for t in d["tags"]:
            add_node(f"tag:{t}", "tag", t)
            add_edge(a, f"tag:{t}", "tagged")
        if d["category"]:
            add_node(f"cat:{d['category']}", "category", d["category"])
            add_edge(a, f"cat:{d['category']}", "in_category")
        if d["board"]:
            add_node(f"board:{d['board']['slug']}", "board", d["board"]["name"])
            add_edge(a, f"board:{d['board']['slug']}", "in_board")

    # --- series edges (로컬은 data.db, CI는 API가 만든 series.json) ---
    db_path = ROOT / "data" / "data.db"
    series_path = ROOT / "data" / "series.json"
    if db_path.exists():
        conn = sqlite3.connect(db_path)
        series_rows = [(sid, name, json.loads(arts or "[]"))
                       for sid, name, arts in conn.execute("SELECT id, name, articles FROM series")]
    else:
        series_rows = [(s["id"], s["name"], s.get("articles", []))
                       for s in json.loads(series_path.read_text(encoding="utf-8"))]
    for sid, name, members_raw in series_rows:
        members = [x for x in members_raw if x in docs]
        if len(members) >= 2:
            add_node(f"series:{sid}", "series", name)
            for m in members:
                add_edge(f"a:{m}", f"series:{sid}", "in_series")

    # --- in-body internal links ---
    n_link, n_unresolved = 0, 0
    for aid, d in docs.items():
        targets = set()
        for m in LINK_RE.finditer(d["content"] or ""):
            path = m.group(0)
            if "/api/files" in path:
                continue
            uid = UID_RE.search(path)
            if uid and uid.group(1) in docs:
                targets.add(uid.group(1))
            elif uid:
                n_unresolved += 1
            else:
                slug = m.group(1).strip("/").split("?")[0].split("/")[-1]
                if slug in by_slug and by_slug[slug] != aid:
                    targets.add(by_slug[slug])
                elif slug:
                    n_unresolved += 1
        for t in targets - {aid}:
            add_edge(f"a:{aid}", f"a:{t}", "links_to")
            n_link += 1

    # --- text-kNN edges ---
    emb = np.load(EMB / "doc_embeddings.npz", allow_pickle=True)
    ids, vecs = list(emb["ids"]), emb["vectors"]
    sims = vecs @ vecs.T
    np.fill_diagonal(sims, -1)
    seen = set()
    for i in range(len(ids)):
        for j in np.argsort(-sims[i])[:KNN_K]:
            if sims[i, j] < KNN_MIN_SIM:
                break
            key = (min(i, j), max(i, j))
            if key not in seen:
                seen.add(key)
                add_edge(f"a:{ids[i]}", f"a:{ids[j]}", "knn", float(sims[i, j]))

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "nodes.json").write_text(json.dumps(list(nodes.values()), ensure_ascii=False), encoding="utf-8")
    with (OUT / "edges.jsonl").open("w", encoding="utf-8") as f:
        for e in edges:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")

    print(f"nodes: {len(nodes)} ({Counter(n['type'] for n in nodes.values())})")
    print(f"edges: {len(edges)} ({Counter(e['type'] for e in edges)})")
    print(f"link edges: {n_link}, unresolved links: {n_unresolved}")
    deg = Counter()
    for e in edges:
        deg[e["src"]] += 1
        deg[e["dst"]] += 1
    isolated = [n for n in nodes if n.startswith("a:") and deg[n] == 0]
    print(f"isolated articles: {len(isolated)}")


if __name__ == "__main__":
    main()
