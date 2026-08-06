"""Export published articles via the public PocketBase API (CI-friendly).

Produces the same per-article JSON files as export_articles.py (which reads a
copied data.db), so the rest of the pipeline is unchanged. Also writes
data/series.json so build_graph.py can run without the database file.
No credentials are needed — everything is public read.
"""
import html
import json
import re
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "articles"
API = "https://miseskorea.org/api"

EXPAND = "article_contributors_via_article.author,tags,board,category"


def get(url: str):
    req = urllib.request.Request(url, headers={"User-Agent": "index-refresh"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.load(r)


def fetch_all(collection: str, params: dict) -> list:
    items, page = [], 1
    while True:
        q = urllib.parse.urlencode({**params, "page": page, "perPage": 200})
        data = get(f"{API}/collections/{collection}/records?{q}")
        items += data["items"]
        if page >= data["totalPages"]:
            return items
        page += 1


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    articles = fetch_all("articles", {
        "filter": "(status='published')",
        "expand": EXPAND,
        "sort": "created",
    })

    for r in articles:
        ex = r.get("expand", {})
        board = ex.get("board")
        contribs = sorted(
            ex.get("article_contributors_via_article", []),
            key=lambda c: c.get("sort_order", 0),
        )
        doc = {
            "id": r["id"],
            "slug": r["slug"],
            "title": html.unescape(r["title"] or ""),
            "board": {"name": board["name"], "slug": board["slug"]} if board else None,
            "category": (ex.get("category") or {}).get("name"),
            "tags": [t["name"] for t in ex.get("tags", []) if t.get("name")],
            "contributors": [
                {
                    "name": c["expand"]["author"]["name"],
                    "slug": c["expand"]["author"].get("slug"),
                    "role": c.get("role"),
                }
                for c in contribs
                if c.get("expand", {}).get("author")
            ],
            "views": r.get("views", 0),
            "created": r["created"],
            "updated": r["updated"],
            "content": html.unescape(r["content"] or ""),
            "thumbnail": r.get("thumbnail"),
            "preview": r.get("preview"),
        }
        safe = re.sub(r"[^\w\-]", "_", r["slug"] or str(r["id"]))
        (OUT / f"{safe}.json").write_text(
            json.dumps(doc, ensure_ascii=False, indent=1), encoding="utf-8"
        )

    series = fetch_all("series", {})
    (ROOT / "data" / "series.json").write_text(
        json.dumps(
            [{"id": s["id"], "name": s.get("name"), "articles": s.get("articles", [])}
             for s in series],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(f"exported {len(articles)} articles + {len(series)} series via API")


if __name__ == "__main__":
    main()
