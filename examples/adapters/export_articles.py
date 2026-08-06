"""Export published articles from PocketBase data.db to per-article JSON files."""
import html
import json
import re
import sqlite3
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DB = ROOT / "data" / "data.db"
OUT = ROOT / "data" / "articles"

conn = sqlite3.connect(DB)
conn.row_factory = sqlite3.Row

boards = {r["id"]: {"name": r["name"], "slug": r["slug"]} for r in conn.execute("SELECT * FROM boards")}
tags = {r["id"]: r["name"] for r in conn.execute("SELECT * FROM tags")}
categories = {r["id"]: r["name"] for r in conn.execute("SELECT * FROM categories")}
authors = {r["id"]: {"name": r["name"], "slug": r["slug"]} for r in conn.execute("SELECT * FROM authors")}

contribs = {}
for r in conn.execute("SELECT article, author, role, sort_order FROM article_contributors ORDER BY sort_order"):
    a = authors.get(r["author"])
    if a:
        contribs.setdefault(r["article"], []).append({"name": a["name"], "slug": a["slug"], "role": r["role"]})

OUT.mkdir(parents=True, exist_ok=True)
n = 0
for r in conn.execute("SELECT * FROM articles WHERE status='published'"):
    doc = {
        "id": r["id"],
        "slug": r["slug"],
        "title": html.unescape(r["title"] or ""),
        "board": boards.get(r["board"]),
        "category": categories.get(r["category"]),
        "tags": [tags[t] for t in json.loads(r["tags"] or "[]") if t in tags],
        "thumbnail": r["thumbnail"],
        "preview": r["preview"],
        "contributors": contribs.get(r["id"], []),
        "views": r["views"],
        "created": r["created"],
        "updated": r["updated"],
        "content": html.unescape(r["content"] or ""),
    }
    safe = re.sub(r"[^\w\-]", "_", r["slug"] or str(r["id"]))
    (OUT / f"{safe}.json").write_text(json.dumps(doc, ensure_ascii=False, indent=1), encoding="utf-8")
    n += 1

print(f"exported {n} articles to {OUT}")
