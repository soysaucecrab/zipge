"""Embed exported articles with multilingual-e5-base.

Chunks article bodies, embeds chunks (passage: prefix), and saves both
chunk-level and doc-level (normalized mean) vectors to data/embeddings/.
"""
import json
import re
import sys
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parent.parent
ARTICLES = ROOT / "data" / "articles"
OUT = ROOT / "data" / "embeddings"

MODELS = {
    "small": "intfloat/multilingual-e5-small",
    "base": "intfloat/multilingual-e5-base",
    "large": "intfloat/multilingual-e5-large",
}
MODEL_ALIAS = sys.argv[1] if len(sys.argv) > 1 else "base"
MODEL = MODELS[MODEL_ALIAS]
SUFFIX = "" if MODEL_ALIAS == "base" else f"_{MODEL_ALIAS}"
CHUNK_CHARS = 550     # ~0.45 tokens/char for Korean -> stays under e5's 512-token limit
HARD_SPLIT_CHARS = 800
MIN_CHUNK_CHARS = 50


def clean_markdown(md: str) -> str:
    text = re.sub(r"!\[[^\]]*\]\([^)]*\)", " ", md)          # images
    text = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", text)      # links -> anchor text
    text = re.sub(r"<[^>]+>", " ", text)                       # html tags
    text = re.sub(r"^[>#*\-\s|]+", "", text, flags=re.M)       # md decorations
    text = re.sub(r"[*_`~]{1,3}", "", text)
    return re.sub(r"\s+", " ", text).strip()


def chunk_text(text: str) -> list[str]:
    paras = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]
    chunks, buf = [], ""
    for p in paras:
        if len(buf) + len(p) > CHUNK_CHARS and buf:
            chunks.append(buf)
            buf = ""
        buf = f"{buf} {p}".strip()
        while len(buf) > HARD_SPLIT_CHARS:  # single huge paragraph
            chunks.append(buf[:HARD_SPLIT_CHARS])
            buf = buf[HARD_SPLIT_CHARS:]
    if len(buf) >= MIN_CHUNK_CHARS or not chunks:
        chunks.append(buf)
    return chunks


def main():
    docs = []
    for f in sorted(ARTICLES.glob("*.json")):
        d = json.loads(f.read_text(encoding="utf-8"))
        body = clean_markdown(d["content"] or "")
        raw_paras = re.split(r"(?:\r?\n){2,}", d["content"] or "")
        cleaned = [clean_markdown(p) for p in raw_paras]
        cleaned = [p for p in cleaned if p]
        chunks = chunk_text("\n\n".join(cleaned)) if cleaned else [""]
        docs.append({"id": d["id"], "slug": d.get("slug") or d["id"], "title": d["title"], "chunks": chunks, "body_len": len(body)})

    texts, chunk_doc_idx = [], []
    for i, d in enumerate(docs):
        for c in d["chunks"]:
            texts.append(f"passage: {d['title']}\n{c}")
            chunk_doc_idx.append(i)
    print(f"{len(docs)} docs, {len(texts)} chunks")

    model = SentenceTransformer(MODEL)
    vecs = model.encode(texts, batch_size=32, normalize_embeddings=True, show_progress_bar=True)
    vecs = np.asarray(vecs, dtype=np.float32)

    chunk_doc_idx = np.asarray(chunk_doc_idx)
    doc_vecs = np.zeros((len(docs), vecs.shape[1]), dtype=np.float32)
    for i in range(len(docs)):
        m = vecs[chunk_doc_idx == i].mean(axis=0)
        doc_vecs[i] = m / np.linalg.norm(m)

    OUT.mkdir(parents=True, exist_ok=True)
    ids = np.array([d["id"] for d in docs])
    np.savez_compressed(OUT / f"doc_embeddings{SUFFIX}.npz", ids=ids, vectors=doc_vecs)
    np.savez_compressed(OUT / f"chunk_embeddings{SUFFIX}.npz", doc_idx=chunk_doc_idx, vectors=vecs, ids=ids)
    meta = [{"id": d["id"], "slug": d["slug"], "title": d["title"], "n_chunks": len(d["chunks"])} for d in docs]
    (OUT / "doc_meta.json").write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")
    print(f"saved to {OUT}")


if __name__ == "__main__":
    main()
