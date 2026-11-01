"""Shared pieces of the student encoder: slotting rules, asset format, numpy runtime, metrics.

The numpy runtime (StudentRuntime) is the reference implementation of the browser runtime
(src/core.ts studentQuery, spec in docs/STUDENT.md) — evaluate.py scores the exported assets
through it, so what gets measured is exactly what ships.
"""
import json
import math
from pathlib import Path

import numpy as np

MAX_SLOTS = 16
FALLBACK_MIN_LEN = 3
FALLBACK_MIN_RATIO = 0.5

ASSET_FILES = {
    "meta": "student_meta.json",
    "vocab": "student_vocab_i8.bin",
    "proj": "student_proj_i8.bin",
    "params": "student_params_f32.bin",
    "docs": "student_doc_vectors_i8.bin",
    "doc_scales": "student_doc_scales.json",
}
# Out-of-corpus extension lexicon (build_lexicon.py), fetched shard by shard on demand.
# Shards are concatenated into a few pack files read with HTTP Range requests, so a host sees
# ~22 files instead of 256; packs stay at 4MB because some asset stores reject large uploads.
LEX_META = "student_lex_meta.json"
LEX_PACK = "student_lex_{}.bin"
LEX_SHARDS = 256
LEX_PACK_BYTES = 4 << 20


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, obj):
    Path(path).write_text(json.dumps(obj, ensure_ascii=False), encoding="utf-8")


def make_idf(vocab_json):
    """IDF from the full noun DF table: ln((N+1)/(df+1)) + 1 (unknown terms get the max)."""
    n, df = vocab_json["n_docs"], vocab_json["df"]
    return lambda t: math.log((n + 1) / (df.get(t, 0) + 1)) + 1.0


def fallback(term, vi):
    """Longest substring of an out-of-vocabulary token that is in the vocabulary
    and covers at least half of it (left-most first at equal length)."""
    n = len(term)
    for length in range(n - 1, 1, -1):
        if length / n < FALLBACK_MIN_RATIO:
            break
        for s in range(n - length + 1):
            sub = term[s:s + length]
            if sub in vi:
                return sub
    return None


def token_kinds(it):
    """Token kinds from build_queries.mjs: n = garu noun (or whitespace piece when garu finds
    none), e = eojeol / particle-stripped eojeol, j = joined whole query. Items written before
    kinds existed fall back to the old rule (joined vs everything else)."""
    if "kinds" in it:
        return it["kinds"]
    joined = "".join(it["text"].split()).lower()
    return ["j" if t == joined else "n" for t in it["toks"]]


def build_slots(items, vi, idf=None, use_fallback=True, max_slots=MAX_SLOTS, lex=None):
    """Map each query's tokens to slots (docs/STUDENT.md, same rule as src/core.ts studentQuery).

    Returns idx [n, L] int64, mask [n, L] bool, fb [n, L] float32 (1 = substring fallback),
    ext [n, L] bool (1 = extension-lexicon slot; idx is then the lexicon row),
    idf [n, L] float32 (0 when idf is None). Per token, in order, at most max_slots slots:
      1. exact core-vocabulary hit
      2. exact extension-lexicon hit (lex: term -> row), when a lexicon is given
      3. noun tokens only (kind n, >= 3 chars): longest in-vocabulary substring covering
         >= half the token -> fallback slot
      a slot whose (table, row) is already used is skipped
    """
    n = len(items)
    idx = np.zeros((n, max_slots), np.int64)
    mask = np.zeros((n, max_slots), bool)
    fb = np.zeros((n, max_slots), np.float32)
    ext = np.zeros((n, max_slots), bool)
    fidf = np.zeros((n, max_slots), np.float32)
    for r, it in enumerate(items):
        used, c = set(), 0
        for t, kind in zip(it["toks"], token_kinds(it)):
            if c >= max_slots:
                break
            if t in vi:
                key, term, is_fb = ("c", vi[t]), t, 0.0
            elif lex is not None and t in lex:
                key, term, is_fb = ("x", lex[t]), t, 0.0
            elif use_fallback and kind == "n" and len(t) >= FALLBACK_MIN_LEN:
                term = fallback(t, vi)
                if term is None:
                    continue
                key, is_fb = ("c", vi[term]), 1.0
            else:
                continue
            if key in used:
                continue
            used.add(key)
            idx[r, c], mask[r, c], fb[r, c], ext[r, c] = key[1], True, is_fb, key[0] == "x"
            if idf is not None:
                fidf[r, c] = idf(term)
            c += 1
    return idx, mask, fb, ext, fidf


# ---------------- extension lexicon shards

def fnv1a_utf16(term):
    """FNV-1a 32-bit over UTF-16 code units — src/core.ts hashes JS strings the same way."""
    h = 0x811C9DC5
    b = term.encode("utf-16-le")
    for i in range(0, len(b), 2):
        h ^= b[i] | (b[i + 1] << 8)
        h = (h * 0x01000193) & 0xFFFFFFFF
    return h


def write_lexicon_shards(out_dir, terms, q, scale, *, basis_sha1, extra=None, n_shards=LEX_SHARDS,
                         pack_bytes=LEX_PACK_BYTES):
    """Shard i = terms with fnv1a_utf16(term) % n_shards == i. Each shard (little-endian):
    u32 count, u32 json_len, utf-8 JSON {"terms": [...], "scales": [...]}, int8 [count][dim].
    Consecutive shards are concatenated into packs of at most pack_bytes; the meta's
    index[i] = [pack, start, length] locates shard i for a Range request."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for old in out.glob(LEX_PACK.format("*")):
        old.unlink()
    buckets = [[] for _ in range(n_shards)]
    for i, t in enumerate(terms):
        buckets[fnv1a_utf16(t) % n_shards].append(i)
    blobs = []
    for rows in buckets:
        head = json.dumps({"terms": [terms[i] for i in rows], "scales": [round(float(scale[i]), 8) for i in rows]},
                          ensure_ascii=False).encode("utf-8")
        body = q[rows].astype(np.int8).tobytes() if rows else b""
        blobs.append(np.asarray([len(rows), len(head)], "<u4").tobytes() + head + body)
    packs, index, cur = [], [], []
    for blob in blobs:
        if cur and sum(map(len, cur)) + len(blob) > pack_bytes:
            packs.append(cur)
            cur = []
        index.append([len(packs), sum(map(len, cur)), len(blob)])
        cur.append(blob)
    packs.append(cur)
    names = [LEX_PACK.format(k) for k in range(len(packs))]
    for name, parts in zip(names, packs):
        (out / name).write_bytes(b"".join(parts))
    write_json(out / LEX_META, {"version": 2, "dim": int(q.shape[1]), "count": len(terms), "shards": n_shards,
                                "hash": "fnv1a32-utf16", "basis_sha1": basis_sha1, "packs": names, "index": index,
                                **(extra or {})})
    return sum(map(len, blobs))


def read_lexicon_shards(index_dir):
    """-> (terms list, q int8 [L, dim], scale [L], meta) or None when no lexicon is shipped."""
    d = Path(index_dir)
    if not (d / LEX_META).exists():
        return None
    meta = load_json(d / LEX_META)
    packs = [(d / name).read_bytes() for name in meta["packs"]]
    terms, qs, scales = [], [], []
    for pack, start, length in meta["index"]:
        raw = packs[pack][start:start + length]
        count, jl = np.frombuffer(raw[:8], "<u4")
        head = json.loads(raw[8:8 + jl].decode("utf-8"))
        terms += head["terms"]
        scales += head["scales"]
        qs.append(np.frombuffer(raw[8 + jl:], np.int8).reshape(int(count), meta["dim"]))
    return terms, np.concatenate(qs) if qs else np.zeros((0, meta["dim"]), np.int8), np.asarray(scales, np.float32), meta


# ---------------- int8 helpers

def quantize_rows(x):
    """Per-row symmetric int8: x ≈ q * scale[:, None]."""
    scale = np.abs(x).max(axis=1) / 127.0
    scale[scale == 0] = 1.0
    q = np.clip(np.round(x / scale[:, None]), -127, 127).astype(np.int8)
    return q, scale.astype(np.float32)


def read_i8(path, cols):
    return np.frombuffer(Path(path).read_bytes(), dtype=np.int8).reshape(-1, cols)


def load_doc_vectors(index_dir, bin_name, scales_name):
    sc = load_json(Path(index_dir) / scales_name)
    q = read_i8(Path(index_dir) / bin_name, sc["dim"]).astype(np.float32)
    return q * np.asarray(sc["scales"], np.float32)[:, None]


# ---------------- deployed student assets

def write_assets(out_dir, *, terms, vocab_q, members, docs, doc_ids, teacher, config, extra=None):
    """members: list of dicts with term_scale [V], proj_w [out, dim] float, proj_b [out], fb_weight float,
    ext_weight float (weight of extension-lexicon slots, times the lexicon row's int8 scale).
    Layout (little-endian):
      student_vocab_i8.bin      int8    [V][dim]
      student_proj_i8.bin       int8    [M][out_dim][dim]
      student_params_f32.bin    float32 per member: term_scale[V], proj_scale[out_dim], proj_bias[out_dim],
                                fb_weight[1], ext_weight[1]
      student_doc_vectors_i8.bin int8   [N][out_dim]  (doc_meta row order)
      student_doc_scales.json   {dim, scales[N]}
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    V, dim = vocab_q.shape
    out_dim = members[0]["proj_w"].shape[0]
    (out / ASSET_FILES["vocab"]).write_bytes(vocab_q.astype(np.int8).tobytes())
    proj_blocks, param_blocks = [], []
    for m in members:
        pq, ps = quantize_rows(m["proj_w"])
        proj_blocks.append(pq.tobytes())
        param_blocks.append(np.concatenate([
            np.asarray(m["term_scale"], np.float32), ps,
            np.asarray(m["proj_b"], np.float32), np.asarray([m["fb_weight"], m.get("ext_weight", 0.0)], np.float32),
        ]).astype("<f4").tobytes())
    (out / ASSET_FILES["proj"]).write_bytes(b"".join(proj_blocks))
    (out / ASSET_FILES["params"]).write_bytes(b"".join(param_blocks))
    dq, dsc = quantize_rows(docs)
    (out / ASSET_FILES["docs"]).write_bytes(dq.tobytes())
    # same shape as the search index's vector_scales.json
    write_json(out / ASSET_FILES["doc_scales"], {"model": teacher, "dim": int(out_dim), "count": int(len(dsc)),
                                                  "scales": [round(float(s), 8) for s in dsc]})
    meta = {
        "version": 2,
        "teacher": teacher,
        "dim": int(dim),
        "out_dim": int(out_dim),
        "members": len(members),
        "vocab_size": int(V),
        "max_slots": MAX_SLOTS,
        "fallback": {"min_len": FALLBACK_MIN_LEN, "min_ratio": FALLBACK_MIN_RATIO},
        "params_layout": ["term_scale[vocab_size]", "proj_scale[out_dim]", "proj_bias[out_dim]", "fb_weight[1]",
                          "ext_weight[1]"],
        "terms": list(terms),
        "doc_ids": [str(d) for d in doc_ids],
        "config": config,
        **(extra or {}),
    }
    write_json(out / ASSET_FILES["meta"], meta)
    return {f: (out / f).stat().st_size for f in ASSET_FILES.values()}


class StudentRuntime:
    """Reference implementation of the browser runtime, reading the shipped files."""

    def __init__(self, index_dir, lexicon_dir=None):
        d = Path(index_dir)
        self.meta = load_json(d / ASSET_FILES["meta"])
        dim, out_dim, V, M = self.meta["dim"], self.meta["out_dim"], self.meta["vocab_size"], self.meta["members"]
        self.vi = {t: i for i, t in enumerate(self.meta["terms"])}
        self.vocab = read_i8(d / ASSET_FILES["vocab"], dim).astype(np.float32)
        proj = np.frombuffer((d / ASSET_FILES["proj"]).read_bytes(), np.int8).reshape(M, out_dim, dim).astype(np.float32)
        tail = 2 if self.meta.get("version", 1) >= 2 else 1  # v2 adds ext_weight after fb_weight
        params = np.frombuffer((d / ASSET_FILES["params"]).read_bytes(), "<f4").reshape(M, V + 2 * out_dim + tail)
        self.members = []
        for m in range(M):
            p = params[m]
            self.members.append({
                "term_scale": p[:V],
                "W": proj[m] * p[V:V + out_dim][:, None],
                "b": p[V + out_dim:V + 2 * out_dim],
                "fb_weight": float(p[V + 2 * out_dim]),
                "ext_weight": float(p[V + 2 * out_dim + 1]) if tail == 2 else 0.0,
            })
        self.docs = load_doc_vectors(d, ASSET_FILES["docs"], ASSET_FILES["doc_scales"])
        # extension lexicon: only when its input basis is the one this student was trained on
        self.lex, self.lex_vec = None, None
        lx = read_lexicon_shards(lexicon_dir or d)
        if lx is not None and tail == 2 and lx[3].get("basis_sha1") == self.meta.get("basis_sha1"):
            terms, q, scale, _ = lx
            self.lex = {t: i for i, t in enumerate(terms)}
            self.lex_vec = q.astype(np.float32) * scale[:, None]

    def encode(self, items, use_lexicon=True):
        """-> (Q [n, out_dim] unit rows, valid [n] bool, exact [n] int). Invalid = no slot (keyword-only);
        exact = slots that hit the core vocabulary or the lexicon as-is (the runtime gate keys on it)."""
        lex = self.lex if use_lexicon else None
        idx, mask, fb, ext, _ = build_slots(items, self.vi, lex=lex)
        valid = mask.any(1)
        core_idx = np.where(ext, 0, idx)
        Vc = self.vocab[core_idx]
        Vx = self.lex_vec[np.where(ext, idx, 0)] if lex is not None and len(self.lex_vec) else np.zeros_like(Vc)
        Vrow = np.where(ext[..., None], Vx, Vc)  # [n, L, dim]
        Q = np.zeros((len(items), self.meta["out_dim"]), np.float32)
        for m in self.members:
            w = np.where(ext, m["ext_weight"], m["term_scale"][core_idx] * np.where(fb > 0, m["fb_weight"], 1.0)) * mask
            p = (w[..., None] * Vrow).sum(1)
            q = p @ m["W"].T + m["b"]
            Q += q / np.maximum(np.linalg.norm(q, axis=1, keepdims=True), 1e-12)
        Q /= np.maximum(np.linalg.norm(Q, axis=1, keepdims=True), 1e-12)
        exact = ((fb == 0) & mask).sum(1)
        return Q, valid, exact


# ---------------- metrics

def ranks(Q, valid, items, D):
    """Best (0-based) rank of any relevant doc = docs scoring strictly higher; inf if invalid."""
    S = Q @ D.T
    out = []
    for i, it in enumerate(items):
        if not valid[i]:
            out.append(math.inf)
            continue
        out.append(min(int((S[i] > S[i, r]).sum()) for r in it["rows"]))
    return out


def metrics(rk):
    n = len(rk)
    if n == 0:
        return {"n": 0, "r1": 0.0, "r5": 0.0, "r10": 0.0, "r20": 0.0, "mrr": 0.0}
    rec = lambda k: 100.0 * sum(r < k for r in rk) / n
    return {"n": n, "r1": rec(1), "r5": rec(5), "r10": rec(10), "r20": rec(20),
            "mrr": sum(1.0 / (r + 1) for r in rk) / n}


def bootstrap_ci(rk, stat, B=2000, seed=0):
    a = np.asarray(rk, dtype=float)
    rng = np.random.default_rng(seed)
    vals = [stat(a[rng.integers(0, len(a), len(a))]) for _ in range(B)]
    return [float(x) for x in np.percentile(vals, [2.5, 97.5])]


mrr_of = lambda a: float(np.mean(1.0 / (a + 1)))
r10_of = lambda a: float(np.mean(a < 10) * 100)
