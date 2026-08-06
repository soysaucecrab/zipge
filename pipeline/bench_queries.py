"""Generate the tier benchmark query set.

For each sampled article, take 2 informative title keywords and emit 4
variants: clean / typo (1-jamo substitution) / split (space inside a word) /
merged (keywords joined without space). Target = the source article.
Output: data/bench/queries.json
"""
import json
import random
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "bench"
SEED = 7
N_ARTICLES = 300

CHO = "ㄱㄲㄴㄷㄸㄹㅁㅂㅃㅅㅆㅇㅈㅉㅊㅋㅌㅍㅎ"
JUNG = "ㅏㅐㅑㅒㅓㅔㅕㅖㅗㅘㅙㅚㅛㅜㅝㅞㅟㅠㅡㅢㅣ"


def inject_typo(word: str, rng: random.Random) -> str:
    idxs = [i for i, ch in enumerate(word) if 0 <= ord(ch) - 0xAC00 <= 11171]
    if not idxs:
        return word
    i = rng.choice(idxs)
    code = ord(word[i]) - 0xAC00
    cho, jung, jong = code // 588, (code % 588) // 28, code % 28
    if rng.random() < 0.5:
        jung = rng.choice([j for j in range(21) if j != jung])
    else:
        cho = rng.choice([c for c in range(19) if c != cho])
    return word[:i] + chr(0xAC00 + cho * 588 + jung * 28 + jong) + word[i + 1 :]


def keywords(title: str) -> list[str]:
    text = re.sub(r"\[[^\]]*\]|\([^)]*\)|[·:|,ㅣ?!'\"“”‘’]", " ", title)
    toks = [t for t in text.split() if len(t) >= 2 and re.search(r"[가-힣]", t)]
    toks.sort(key=len, reverse=True)
    return toks[:2]


def main():
    rng = random.Random(SEED)
    docs = []
    for f in sorted((ROOT / "data" / "articles").glob("*.json")):
        d = json.loads(f.read_text(encoding="utf-8"))
        kws = keywords(d["title"])
        if len(kws) == 2:
            docs.append({"id": d["id"], "kws": kws})
    rng.shuffle(docs)
    docs = docs[:N_ARTICLES]

    queries = []
    for d in docs:
        a, b = d["kws"]
        clean = f"{a} {b}"
        typo_word = inject_typo(a, rng)
        split_at = len(a) // 2 or 1
        variants = {
            "clean": clean,
            "typo": f"{typo_word} {b}",
            "split": f"{a[:split_at]} {a[split_at:]} {b}",
            "merged": f"{a}{b}",
        }
        for variant, text in variants.items():
            queries.append({"article": d["id"], "variant": variant, "text": text})

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "queries.json").write_text(json.dumps(queries, ensure_ascii=False, indent=0),
                                      encoding="utf-8")
    print(f"{len(docs)} articles, {len(queries)} queries -> {OUT/'queries.json'}")
    for q in queries[:4]:
        print("  ", q["variant"], ":", q["text"])


if __name__ == "__main__":
    main()
