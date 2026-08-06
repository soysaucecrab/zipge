"""락다운 폴백용 어휘 벡터 — 사전의 단어들을 e5로 임베딩해 int8로 싣는다.

브라우저는 (모델 로딩이 실패했을 때만) 이 표를 받아 쿼리 단어 벡터들의
평균으로 임베딩을 합성한다. 문서 벡터와 같은 e5 공간이므로 기존 인덱스를
그대로 쓴다. terms.json의 단어 + 제목 인접 결합형(교정 사전과 동일)을 수록.

Run (Mises 작업 폴더에서): .venv/bin/python miseskorea-frontend/pipeline/build_fallback_vectors.py
"""
import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

FRONT = Path(__file__).resolve().parent.parent
IDX = FRONT / "static" / "search-index"


def main():
    terms = json.loads((IDX / "terms.json").read_text(encoding="utf-8"))["docs"]
    single = [w for d in terms for w in d["t"] + d["r"]]
    bigrams = [d["t"][i] + d["t"][i + 1] for d in terms for i in range(len(d["t"]) - 1)]
    vocab = sorted(set(single + bigrams))
    print(f"vocab: {len(vocab)} terms")

    model = SentenceTransformer("intfloat/multilingual-e5-small")
    V = np.asarray(
        model.encode([f"query: {w}" for w in vocab], batch_size=64,
                     normalize_embeddings=True, show_progress_bar=False),
        dtype=np.float32,
    )
    scales = np.abs(V).max(axis=1) / 127.0
    Q = np.clip(np.round(V / scales[:, None]), -127, 127).astype(np.int8)

    (IDX / "vocab_vectors_i8.bin").write_bytes(Q.tobytes())
    (IDX / "vocab_meta.json").write_text(
        json.dumps({"dim": int(V.shape[1]), "terms": vocab,
                    "scales": [round(float(s), 8) for s in scales]}, ensure_ascii=False),
        encoding="utf-8",
    )
    kb = (IDX / "vocab_vectors_i8.bin").stat().st_size / 1024
    print(f"vocab_vectors_i8.bin {kb:.0f}KB + vocab_meta.json")


if __name__ == "__main__":
    main()
