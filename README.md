# zipge (집게)

서버 없이 **브라우저에서 완결되는 한국어 하이브리드 검색엔진**.
RAM 1GB 서버에서 실서비스로 검증된 구조를 라이브러리 + 파이프라인으로 추린 것이다.

이름은 게의 **집게**에서 왔다 — 문서 더미에서 원하는 것을 집어 올린다.
[garu](https://github.com/ongjin/garu)가 텍스트를 가루로 갈면, zipge가 집어낸다.
**zip**은 덤이다: 모델 가지치기(118MB→37MB)와 int8 양자화로 눌러 담았다.

- **리터럴 → 형태소 → 시맨틱** 계층 융합: 문자 그대로 일치가 항상 추론을 이긴다
- **한국어 오타 교정**: 자모 편집거리 + 코퍼스 어휘 사전 ("쎄금"→"세금", "사유재솬"→"사유재산")
- **시맨틱 검색**: 어휘 가지치기한 multilingual-e5-small (118MB → **37MB**, 임베딩 공간 손실 0) 을 transformers.js 로 브라우저에서 실행 (WebGPU → WASM 강등 사다리)
- **WASM 차단 환경(락다운 모드) 폴백**: 미리 계산한 사전 + 어휘 벡터 합성 임베딩으로 순수 JS 동작
- **메모리 관리**: N회 검색마다 워커 재활용, 유휴 시 해제 — WASM 힙 누적 차단
- 문서 그래프(node2vec) 기반 **관련 글 추천** 인덱스까지 오프라인 파이프라인으로 생성

## 구성

```
src/        TS 라이브러리 (프레임워크 무관: fetch·Web Worker 표준 API만 사용)
pipeline/   Python 인덱스 빌더 (임베딩·그래프·관련 글·가지치기 모델·폴백 자산)
examples/   실서비스(miseskorea.org) 어댑터 — PocketBase export, CI 자동 재학습 등
docs/       아키텍처 문서
```

## 빠른 시작

### 1. 인덱스 만들기 (오프라인, Python)

문서를 아래 스키마의 JSON 파일들로 준비한다 (`data/articles/*.json`):

```jsonc
{
  "id": "고유값", "slug": "url-slug", "title": "제목", "content": "본문(markdown/text)",
  "board": {"name": "게시판", "slug": "board"},   // 없으면 null
  "category": "분류", "tags": ["태그"],
  "contributors": [{"name": "저자", "slug": "", "role": "author"}],
  "views": 0, "created": "2024-01-01 00:00:00.000Z", "updated": "...",
  "thumbnail": "파일명 또는 null", "preview": "요약 또는 null"
}
```

```bash
pip install -r pipeline/requirements.txt
python pipeline/embed_articles.py small     # 문서 임베딩 (e5-small)
python pipeline/embed_articles.py base      # 그래프 kNN용 (관련 글을 쓸 때만)
python pipeline/build_graph.py              # 문서 그래프 (관련 글용)
cd pipeline && python build_index.py        # 검색 인덱스 + 관련 글 top-k
python pipeline/prune_model.py              # 자기 코퍼스 기준 모델 가지치기 → 37MB ONNX
node pipeline/build_terms.js                # 폴백 사전
python pipeline/build_fallback_vectors.py   # 폴백 어휘 벡터
```

산출물을 정적 경로에 배치한다:

```
/search-index/{doc_meta.json, doc_vectors_i8.bin, vector_scales.json, tags.json, terms.json, vocab_*.bin/json}
/models/e5-small-ko/{config.json, tokenizer.json, onnx/model_quantized.onnx}
```

### 2. 브라우저에서 검색 (TS)

```ts
import { SearchController } from 'zipge';

const search = new SearchController({
  indexBase: '/search-index',
  worker: { localModelPath: '/models', localModel: 'e5-small-ko' }
});

search.onResults = ({ results, tier, correctedTo }) => render(results);
search.onStatus = (status) => showSpinner(status === 'loading');

input.addEventListener('focus', () => search.warmup()); // 미리 데우기
form.addEventListener('submit', () => search.search(input.value));
```

워커 파일은 번들러가 처리한다 (Vite/webpack: `new URL('zipge/src/embed-worker.ts', import.meta.url)` 패턴이 라이브러리 내부에 이미 들어 있으므로 소스째 소비하는 것을 권장).

## 동작 원리 (요약)

쿼리 하나가 들어오면:

1. **리터럴** — 제목에 통째로 포함되면 무조건 1위, 교정 비활성
2. **형태소 (T0)** — garu(WASM)로 명사 추출, 자모 편집거리로 오타 흡수
3. **시맨틱 (T1)** — e5 임베딩 코사인. 오타는 T0가 교정한 문장으로 재임베딩
4. **융합** — 키워드 증거가 강한데 시맨틱과 겹침이 없으면 키워드 주도(임베딩 노이즈 방어), 그 외엔 시맨틱 주도
5. **연관도 게이트** — 기준 미달 결과는 표시하지 않음

실측 성능 (785편 코퍼스, 원본 글 검색 Recall@10): 정상 0.88 / 오타 0.87 /
띄어쓰기 오류 0.80 / 붙여쓰기 0.87.

상세: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)

## 왜 이렇게 만들었나

호스팅 서버가 RAM 756MB였다. 서버에 검색 엔진도, 임베딩 모델도 올릴 수
없어서 전부 사용자 브라우저로 보냈다 — 대신 다운로드를 37MB까지 깎고,
안 쓰면 아무것도 받지 않게 하고, WASM이 막힌 브라우저까지 동작하게 만들었다.
결과적으로 서버 비용 0원의 시맨틱 검색이 됐다.

## 라이선스

**Apache-2.0** — 재배포·파생물은 `NOTICE`의 제작자 표기
(KIM ZINU / soysaucecrab)를 유지해야 합니다.
의존: [garu](https://github.com/ongjin/garu)(MIT),
transformers.js(Apache-2.0), multilingual-e5(MIT).
