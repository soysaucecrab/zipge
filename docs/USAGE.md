# zipge 사용 방법

zipge는 두 부분으로 이루어진다.

1. **파이프라인 (Python + Node, 오프라인)** — 문서를 임베딩·인덱싱해서 정적 파일을 만든다
2. **런타임 (TS, 브라우저)** — 그 정적 파일만으로 검색을 수행한다. 서버 연산 없음

```
[내 문서들] → 파이프라인 → [정적 자산] → 정적 호스팅 → 브라우저에서 zipge 런타임
```

---

## 1. 설치

```bash
npm install zipge @huggingface/transformers
npm install garu-ko        # 선택: 형태소 매칭·오타 교정 품질 향상 (권장)
```

- `@huggingface/transformers` — 시맨틱 검색(임베딩 모델 실행)에 필요 (peer)
- `garu-ko` — 한국어 형태소 분석기(WASM). 없거나 로딩에 실패하면 zipge는
  자동으로 규칙 기반 폴백(조사 제거 + 미리 계산된 사전)으로 동작한다

파이프라인 쪽:

```bash
pip install -r pipeline/requirements.txt   # sentence-transformers, gensim, scipy 등
```

---

## 2. 인덱스 만들기 (파이프라인)

### 2-1. 입력 스키마

문서를 `data/articles/*.json` (파일당 글 1편)으로 준비한다.
**필수는 3개뿐이다:**

```jsonc
{
  "id": "고유값(문자열)",
  "title": "제목",
  "content": "본문 — markdown 또는 평문"
}
```

이것만으로 시맨틱 검색·키워드 검색·오타 교정·관련 글(텍스트 kNN 기반)이
전부 동작한다. 아래 선택 필드는 있으면 그만큼 기능이 풍부해진다:

| 선택 필드 | 형태 | 있으면 좋아지는 것 | 기본값 |
|---|---|---|---|
| `slug` | `"url-slug"` | 결과 링크 경로 | `id` 사용 |
| `board` | `{"name","slug"}` | 게시판 필터·그래프 엣지 | 없음 |
| `category` | `"분류명"` | 분류 필터 | 없음 |
| `tags` | `["태그"]` | 태그 검색·칩·그래프 엣지 | `[]` |
| `contributors` | `[{"name","slug","role"}]` | 저자 표시·필터·그래프 엣지 | `[]` |
| `created` | `"YYYY-MM-DD ..."` | 날짜 표시·정렬 | epoch |
| `views` | 숫자 | 인기순 정렬 | `0` |
| `thumbnail` / `preview` | 문자열 | 결과 목록 썸네일·미리보기 | 없음 |

파이프라인이 결측 필드를 위 기본값으로 정규화하므로 런타임 쪽 처리는
동일하다. `examples/adapters/`에 PocketBase에서 전체 스키마로 뽑아내는
실전 어댑터가 있다 — 어떤 CMS든 이 형태로만 맞추면 된다.

### 2-2. 실행 순서

작업 폴더 배치: `<루트>/pipeline/`(이 레포의 pipeline 복사) + `<루트>/data/articles/`.

```bash
# ① 문서 임베딩 — 검색용(small)은 필수
python pipeline/embed_articles.py small

# ② 관련 글 추천을 쓸 경우에만: 그래프용 임베딩 + 그래프 + (③에 포함됨)
python pipeline/embed_articles.py base
python pipeline/build_graph.py

# ③ 검색 인덱스 (+ 그래프가 있으면 관련 글 top-k까지)
cd pipeline && python build_index.py && cd ..

# ④ 모델 가지치기 — 내 코퍼스 기준으로 e5-small을 37MB로 (최초 1회)
python pipeline/prune_model.py

# ⑤ WASM 차단 환경 폴백 자산 (선택, 권장)
node pipeline/build_terms.js
python pipeline/build_fallback_vectors.py
```

### 2-3. 산출물 배치

정적 호스팅의 아무 경로에나 올린다 (아래는 기본 경로 기준):

| 파일 | 내용 | 크기(785편 기준) |
|---|---|---|
| `/search-index/doc_meta.json` | 글 메타 (벡터 행 순서와 정렬) | ~360KB |
| `/search-index/doc_vectors_i8.bin` | 문서 벡터 (int8) | ~300KB |
| `/search-index/vector_scales.json` | 역양자화 스케일 | ~9KB |
| `/search-index/tags.json` | 태그 → 글 목록 | ~10KB |
| `/search-index/terms.json` | (폴백) 미리 계산된 형태소 사전 | ~64KB |
| `/search-index/vocab_meta.json` + `vocab_vectors_i8.bin` | (폴백) 어휘 벡터 표 | ~1.9MB |
| `/models/e5-small-ko/…` | 가지치기 모델 (ONNX + 토크나이저) | ~40MB |
| `related.json` / `related_links.json` | (선택) 글별 관련 글 top-k | ~0.6/1.4MB |

폴백 자산과 관련 글 파일은 없어도 검색은 동작한다 — 해당 기능만 조용히 빠진다.

### 2-4. 갱신

문서가 바뀌면 전체를 다시 돌린다(증분 없음 — 전부 재계산해도 몇 분).
가지치기 모델(④)은 코퍼스 어휘가 크게 바뀔 때만 다시 만들면 된다.
CI 자동화 예시는 `examples/adapters/`(감지 스크립트 + GitHub Actions 워크플로) 참고.

---

## 3. 브라우저에서 검색 (런타임)

### 3-1. 최소 예제

```ts
import { SearchController } from 'zipge';

const search = new SearchController({
  indexBase: '/search-index',                              // 기본값
  worker: {
    localModelPath: '/models',                             // 모델 서빙 루트
    localModel: 'e5-small-ko',                             // 가지치기 모델 디렉토리명
    hubModel: 'Xenova/multilingual-e5-small'               // 로컬 실패 시 HF 허브 폴백
  }
});

search.onResults = ({ query, results, tier, correctedTo, tags }) => {
  renderList(results);                    // SearchHit[]
  if (correctedTo) showNote(`'${correctedTo}'(으)로 검색된 결과입니다`);
  if (tags?.length) showTagChips(tags);   // [{name, count}]
};

search.onStatus = (status) => {
  // 'idle' | 'loading' | 'ready' | 'error'
  spinner.hidden = status !== 'loading';
};

// 검색창 포커스에 미리 데우면 첫 검색이 빨라진다
input.addEventListener('focus', () => search.warmup());
form.addEventListener('submit', (e) => {
  e.preventDefault();
  search.search(input.value);
});
```

`search()` 한 번에 결과가 **여러 번** 도착한다는 것이 핵심이다:

1. 즉시 — 키워드 결과 (`tier: 'keyword'`)
2. 형태소 분석기가 준비되면 — 개선된 키워드 결과 (같은 tier)
3. 임베딩 모델이 준비되면 — 융합된 최종 결과 (`tier: 'semantic'` 또는,
   키워드 증거가 더 강하면 `'keyword'`)

마지막으로 도착한 것을 그리면 된다. 화면이 뒤로 가는 일은 없다(내부에서 방지).

### 3-2. SearchHit

```ts
type SearchHit = {
  id: string; slug: string; title: string;
  board: string | null;        // 게시판 slug
  boardName: string | null;
  category: string | null;
  tags: string[];
  authors: { name: string; role: string }[];
  created: string;             // "YYYY-MM-DD"
  thumbnail: string | null;
  preview: string | null;
  views: number;
  score: number;               // keyword: 0~1.5 (1.5 = 제목 리터럴), semantic: 코사인
  source: 'keyword' | 'semantic';
};
```

정렬·필터는 소비자 몫이다 — `created`/`views`로 재정렬하거나
`board`/`tags`로 걸러도 재검색은 필요 없다(이미 받은 배열만 조작).

### 3-3. SearchController 옵션·API

```ts
new SearchController(options?: {
  indexBase?: string;            // 기본 '/search-index'
  worker?: {
    localModelPath?: string;     // 기본 '/models'
    localModel?: string;         // 기본 'e5-small-ko'
    hubModel?: string;           // 기본 'Xenova/multilingual-e5-small'
  };
  recycleAfterEmbeds?: number;   // 기본 10 — N회 검색마다 워커 재활용(WASM 힙 리셋)
  idleUnloadMs?: number;         // 기본 180000 — 유휴 시 모델 해제
})
```

| 멤버 | 설명 |
|---|---|
| `search(query)` | 검색 실행. 결과는 `onResults`로 여러 번 도착 |
| `warmup()` | 메타·분석기·모델을 미리 로딩 (검색창 포커스 시점 권장) |
| `destroy()` | 워커 종료·타이머 해제 (페이지/컴포넌트 이탈 시 호출) |
| `allowCorrection` | `false`면 오타 교정 비활성 |
| `exactMode` | `true`면 추론 전부 끔 — 제목 리터럴 포함만, 모델 미로딩 |
| `status` | 현재 모델 상태 |
| `index` | 내부 `SearchIndex` 접근 (아래) |

**"원문 그대로 검색" UX 패턴**: 교정이 발동하면(`correctedTo`) 안내 문구와 함께
원문 검색 링크를 제공하고, 그 링크에서는 `exactMode = true`로 검색하라.
교정 루프가 원천 차단된다.

### 3-4. SearchIndex 직접 사용 (저수준)

```ts
import { SearchIndex } from 'zipge';

const index = new SearchIndex('/search-index');
await index.loadMeta();
await index.prepareKeyword();          // 형태소 사전 준비 (garu 또는 terms.json)

index.keywordSearch('세금');            // T0만
await index.loadVectors();
index.semanticSearch(queryVector);      // 직접 만든 벡터로 T1만
index.matchTags(['세금']);              // 연관 태그
await index.composeEmbedding('세금');   // 모델 없이 합성 임베딩 (폴백 자산 필요)
SearchIndex.fuse(kwHits, semHits, tokenCount, corrected);  // 융합 정책
```

한국어 유틸도 노출된다: `extractTerms`, `stripParticle`, `termsMatch`,
`toJamo`, `withinEditDistance`, `hasHangul`.

### 3-5. 워커 번들링

라이브러리 내부가 `new Worker(new URL('./embed-worker.ts', import.meta.url), { type: 'module' })`
패턴을 쓰므로 **Vite·webpack 5 등 표준 번들러에서 소스째 소비하면 그대로 동작**한다
(`zipge/src/*`가 exports에 열려 있음). 번들러 없이 쓸 경우 `dist/embed-worker.js`를
직접 서빙하고 같은 오리진에서 로딩되게 하라.

---

## 4. 자동으로 되는 것들 (설정 불필요)

- **강등 사다리**: WebGPU → WASM(로컬 모델) → WASM(허브) → 합성 임베딩.
  각 단계는 타임아웃 + 실제 추론 웜업 검증을 통과해야 채택된다
- **락다운 모드(WASM 차단)**: `terms.json`·어휘 벡터가 배치돼 있으면
  형태소 매칭·오타 교정·간이 시맨틱까지 순수 JS로 동작
- **메모리**: N회 검색마다 워커 재활용, 유휴 시 해제 — `destroy()`만 잊지 말 것
- **연관도 게이트**: 기준 미달 결과는 반환되지 않는다 (무관한 결과로 채우지 않음)

---

## 5. 트러블슈팅

| 증상 | 원인/해결 |
|---|---|
| `status`가 `error` | 사다리 전 단계 실패 — 에러 문자열에 단계별 사유 포함. 모델 경로(`localModelPath`/`localModel`)와 파일 배치를 확인 |
| `index dim ... != 384` | 문서 벡터를 e5-small(384차원)이 아닌 모델로 만들었음 — 파이프라인 `embed_articles.py small` 사용 |
| 시맨틱 결과가 안 뜸 | ① 모델 로딩 중(상태 확인) ② 연관도 게이트에 걸림(정상 — 정말 관련 글이 없는 것) |
| 교정이 이상함 | 리터럴 일치가 있으면 교정은 발동하지 않음(의도). `allowCorrection = false`로 끌 수도 있음 |
| 메모리 점유 큼 | WASM 실행 중엔 정상(수백 MB). 유휴/이탈 시 해제됨. WebGPU 지원 브라우저는 더 낮음 |
| CORS 에러 | 인덱스·모델은 같은 오리진 또는 CORS 허용된 정적 호스팅에서 서빙해야 함 |
