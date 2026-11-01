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
npm install zipge
npm install garu-ko        # 선택: 형태소 매칭·오타 교정 품질 향상 (권장)
```

- 시맨틱 검색은 **학생 인코더**(모델 없는 쿼리 인코더, 순수 JS)가 맡는다 — 추론 런타임 설치가 필요 없다
- `garu-ko` — 한국어 형태소 분석기(WASM). 없거나 로딩에 실패하면 zipge는
  자동으로 규칙 기반 폴백(조사 제거 + 미리 계산된 사전)으로 동작한다
- 번들러(Vite 등)는 선택적 import도 빌드 때 해석하려 한다. 번들러로 빌드한다면 `garu-ko`도
  설치하라 — 설치하지 않으면 "garu-ko를 해석할 수 없다"는 빌드 에러가 난다

파이프라인 쪽:

```bash
pip install -r pipeline/requirements.txt   # sentence-transformers, gensim, scipy 등
npm install                                # 레포 루트에서 — 파이프라인 Node 스크립트가 garu-ko를 쓴다
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
# ① 문서 임베딩 — 색인(doc_meta) 생성에 필요
python pipeline/embed_articles.py small

# ② 관련 글 추천을 쓸 경우에만: 그래프용 임베딩 + 그래프 + (③에 포함됨)
python pipeline/embed_articles.py base
python pipeline/build_graph.py

# ③ 검색 인덱스 (+ 그래프가 있으면 관련 글 top-k까지)
cd pipeline && python build_index.py && cd ..

# ④ 학생 인코더 증류 — 시맨틱 검색 (bge-m3 교사, GPU 권장)
bash pipeline/student/run.sh                  # 확장 어휘까지: BUILD_LEXICON=1 bash pipeline/student/run.sh

# ⑤ WASM 차단 환경 키워드 폴백 사전 (선택, 권장)
node pipeline/build_terms.js
```

④의 옵션·평가·게이트 보정은 [STUDENT.md](STUDENT.md). GPU가 없어도 돌지만 교사 임베딩이 느리다.

### 2-3. 산출물 배치

정적 호스팅의 아무 경로에나 올린다 (아래는 기본 경로 기준):

`data/index/`의 산출물을 정적 호스팅의 아무 경로에나 올린다 (아래는 기본 경로 기준):

| 파일 | 내용 | 크기(800편 기준) |
|---|---|---|
| `/search-index/doc_meta.json` | 글 메타 | ~420KB |
| `/search-index/student_meta.json` | 학생 인코더 메타 (어휘·문서 id·설정) | ~130KB |
| `/search-index/student_vocab_i8.bin` | 형태소 벡터표 (int8) | ~4.2MB |
| `/search-index/student_proj_i8.bin` + `student_params_f32.bin` | 투영·가중치 (2멤버) | ~0.9MB |
| `/search-index/student_doc_vectors_i8.bin` + `student_doc_scales.json` | 문서 벡터 (글당 1KB) | ~0.8MB |
| `/search-index/student_lex_meta.json` + `student_lex_*.bin` | (선택) 확장 어휘 — 단어당 ~230KB만 Range로 받음 | 묶음 22개, 합계 ~83MB |
| `/search-index/tags.json` | 태그 → 글 목록 | ~10KB |
| `/search-index/terms.json` | (폴백) 미리 계산된 형태소 사전 | ~64KB |
| `related.json` / `related_links.json` | (선택) 글별 관련 글 top-k | ~0.6/1.4MB |

확장 어휘·폴백 사전·관련 글 파일은 없어도 검색은 동작한다 — 해당 기능만 조용히 빠진다.
확장 어휘 묶음은 **HTTP Range 요청**을 지원하는 호스팅에 두는 게 좋다(대부분의 정적 호스팅·CDN이 지원;
미지원이면 묶음 전체 ≤4MB를 받아 잘라 쓴다). `build_index.py`가 함께 만드는
`doc_vectors_i8.bin`·`vector_scales.json`은 학생 평가의 비교 기준용이라 올리지 않는다.

### 2-4. 갱신

문서가 바뀌면 전체를 다시 돌린다(증분 없음 — 전부 재계산해도 몇 분).
학생 인코더(④)도 **같은 회차에** 다시 증류해야 새 글을 잘 찾는다. 확장 어휘(`BUILD_LEXICON=1`)는
처음 한 번 만들면 이후 재학습이 그 입력 기저를 재사용하므로, 출처나 도메인이 바뀔 때만 다시 만든다.
CI 자동화 예시는 `examples/adapters/`(감지 스크립트 + GitHub Actions 워크플로) 참고.

---

## 3. 브라우저에서 검색 (런타임)

### 3-1. 최소 예제

```ts
import { SearchController } from 'zipge';

const search = new SearchController({
  indexBase: '/search-index'                               // 기본값
});

search.onResults = ({ query, results, tier, correctedTo, tags }) => {
  renderList(results);                    // SearchHit[]
  if (correctedTo) showNote(`'${correctedTo}'(으)로 검색된 결과입니다`);
  if (tags?.length) showTagChips(tags);   // [{name, count}]
};

search.onStatus = (status) => {
  // 'idle' | 'loading' | 'ready' | 'error' — 쿼리 인코더(~6MB) 로딩 상태
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
2. 쿼리 인코더(~6MB, 이후 캐시)와 형태소 분석기가 준비되면 — 융합된 최종 결과
   (`tier: 'semantic'` 또는, 키워드 증거가 더 강하거나 연관도 게이트에 걸리면 `'keyword'`)

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
  score: number;               // keyword: 0~1.5 (1.5 = 제목 리터럴), semantic: 교사 공간 코사인
  source: 'keyword' | 'semantic';
};
```

정렬·필터는 소비자 몫이다 — `created`/`views`로 재정렬하거나
`board`/`tags`로 걸러도 재검색은 필요 없다(이미 받은 배열만 조작).

### 3-3. SearchController 옵션·API

```ts
new SearchController(options?: {
  indexBase?: string;            // 기본 '/search-index'
  studentGate?: {                // 시맨틱 결과를 섞는 최고 코사인 문턱 — 코퍼스별 보정 권장 (STUDENT.md §4)
    exact?: number;              // 기본 0.3   — 어휘·확장 어휘에 정확히 걸린 토큰이 있을 때
    fallbackOnly?: number;       // 기본 0.635 — 부분문자열 폴백으로만 걸렸을 때
  };
})
```

| 멤버 | 설명 |
|---|---|
| `search(query)` | 검색 실행. 결과는 `onResults`로 여러 번 도착 |
| `warmup()` | 메타·분석기·쿼리 인코더를 미리 로딩 (검색창 포커스 시점 권장) |
| `destroy()` | 진행 중인 검색의 결과 전달을 끊는다 (페이지/컴포넌트 이탈 시 호출) |
| `allowCorrection` | `false`면 오타 교정 비활성 |
| `exactMode` | `true`면 추론 전부 끔 — 제목 리터럴 포함만 |
| `status` | 쿼리 인코더 상태 — 학생 자산이 없으면 `'error'`이고 키워드 계층만 동작 |
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
await index.loadStudent();              // 학생 인코더 자산 (없으면 false)
const enc = await index.studentQuery('세금 인상의 효과');  // {vector, core, lexicon, fallback} | null
index.studentSearch(enc.vector);        // T1만
index.matchTags(['세금']);              // 연관 태그
SearchIndex.fuse(kwHits, semHits, tokenCount, corrected);  // 융합 정책
```

한국어 유틸도 노출된다: `extractTerms`, `studentTokens`, `stripParticle`, `termsMatch`,
`toJamo`, `withinEditDistance`, `hasHangul`.

---

## 4. 자동으로 되는 것들 (설정 불필요)

- **락다운 모드(WASM 차단)**: 학생 인코더는 순수 JS라 그대로 동작한다. garu WASM이 막히면
  규칙 기반 토큰화로, 키워드 계층은 `terms.json`으로 대체 — 오타 교정·의미 검색까지 유지
- **확장 어휘 지연 로딩**: 코퍼스에 없는 단어가 쿼리에 있을 때만 그 단어의 조각(~230KB)을 Range로 받는다
- **연관도 게이트**: 기준 미달 결과는 반환되지 않는다 (무관한 결과로 채우지 않음)

---

## 5. 트러블슈팅

| 증상 | 원인/해결 |
|---|---|
| 빌드 에러: `garu-ko`를 해석할 수 없음 | 번들러가 선택적 import까지 빌드 때 해석한다. `garu-ko`를 설치하라 |
| `status`가 `error` (`student assets not found`) | 색인에 `student_meta.json` 등이 없다 — `bash pipeline/student/run.sh` 산출물(`student_*`)을 함께 올렸는지 확인. 그동안은 키워드 계층만 동작 |
| 시맨틱 결과가 안 뜸 | ① 인코더 로딩 중(상태 확인) ② 연관도 게이트에 걸림(정상 — 정말 관련 글이 없는 것). 너무 자주 걸리면 `calib_gate.mjs`로 `studentGate` 보정 |
| 새 글이 시맨틱으로 안 잡힘 | 학생 인코더를 색인과 같은 회차에 다시 증류하지 않았음 |
| 확장 어휘 단어가 안 잡힘 | `student_lex_meta.json`의 `basis_sha1`이 `student_meta.json`과 다르다(확장 어휘를 다시 만든 뒤 재학습 안 함) |
| 교정이 이상함 | 리터럴 일치가 있으면 교정은 발동하지 않음(의도). `allowCorrection = false`로 끌 수도 있음 |
| CORS 에러 | 인덱스는 같은 오리진 또는 CORS 허용된 정적 호스팅에서 서빙해야 함 (확장 어휘는 `Range: bytes=시작-끝` 단일 구간 요청 — CORS 안전 헤더라 보통 사전 요청이 없다) |
