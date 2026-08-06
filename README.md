# 집게 (Zipge)

[![npm version](https://img.shields.io/npm/v/zipge.svg)](https://www.npmjs.com/package/zipge)
[![npm downloads](https://img.shields.io/npm/dm/zipge.svg)](https://www.npmjs.com/package/zipge)
[![GitHub stars](https://img.shields.io/github/stars/soysaucecrab/zipge.svg?style=social)](https://github.com/soysaucecrab/zipge)

**서버 연산 0으로 브라우저에서 완결되는 한국어 하이브리드 검색엔진 (시맨틱 모델 37MB).**

정적 호스팅만 있으면 시맨틱 검색·형태소 매칭·오타 교정이 전부 사용자
브라우저 안에서 동작합니다. RAM 756MB 서버에서 실서비스로 검증된
구조를 라이브러리로 추린 것입니다.

- **계층 융합 검색** — 리터럴 → 형태소 → 시맨틱 순으로 서열화. 문자
  그대로 일치가 항상 추론을 이깁니다
- **한국어 오타 교정** — 자모 편집거리 + 코퍼스 어휘 사전
  (`쎄금`→`세금`, `사유재솬`→`사유재산`). 교정문이 시맨틱 검색에
  재투입되어 오타 상태에서도 의미 검색이 됩니다
- **37MB 시맨틱 모델** — multilingual-e5-small(118MB)을 자기 코퍼스
  어휘로 가지치기. 임베딩 공간 손실 0 (fp32 코사인 1.0000)
- **강등 사다리** — WebGPU → WASM → HF 허브 → 순수 JS 합성 임베딩.
  각 단계는 타임아웃 + 실추론 웜업 검증을 통과해야 채택됩니다.
  WASM이 차단된 환경(iOS/macOS 락다운 모드)에서도 검색이 동작합니다
- **WASM 힙 수명 관리** — N회 검색마다 워커 재활용, 유휴 시 해제.
  장시간 세션에서도 메모리가 누적되지 않습니다
- **관련 글 추천** — 문서 그래프(node2vec) 기반 top-k를 오프라인에서
  미리 계산해 정적 파일로 서빙
- 검색을 쓰지 않는 방문자는 **아무것도 다운로드하지 않습니다** (지연 로딩)

## Why Zipge?

|  | 서버 검색엔진 (Elasticsearch 등) | 클라이언트 키워드 검색 (Pagefind, Lunr) | **Zipge** |
|---|---|---|---|
| 서버 요구 | 전용 인스턴스 (RAM GB급) | 정적 호스팅 | **정적 호스팅** |
| 시맨틱(의미) 검색 | 가능 | 불가 | **가능 (브라우저 내 추론)** |
| 한국어 형태소 매칭 | 플러그인 필요 | 미지원·부분적 | **내장 ([garu](https://github.com/ongjin/garu))** |
| 오타 교정 | 가능 | 불가·제한적 | **자모 단위 교정 + 의미 검색 연동** |
| 운영 비용 | 인스턴스 상시 비용 | 0 | **0** |
| 적정 규모 | 무제한 | ~수만 편 | 수백~5천 편 (한계 ~3만, 아래 참고) |

이름은 게의 **집게**에서 왔습니다 — 문서 더미에서 원하는 것을 집어
올립니다. [garu](https://github.com/ongjin/garu)가 텍스트를 가루로
갈면, zipge가 집어냅니다. **zip**은 덤입니다: 가지치기(118MB→37MB)와
int8 양자화로 눌러 담았습니다.

## Quick Start

```bash
npm install zipge @huggingface/transformers
npm install garu-ko   # 선택 — 형태소 매칭·오타 교정 품질 향상 (권장)
```

```ts
import { SearchController } from 'zipge';

const search = new SearchController({
  indexBase: '/search-index',
  worker: { localModelPath: '/models', localModel: 'e5-small-ko' }
});

search.onResults = ({ results, tier, correctedTo }) => {
  render(results);                        // SearchHit[]
  // tier: 'keyword' -> 'semantic' 순으로 결과가 점진 업그레이드된다
  if (correctedTo) note(`'${correctedTo}'(으)로 검색된 결과입니다`);
};

input.addEventListener('focus', () => search.warmup());  // 미리 데우기
form.addEventListener('submit', () => search.search(input.value));
```

`search()` 한 번에 결과가 여러 번 도착합니다 — 키워드 결과가 즉시
나오고, 형태소 분석기(~1초)·임베딩 모델(최초 ~4초, 이후 캐시)이
준비되는 대로 같은 화면이 상위 계층으로 승격됩니다.

인덱스는 오프라인 파이프라인(Python)으로 만듭니다:

```bash
pip install -r pipeline/requirements.txt
python pipeline/embed_articles.py small   # 문서 임베딩
cd pipeline && python build_index.py      # 검색 인덱스
python pipeline/prune_model.py            # 내 코퍼스 기준 37MB 모델 생성
```

문서 입력은 **`{id, title, content}` 3개 필드면 충분**하다 — slug·태그·
저자·게시판 등은 전부 선택이며 있으면 그만큼 기능이 붙는다.
전체 스키마·절차는 **[docs/USAGE.md](docs/USAGE.md)** 참고.

## 아키텍처

```
오프라인 (Python, 1회/글 갱신 시)                브라우저 (TS 런타임)
─────────────────────────────                ──────────────────────────────
문서 JSON ─┬─ e5 임베딩 ──→ 문서벡터(int8)      쿼리 ─→ ① 리터럴 제목 일치
           ├─ 문서 그래프 ─→ 관련 글 top-k              ② 형태소 매칭 (garu WASM)
           ├─ 어휘 가지치기 → 모델 37MB                 ③ 오타 교정 → 재임베딩
           └─ 폴백 사전·어휘벡터                        ④ 시맨틱 (WebGPU→WASM)
                    │                                   ⑤ 융합 + 연관도 게이트
                    └────── 정적 호스팅 ──────→  fetch (검색 시에만, 이후 캐시)
```

검색 품질 실측 (785편 코퍼스, 원본 글 검색 Recall@10):

| 정상 쿼리 | 오타 1글자 | 띄어쓰기 오류 | 붙여쓰기 |
|---|---|---|---|
| 0.880 | 0.870 | 0.803 | 0.870 |

## 규모 한계

브라우저가 받는 것은 원문이 아니라 글당 벡터(384B)+메타(~0.5KB)입니다.

| 글 수 | 첫 검색 다운로드 | 검색 지연 | 판정 |
|---|---|---|---|
| ~5,000 | ~4MB | <50ms | 권장 범위 |
| ~10,000 | ~9MB | ~100ms | 실용 가능 |
| ~30,000 | ~25MB | 수백 ms | 실용 한계 |
| 100,000+ | — | — | 서버 검색의 영역 |

원문 총량은 글 수 한계 안이라면 GB여도 무방합니다(파이프라인 시간만
증가). 단 초장문은 청크 평균으로 벡터가 희석되므로 섹션 단위 분할을
권장합니다.

## API

핵심만 요약합니다. 전체는 [docs/USAGE.md](docs/USAGE.md).

```ts
new SearchController(options?: {
  indexBase?: string;                 // 기본 '/search-index'
  worker?: { localModelPath?, localModel?, hubModel? };
  recycleAfterEmbeds?: number;        // 기본 10
  idleUnloadMs?: number;              // 기본 180000
})

controller.search(query)              // 결과는 onResults로 여러 번 도착
controller.warmup()                   // 미리 로딩 (검색창 포커스 권장)
controller.destroy()                  // 워커·타이머 해제
controller.exactMode = true           // 추론 전부 끔 (리터럴만)
controller.allowCorrection = false    // 오타 교정만 끔

// 저수준
import { SearchIndex, extractTerms, termsMatch, toJamo } from 'zipge';
```

## FAQ

**Q. 서버가 정말 아무 일도 안 하나요?**
정적 파일 전송뿐입니다. 임베딩 추론·매칭·교정 전부 브라우저에서
일어납니다. 실서비스는 RAM 756MB(가용 ~100MB) 서버에서 돌고 있습니다.

**Q. 사용자가 받는 용량은?**
검색을 처음 실행할 때 모델 37MB + 인덱스 ~1MB(785편 기준). 브라우저
캐시로 이후 방문은 0. 검색을 안 쓰면 아무것도 받지 않습니다.

**Q. garu 없이도 동작하나요?**
동작합니다. garu(또는 WASM 자체)가 없으면 미리 계산된 사전과 규칙 기반
조사 제거로 자동 폴백합니다 — 형태소 매칭·오타 교정이 유지됩니다.

**Q. iOS 락다운 모드처럼 WASM이 막힌 환경은?**
강등 사다리의 마지막 단인 순수 JS 합성 임베딩(미리 계산된 어휘 벡터
평균)으로 내려갑니다. 리터럴·형태소·오타 교정·단어 수준 의미 검색이
전부 살아 있고, 긴 자연어 질문의 정밀도만 낮아집니다.

**Q. 다른 언어도 되나요?**
매칭·교정 계층은 한국어 특화입니다(자모 분해·조사 제거). 시맨틱 계층은
multilingual-e5라 영어 쿼리는 그대로 동작합니다. 다른 언어 코퍼스는
가지치기 어휘 선정을 바꾸면 이론상 가능하지만 검증되지 않았습니다.

**Q. 글을 추가하면?**
파이프라인을 다시 돌립니다(증분 없음 — 785편 기준 GPU 5분/CPU 20분).
`examples/adapters/`에 새 글 감지 → 자동 재학습 → PR 생성까지의
GitHub Actions 예시가 있습니다.

## 문서

- [사용 방법 (USAGE.md)](docs/USAGE.md) — 설치·파이프라인·API 전체
- [아키텍처 (ARCHITECTURE.md)](docs/ARCHITECTURE.md) — 설계 근거·벤치마크·실서비스 기록

## License

Apache-2.0 — 재배포·파생물은 [`NOTICE`](NOTICE)의 제작자 표기
(KIM ZINU / soysaucecrab)를 유지해야 합니다.

의존·감사: [garu](https://github.com/ongjin/garu) (MIT) ·
[transformers.js](https://github.com/huggingface/transformers.js) (Apache-2.0) ·
[multilingual-e5](https://huggingface.co/intfloat/multilingual-e5-small) (MIT)
