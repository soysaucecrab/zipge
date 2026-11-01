# 집게 (Zipge)

[![npm version](https://img.shields.io/npm/v/zipge.svg)](https://www.npmjs.com/package/zipge)
[![npm downloads](https://img.shields.io/npm/dm/zipge.svg)](https://www.npmjs.com/package/zipge)
[![GitHub stars](https://img.shields.io/github/stars/soysaucecrab/zipge.svg?style=social)](https://github.com/soysaucecrab/zipge)

**정적 사이트에 AI 검색을 붙여주는 엔진.**

방문자가 키워드가 아니라 **의미로** 자료를 찾게 됩니다 — 조사·어미
변형 때문에 키워드 매칭이 잘 깨지는 한국어에서도. 시맨틱 검색·형태소
매칭·오타 교정이 전부 사용자 브라우저 안에서 동작하고(모델 없이 약 6MB),
서버는 정적 파일만 전송하므로 서버·API 비용이 0입니다. RAM 756MB
서버에서 실서비스로 검증된 구조를 라이브러리로 추린 것입니다.

- **계층 융합 검색** — 리터럴 → 형태소 → 시맨틱 순으로 서열화. 문자
  그대로 일치가 항상 추론을 이깁니다
- **한국어 오타 교정** — 자모 편집거리 + 코퍼스 어휘 사전
  (`쎄금`→`세금`, `사유재솬`→`사유재산`). 교정문이 시맨틱 검색에
  재투입되어 오타 상태에서도 의미 검색이 됩니다
- **모델 없는 쿼리 인코더 (학생)** — bge-m3의 쿼리 임베딩을 증류한
  형태소 표(int8) + 투영 1회. 약 6MB로, 개념 검색에서 37MB 모델보다
  정확합니다 (MRR 0.621 vs 0.530). 순수 JS라 WASM이 차단된 환경
  (iOS/macOS 락다운 모드)에서도 시맨틱 검색이 그대로 동작합니다
- **확장 어휘** — 코퍼스에 없는 단어(신조어·속어)는 20만 단어급 확장
  어휘에서 그 단어가 든 조각(~230KB)만 Range 요청으로 받아 처리
- **37MB 시맨틱 모델 (선택)** — multilingual-e5-small(118MB)을 자기 코퍼스
  어휘로 가지치기. 임베딩 공간 손실 0 (fp32 코사인 1.0000).
  학생 자산이 없으면 이 경로로 동작합니다
- **강등 사다리 (모델 경로)** — WebGPU → WASM → HF 허브 → 순수 JS 합성
  임베딩. 각 단계는 타임아웃 + 실추론 웜업 검증을 통과해야 채택됩니다
- **WASM 힙 수명 관리 (모델 경로)** — N회 검색마다 워커 재활용, 유휴 시
  해제. 장시간 세션에서도 메모리가 누적되지 않습니다
- **관련 글 추천** — 문서 그래프(node2vec) 기반 top-k를 오프라인에서
  미리 계산해 정적 파일로 서빙
- 검색을 쓰지 않는 방문자는 **아무것도 다운로드하지 않습니다** (지연 로딩)

## Why Zipge?

|  | 서버 검색엔진 (Elasticsearch 등) | 클라이언트 키워드 검색 (Pagefind, Lunr) | **Zipge** |
|---|---|---|---|
| 서버 요구 | 전용 인스턴스 (RAM GB급) | 정적 호스팅 | **정적 호스팅** |
| 시맨틱(의미) 검색 | 가능 | 불가 | **가능 (브라우저 내, 모델 없이 6MB)** |
| 한국어 형태소 매칭 | 플러그인 필요 | 미지원·부분적 | **내장 ([garu](https://github.com/ongjin/garu))** |
| 오타 교정 | 가능 | 불가·제한적 | **자모 단위 교정 + 의미 검색 연동** |
| 운영 비용 | 인스턴스 상시 비용 | 0 | **0** |
| 적정 규모 | 무제한 | ~수만 편 | 수백~5천 편 (한계 ~3만, 아래 참고) |

이름은 게의 **집게**에서 왔습니다 — 문서 더미에서 원하는 것을 집어
올립니다. [garu](https://github.com/ongjin/garu)가 텍스트를 가루로
갈면, zipge가 집어냅니다. **zip**은 덤입니다: 증류(모델 → 6MB 표)·
가지치기(118MB→37MB)·int8 양자화로 눌러 담았습니다.

## Quick Start

```bash
npm install zipge
npm install garu-ko                    # 선택 — 형태소 매칭·오타 교정 품질 향상 (권장)
npm install @huggingface/transformers  # 선택 — 37MB 모델 경로를 쓸 때만
```

```ts
import { SearchController } from 'zipge';

const search = new SearchController({
  indexBase: '/search-index'   // student_meta.json이 있으면 학생 인코더, 없으면 모델 경로
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
나오고, 형태소 분석기(~1초)·쿼리 인코더(학생 ~6MB, 모델이면 최초 ~4초,
이후 캐시)가 준비되는 대로 같은 화면이 상위 계층으로 승격됩니다.

인덱스는 오프라인 파이프라인(Python)으로 만듭니다:

```bash
pip install -r pipeline/requirements.txt
python pipeline/embed_articles.py small   # 문서 임베딩
cd pipeline && python build_index.py      # 검색 인덱스
bash pipeline/student/run.sh              # 학생 인코더 증류 (bge-m3 교사, GPU 권장)
python pipeline/prune_model.py            # (선택) 37MB 모델 경로
```

문서 입력은 **`{id, title, content}` 3개 필드면 충분**하다 — slug·태그·
저자·게시판 등은 전부 선택이며 있으면 그만큼 기능이 붙는다.
전체 스키마·절차는 **[docs/USAGE.md](docs/USAGE.md)** 참고.

## 아키텍처

```
오프라인 (Python, 1회/글 갱신 시)                브라우저 (TS 런타임)
─────────────────────────────                ──────────────────────────────
문서 JSON ─┬─ e5 임베딩 ──→ 문서벡터(int8)      쿼리 ─→ ① 리터럴 제목 일치
           ├─ bge-m3 증류 ─→ 학생 인코더 ~6MB          ② 형태소 매칭 (garu WASM)
           ├─ 문서 그래프 ─→ 관련 글 top-k              ③ 오타 교정 → 재임베딩
           ├─ 어휘 가지치기 → 모델 37MB (선택)          ④ 시맨틱 (학생 / 모델)
           └─ 폴백 사전·확장 어휘                       ⑤ 융합 + 연관도 게이트
                    │
                    └────── 정적 호스팅 ──────→  fetch (검색 시에만, 이후 캐시)
```

검색 품질 실측 (785편 코퍼스, 원본 글 검색 Recall@10, 모델 경로):

| 정상 쿼리 | 오타 1글자 | 띄어쓰기 오류 | 붙여쓰기 |
|---|---|---|---|
| 0.880 | 0.870 | 0.803 | 0.870 |

개념 검색 실측 (800편 코퍼스, 키워드로 안 걸리는 수작업 쿼리 122개, MRR):

| 학생 인코더 (~6MB) | 37MB 모델 | 참고: bge-m3 (서버용 상한) |
|---|---|---|
| 0.621 | 0.530 | 0.816 |

## 규모 한계

브라우저가 받는 것은 원문이 아니라 쿼리 인코더(학생 ~6MB 또는 모델 37MB)와
글당 벡터(모델 384B / 학생 1KB)+메타(~0.5KB)입니다. 아래 표는 모델 경로
인덱스 기준이고, 학생 경로는 글당 벡터가 1KB라 인덱스가 그만큼 커집니다.

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
  encoder?: 'auto' | 'student' | 'model';   // 기본 'auto' — 학생 자산이 있으면 학생
  studentGate?: { exact?, fallbackOnly? };  // 학생 결과 노출 문턱 (코퍼스별 보정 권장)
  worker?: { localModelPath?, localModel?, hubModel? };   // 모델 경로
  recycleAfterEmbeds?: number;        // 기본 10
  idleUnloadMs?: number;              // 기본 180000
})

controller.search(query)              // 결과는 onResults로 여러 번 도착
controller.warmup()                   // 미리 로딩 (검색창 포커스 권장)
controller.destroy()                  // 워커·타이머 해제
controller.exactMode = true           // 추론 전부 끔 (리터럴만)
controller.allowCorrection = false    // 오타 교정만 끔

// 저수준
import { SearchIndex, extractTerms, studentTokens, termsMatch, toJamo } from 'zipge';
```

## FAQ

**Q. 서버가 정말 아무 일도 안 하나요?**
정적 파일 전송뿐입니다. 임베딩 추론·매칭·교정 전부 브라우저에서
일어납니다. 실서비스는 RAM 756MB(가용 ~100MB) 서버에서 돌고 있습니다.

**Q. 사용자가 받는 용량은?**
학생 인코더 경로는 검색을 처음 실행할 때 ~6MB + 인덱스 ~1.5MB(800편
기준)이고, 코퍼스에 없는 단어가 나오면 단어당 ~230KB를 더 받습니다.
모델 경로는 모델 37MB + 인덱스 ~1MB(785편 기준). 브라우저 캐시로 이후
방문은 0. 검색을 안 쓰면 아무것도 받지 않습니다.

**Q. 정말 모델이 없나요?**
검색 시점에는 추론 모델이 없습니다. 오프라인에서 bge-m3로 증류한
형태소 표(int8)와 투영 행렬 하나가 쿼리를 문서 벡터 공간으로 옮깁니다.
코퍼스에 맞춰 학습되므로 글을 추가하면 다시 증류해야 합니다.

**Q. garu 없이도 동작하나요?**
동작합니다. garu(또는 WASM 자체)가 없으면 미리 계산된 사전과 규칙 기반
조사 제거로 자동 폴백합니다 — 형태소 매칭·오타 교정이 유지되고, 학생
인코더도 같은 규칙으로 토큰을 만듭니다.

**Q. iOS 락다운 모드처럼 WASM이 막힌 환경은?**
학생 인코더는 순수 JS라 그대로 동작합니다(garu WASM이 막히면 규칙 기반
토큰화로 대체). 모델 경로는 강등 사다리의 마지막 단인 순수 JS 합성
임베딩(미리 계산된 어휘 벡터 평균)으로 내려가 긴 자연어 질문의 정밀도가
낮아집니다.

**Q. 다른 언어도 되나요?**
매칭·교정 계층과 학생 인코더의 토큰화는 한국어 특화입니다(자모 분해·조사
제거·형태소). 모델 경로는 multilingual-e5라 영어 쿼리는 그대로 동작합니다. 다른 언어 코퍼스는
가지치기 어휘 선정을 바꾸면 이론상 가능하지만 검증되지 않았습니다.

**Q. 글을 추가하면?**
파이프라인을 다시 돌립니다(증분 없음 — 785편 기준 GPU 5분/CPU 20분).
학생 인코더도 같은 회차에 다시 증류해야 새 글을 잘 찾습니다.
`examples/adapters/`에 새 글 감지 → 자동 재학습 → PR 생성까지의
GitHub Actions 예시가 있습니다.

## 문서

- [사용 방법 (USAGE.md)](docs/USAGE.md) — 설치·파이프라인·API 전체
- [학생 인코더 (STUDENT.md)](docs/STUDENT.md) — 모델 없는 쿼리 인코더의 방법·자산 포맷·파이프라인·게이트 보정
- [아키텍처 (ARCHITECTURE.md)](docs/ARCHITECTURE.md) — 설계 근거·벤치마크·실서비스 기록

## License

Apache-2.0 — 재배포·파생물은 [`NOTICE`](NOTICE)의 제작자 표기
(KIM ZINU / soysaucecrab)를 유지해야 합니다.

의존·감사: [garu](https://github.com/ongjin/garu) (MIT) ·
[transformers.js](https://github.com/huggingface/transformers.js) (Apache-2.0) ·
[multilingual-e5](https://huggingface.co/intfloat/multilingual-e5-small) (MIT) ·
[BGE-M3](https://huggingface.co/BAAI/bge-m3) (MIT, 오프라인 증류 교사)
