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
  형태소 표(int8) + 투영 1회. 약 6MB로, 개념 검색에서 기존 e5-small
  모델보다 정확합니다 (MRR 0.621 vs 0.530). 순수 JS라 WASM이 차단된 환경
  (iOS/macOS 락다운 모드)에서도 시맨틱 검색이 그대로 동작합니다
- **확장 어휘** — 코퍼스에 없는 단어(신조어·속어)는 20만 단어급 확장
  어휘에서 그 단어가 든 조각(~230KB)만 Range 요청으로 받아 처리. 단어를
  해시해 조각을 정하므로 단어 목록 자체도 받지 않습니다
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
갈면, zipge가 집어냅니다. **zip**은 덤입니다: 증류(모델 → 6MB 표)와
int8 양자화로 눌러 담았습니다.

## Quick Start

```bash
npm install zipge
npm install garu-ko   # 선택 — 형태소 매칭·오타 교정 품질 향상 (권장, 번들러를 쓰면 필수)
```

```ts
import { SearchController } from 'zipge';

const search = new SearchController({
  indexBase: '/search-index'
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
나오고, 형태소 분석기(~1초)·쿼리 인코더(~6MB, 이후 캐시)가
준비되는 대로 같은 화면이 상위 계층으로 승격됩니다.

인덱스는 오프라인 파이프라인(Python)으로 만듭니다:

```bash
pip install -r pipeline/requirements.txt
npm install                               # 학생 파이프라인의 형태소 분석(garu-ko)
python pipeline/embed_articles.py small   # 문서 임베딩
(cd pipeline && python build_index.py)    # 검색 인덱스
bash pipeline/student/run.sh              # 학생 인코더 증류 (bge-m3 교사, GPU 권장)
```

문서 입력은 **`{id, title, content}` 3개 필드면 충분**하다 — slug·태그·
저자·게시판 등은 전부 선택이며 있으면 그만큼 기능이 붙는다.
전체 스키마·절차는 **[docs/USAGE.md](docs/USAGE.md)** 참고.

## 아키텍처

```
오프라인 (Python, 1회/글 갱신 시)                브라우저 (TS 런타임)
─────────────────────────────                ──────────────────────────────
문서 JSON ─┬─ bge-m3 임베딩 ─→ 문서벡터(int8)   쿼리 ─→ ① 리터럴 제목 일치
           ├─ bge-m3 증류 ─→ 학생 인코더 ~6MB           ② 형태소 매칭 (garu WASM)
           ├─ 문서 그래프 ─→ 관련 글 top-k              ③ 오타 교정 → 재인코딩
           └─ 폴백 사전·확장 어휘                       ④ 시맨틱 (학생 인코더)
                    │                                   ⑤ 융합 + 연관도 게이트
                    └────── 정적 호스팅 ──────→  fetch (검색 시에만, 이후 캐시)
```

쿼리 인코딩(학생)은 모델 없이 표 조회와 행렬 곱 한 번입니다:

```
토큰(garu 명사·어절) → 단어 벡터 표(int8, 384) 조회 → 학습된 가중합
                     → 투영 W(384→1024) 1회 → 문서 벡터와 코사인
```

개념 검색 실측 (800편 코퍼스, 키워드로 안 걸리는 수작업 쿼리 122개, MRR):

| 학생 인코더 (~6MB) | 참고: e5-small 모델 | 참고: bge-m3 (서버용 상한) |
|---|---|---|
| 0.621 | 0.530 | 0.816 |

## 규모 한계

브라우저가 받는 것은 원문이 아니라 쿼리 인코더(~5MB)와 글당 벡터(1KB)+
메타(~0.5KB)입니다. 아래 다운로드는 인코더를 뺀 인덱스 기준입니다.

| 글 수 | 첫 검색 다운로드 | 검색 지연 | 판정 |
|---|---|---|---|
| ~5,000 | ~8MB | <50ms | 권장 범위 |
| ~10,000 | ~15MB | ~100ms | 실용 가능 |
| ~30,000 | ~45MB | 수백 ms | 실용 한계 |
| 100,000+ | — | — | 서버 검색의 영역 |

원문 총량은 글 수 한계 안이라면 GB여도 무방합니다(파이프라인 시간만
증가). 단 초장문은 청크 평균으로 벡터가 희석되므로 섹션 단위 분할을
권장합니다.

## API

핵심만 요약합니다. 전체는 [docs/USAGE.md](docs/USAGE.md).

```ts
new SearchController(options?: {
  indexBase?: string;                 // 기본 '/search-index'
  studentGate?: { exact?, fallbackOnly? };  // 시맨틱 결과 노출 문턱 (코퍼스별 보정 권장)
})

controller.search(query)              // 결과는 onResults로 여러 번 도착
controller.warmup()                   // 미리 로딩 (검색창 포커스 권장)
controller.destroy()                  // 진행 중인 결과 전달 중단
controller.exactMode = true           // 추론 전부 끔 (리터럴만)
controller.allowCorrection = false    // 오타 교정만 끔

// 저수준
import { SearchIndex, extractTerms, studentTokens, termsMatch, toJamo } from 'zipge';
```

## FAQ

**Q. 서버가 정말 아무 일도 안 하나요?**
정적 파일 전송뿐입니다. 쿼리 인코딩·매칭·교정 전부 브라우저에서
일어납니다. 실서비스는 RAM 756MB(가용 ~100MB) 서버에서 돌고 있습니다.

**Q. 사용자가 받는 용량은?**
검색을 처음 실행할 때 학생 자산 ~6MB(문서 벡터 포함) + 문서 메타 ~0.4MB
(800편 기준)이고, 코퍼스에 없는 단어가 나오면 단어당 ~230KB를 더 받습니다. 브라우저
캐시로 이후 방문은 0. 검색을 안 쓰면 아무것도 받지 않습니다.

**Q. 정말 모델이 없나요?**
검색 시점에는 추론 모델이 없습니다. 오프라인에서 bge-m3로 증류한
형태소 표(int8)와 투영 행렬 하나가 쿼리를 문서 벡터 공간으로 옮깁니다.
학습은 bge-m3가 매긴 문서 순위 분포를 따라 하도록(KL 증류) 단어 가중치와
투영만 맞추고, 단어·문서 벡터는 교사 것을 그대로 씁니다. 코퍼스에 맞춰
학습되므로 글을 추가하면 다시 증류해야 합니다.

**Q. 브라우저 대신 서버에서 돌려도 되나요?**
됩니다. 1코어로 고정해 실측하면(800편, 쿼리 292개) 쿼리당 p50 8ms·p95
19ms이고, 메모리는 garu 포함 ~86MB, 확장 어휘를 전부 올려도 ~210MB입니다.
384MB 메모리 상한에서도 정상 동작했습니다.

**Q. garu 없이도 동작하나요?**
동작합니다. garu(또는 WASM 자체)가 없으면 미리 계산된 사전과 규칙 기반
조사 제거로 자동 폴백합니다 — 형태소 매칭·오타 교정이 유지되고, 학생
인코더도 같은 규칙으로 토큰을 만듭니다.

**Q. iOS 락다운 모드처럼 WASM이 막힌 환경은?**
학생 인코더는 순수 JS라 그대로 동작합니다(garu WASM이 막히면 규칙 기반
토큰화로 대체). 리터럴·형태소·오타 교정·의미 검색이 전부 살아 있습니다.

**Q. 다른 언어도 되나요?**
매칭·교정 계층과 학생 인코더의 토큰화는 한국어 특화입니다(자모 분해·조사
제거·형태소). 증류 교사 bge-m3는 다국어라 토큰화를 바꾸면 다른 언어
코퍼스도 이론상 가능하지만 검증되지 않았습니다.

**Q. 글을 추가하면?**
파이프라인을 다시 돌립니다(증분 없음 — 800편 기준 RTX 3060에서 10~15분).
학생 인코더도 같은 회차에 다시 증류해야 새 글을 잘 찾습니다. 교사 bge-m3
때문에 GPU를 권장합니다(실측: 가중치 2.3GB, VRAM 피크 ~4.5GB, RAM ~3.6GB —
8GB VRAM GPU면 충분). GPU가 없어도 돌지만 교사 임베딩이 느립니다.
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
[multilingual-e5](https://huggingface.co/intfloat/multilingual-e5-small) (MIT) ·
[BGE-M3](https://huggingface.co/BAAI/bge-m3) (MIT, 오프라인 증류 교사)
