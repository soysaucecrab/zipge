# 학생 인코더 — 모델 없는 쿼리 인코더

zipge의 시맨틱 검색은 브라우저에서 **임베딩 모델 없이** 쿼리를 문서 벡터 공간에 올린다.
형태소 벡터표(int8) + 형태소별 가중치 + 선형 투영 1회로 끝나며, 큰 모델(BAAI/bge-m3)의
쿼리 임베딩을 오프라인에서 **증류**해 학습한다. 쿼리 시점 연산은 행렬곱 몇 번이고 ONNX·WASM
런타임이 필요 없다(순수 JS).

---

## 1. 구조

```
쿼리 ─ 토큰화(garu 명사 · 어절 · 통째) ─→ 어휘 조회 → 확장 어휘 조회 → 최장 부분문자열 폴백
     ─→ p = Σ_slot  w_slot · v_slot           (v: 교사 단어 벡터 → PCA 384 → int8)
     ─→ q = W·p + b  (384 → 1024)  ─→ 정규화  ─→ 멤버 평균 ─→ 정규화
     ─→ 교사(bge-m3) 문서 벡터(int8, 1024)와 코사인
```

| 구성 | 크기 (800편 코퍼스 기준) | 학습 여부 |
|---|---|---|
| 어휘 int8 표 `student_vocab_i8.bin` | 약 1.1만 단어 × 384 = 4.2MB | 교사 임베딩 + PCA (학습 없음) |
| 형태소 가중치 θ, 폴백·확장 어휘 가중치 | 멤버당 V+2 float | 학습 (scale에 접어서 배포) |
| 투영 W, b | 멤버당 1024×384 int8 + 1024×2 float | 학습 (릿지 초기화 → 미세조정) |
| 문서 벡터 `student_doc_vectors_i8.bin` | 글당 1KB (800편 0.8MB) | 교사 임베딩 (학습 없음) |
| 확장 어휘 `student_lex_*` (선택) | 20.9만 단어, 4MB 이하 묶음 22개 (합계 83MB) | 교사 임베딩 (학습 없음) |

2멤버 기준 첫 검색 다운로드 ≈ 6.2MB(+ `doc_meta.json`). 확장 어휘는 코퍼스에 없는 단어가
쿼리에 있을 때만 그 단어가 든 조각 하나(약 230KB)를 HTTP Range로 받는다.

## 2. 학습

- **교사**: BAAI/bge-m3 (1024차원, MIT). 문서 벡터는 `pipeline/embed_articles.py`와 같은 청킹
  (550자 청크, `제목\n본문`, 청크 평균).
- **학습 쿼리(라벨 없음)**: 글당 본문 명사 2~5개 연속창 40개 + 본문 구절(8~40자) 20개 + 제목 + 태그.
  평가 쿼리와 토큰 자카드 ≥ 0.6인 것은 제거.
- **손실**: `KL(softmax(학생·D/τ) ‖ softmax(교사·D/τ)) + 0.5·(1 − cos(학생, 교사)) + 0.3·CE(학생·D/τ, 출처 글)`,
  τ = 0.05. 교사의 쿼리 임베딩이 정답이므로 손라벨이 필요 없다.
- **최적화**: (W, b)를 릿지 회귀 해로 초기화 → Adam, 배치 512, 5% 홀드아웃 손실로 early stopping.
  시드 2개를 앙상블.
- **재학습 주기**: 색인과 **같은 회차**에 다시 학습해야 한다. 학습 때 없던 글은 잘 못 찾는다
  (실험: 신규 글 R@10 26.6% 대 기존 글 47.8%).

## 3. 파이프라인

`pipeline/embed_articles.py small` → `build_index.py`로 `data/index/doc_meta.json`을 만든 뒤:

```bash
bash pipeline/student/run.sh                  # 인자 없음 — data/articles, data/index, data/student
BUILD_LEXICON=1 bash pipeline/student/run.sh  # 확장 어휘까지 (처음 한 번, 이후엔 드물게)
```

| 단계 | 스크립트 | 산출물 |
|---|---|---|
| 1 | `build_queries.mjs` (Node, garu) | 어휘·DF, 학습 쿼리, 평가셋 |
| 2 | `embed_teacher.py` (GPU 권장) | 교사 문서 벡터, 입력 어휘 표(PCA 384 int8), 학습 쿼리 교사 임베딩 |
| 2b | `build_lexicon_candidates.mjs` → `build_lexicon.py` (`BUILD_LEXICON=1`) | 후보 다운로드 → 교사 임베딩 → 코퍼스 친화도 필터 → 고정 입력 기저 → 묶음 |
| 3 | `train.py` | 체크포인트 `data/student/checkpoints/` + 자산 `data/index/student_*` |
| 4 | `evaluate.py` | `data/student/reports/eval-*.md` |

| 환경변수 | 기본값 | 의미 |
|---|---|---|
| `ARTICLES` / `INDEX` / `STUDENT_WORK` | `data/articles` / `data/index` / `data/student` | 입력·자산·작업 경로 |
| `STUDENT_EVAL_QUERIES` | — | 직접 쓴 평가 쿼리 `[{text, relevant:[id]}]` — 키워드로 안 잡히는 것만 dev/test로 반씩 |
| `STUDENT_OOV_QUERIES` | — | 코퍼스에 없는 단어 평가 쿼리 (같은 형식) |
| `STUDENT_SEEDS` | `2` | 앙상블 멤버 수 |
| `STUDENT_EVAL=0` | — | 평가 생략 |
| `KEEP_TRAIN_DATA=1` | — | 학습셋을 남김 (기본은 학습 후 삭제) |
| `BUILD_LEXICON=1` | — | 확장 어휘 재생성 |
| `PYTHON` / `NODE` | `python3` / `node` | 인터프리터 |

- 체크포인트만으로 자산을 다시 만들 수 있다(학습셋·GPU 불필요):
  `python pipeline/student/train.py --export-only`
- 평가는 GPU가 없으면 `python pipeline/student/evaluate.py --skip-models`(학생 행만).
- 확장 어휘 후보 출처는 빌드할 때 내려받고 레포·npm 패키지에는 넣지 않는다:
  [hermitdave/FrequencyWords](https://github.com/hermitdave/FrequencyWords)의 OpenSubtitles 2018 한국어 빈도 목록
  (내용 CC BY-SA 4.0), [한국어 위키백과](https://dumps.wikimedia.org/kowiki/) 문서 제목(CC BY-SA 4.0).
  만든 묶음을 배포할 때는 출처 표기를 권장한다.
- 입력 기저는 확장 어휘를 만들 때 고정되고(`data/student/lexicon/basis.npz`) 이후 재학습이 재사용한다.
  그래서 묶음은 확장 어휘를 다시 만들 때만 바뀐다.

## 4. 관련도 게이트 보정

브라우저는 시맨틱 결과의 최고 코사인이 문턱 이상일 때만 결과를 섞는다. 점수만으로는 관련·무관
쿼리를 가를 수 없어서 **슬롯 구성**으로 문턱을 나눈다.

- 어휘나 확장 어휘에 정확히 걸린 토큰이 있음 → `exact` (기본 0.3)
- 부분문자열 폴백으로만 걸림 → `fallbackOnly` (기본 0.635)
- 아무것도 안 걸림 → 시맨틱 생략(키워드만)

기본값은 한 코퍼스(경제·정치 칼럼 800편)에서 보정한 것이다. 내 코퍼스로 다시 맞추려면:

```bash
node pipeline/student/calib_gate.mjs --eval my_queries.json --offdomain offdomain.json [--oov oov.json]
```

`--offdomain`은 내 사이트가 답할 수 없는 실제 질문 목록(`["질문", …]`)이다. 스크립트가 제안한 값을
`new SearchController({ studentGate: { exact, fallbackOnly } })`에 넣는다.

## 5. 품질 (경제·정치 칼럼 800편 코퍼스)

| 방식 | 개념 test MRR | 제목 R@5 (%) | 신조어 R@10 (%) |
|---|---|---|---|
| 학생 + 확장 어휘 (약 6MB) | 0.621 | 79.2 | 57.5 |
| 참고: zipge 0.1의 브라우저 모델 (e5-small 37MB, 제거됨) | 0.530 | 88.3 | 62.5 |
| 참고: bge-m3 직접 (서버 임베딩 상한) | 0.816 | 88.3 | — |

개념은 키워드로 안 잡히는 손제작 쿼리 122개, 신조어는 코퍼스에 없는 단어(속어·신조어) 40개다.
제목형은 실제 앱에서 키워드 계층이 먼저 잡아준다.

**한계**: 단어 가중합 + 선형 투영이라 조합으로만 생기는 의미("모두에게 똑같이 돈을 주는 정책" → 기본소득)는
완전히 만들지 못한다. 학습에는 GPU를 권장한다(RTX 3060 기준 문서 임베딩 약 4분, 시드당 학습 1~2분).

## 6. 런타임 명세

`src/core.ts`의 `loadStudent` / `studentQuery` / `studentSearch`와 `pipeline/student/common.py`의
`StudentRuntime`이 같은 규칙을 구현한다(평가는 배포 파일을 이 규칙 그대로 채점한다).

### 6.1 토큰 (`src/korean.ts studentTokens` = `build_queries.mjs tokenize`)
```
cleaned = text.trim();  lower = cleaned.toLowerCase()
nouns   = garu.nouns(cleaned, { includeSL: true }).map(toLowerCase)     (garu가 없으면 [])
toks    = 순서 유지, 처음 나온 kind 유지, 중복 제거:
            (nouns.length ? nouns : lower.split(/\s+/))      kind n  (부분문자열 폴백 허용)
            각 어절 e (길이 ≥ 2): e, stripParticle(e)         kind e  (정확 일치만)
            lower에서 공백 제거한 전체 쿼리                    kind j  (정확 일치만)
```
`stripParticle`·조사 목록은 `src/korean.ts`와 `build_queries.mjs`가 글자까지 같아야 한다.

### 6.2 슬롯
```
for (t, kind) in toks:   (최대 max_slots = 16)
  1) t in terms                         -> core 슬롯
  2) 확장 어휘에 t가 있음                -> lexicon 슬롯 (그 조각의 벡터·scale)
  3) kind == n 이고 길이 >= 3: 최장 부분문자열(길이 ≥ 절반, 같은 길이는 왼쪽부터)이 terms에 있음 -> core 슬롯, fb = 1
  같은 core 행·같은 lexicon 단어는 한 번만
```

### 6.3 인코딩 (멤버 m)
```
w_slot  = core:    term_scale[m][row] · (fb ? fb_weight[m] : 1)
          lexicon: lex_scale[단어] · ext_weight[m]
p[d]    = Σ_slots  vec_i8[d] · w_slot                                        d < dim (384)
q_m[o]  = proj_scale[m][o] · Σ_d proj_i8[m][o][d] · p[d] + proj_bias[m][o]    o < out_dim (1024)
q_m /= ‖q_m‖;   q = Σ_m q_m;   q /= ‖q‖
score_i = doc_scale[i] · Σ_o doc_i8[i][o] · q[o]
```
학생 문서 벡터의 순서는 `student_meta.json.doc_ids`이고, 런타임은 이를 `doc_meta.json`과 **id로** 맞춘다.

### 6.4 자산 포맷 (리틀엔디언, 색인 디렉토리)
| 파일 | 내용 |
|---|---|
| `student_meta.json` | `{version: 2, teacher, dim, out_dim, members, vocab_size, max_slots, fallback:{min_len,min_ratio}, params_layout, terms[], doc_ids[], config, trained, basis_sha1, …}` |
| `student_vocab_i8.bin` | int8 `[vocab_size][dim]` |
| `student_proj_i8.bin` | int8 `[members][out_dim][dim]` |
| `student_params_f32.bin` | float32, 멤버마다 `term_scale[vocab_size]`, `proj_scale[out_dim]`, `proj_bias[out_dim]`, `fb_weight[1]`, `ext_weight[1]` |
| `student_doc_vectors_i8.bin` | int8 `[N][out_dim]` |
| `student_doc_scales.json` | `{model, dim, count, scales[N]}` |
| `student_lex_meta.json` (선택) | `{version: 2, dim, count, shards, hash: "fnv1a32-utf16", basis_sha1, packs[], index[shards], teacher, aff, sources}` |
| `student_lex_K.bin` (선택) | 연속한 조각을 이어 붙인 묶음(4MB 이하). 조각 `i`는 `index[i] = [K, start, length]` |

- 조각 `i` = `fnv1a32(단어) % shards`, 내용은 `u32 count`, `u32 json_len`, UTF-8 JSON `{terms[], scales[]}`,
  int8 `[count][dim]`. `fnv1a32`: h = 0x811C9DC5; UTF-16 code unit마다 `h ^= cu; h = (h · 0x01000193) mod 2³²`.
- 런타임은 `Range: bytes=start-(start+length-1)`로 조각만 받는다. Range를 무시하는 서버면 묶음 전체에서 잘라 쓴다.
- 확장 어휘는 **`student_lex_meta.basis_sha1 == student_meta.basis_sha1`일 때만** 쓴다.
- 조각을 파일 256개로 두지 않는 것은 정적 호스트·자산 저장소의 파일 수 제한과 업로드 크기 제한을 피하기 위해서다.
