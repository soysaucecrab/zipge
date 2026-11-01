#!/usr/bin/env bash
# 학생 인코더(모델 없는 쿼리 인코더) — 쿼리 생성 → 교사 임베딩 → 증류 학습 → 평가.
# 교사 BAAI/bge-m3 (GPU 권장). build_index.py가 data/index를 만든 뒤 같은 회차에 돌린다 —
# 학생은 학습 때 본 문서에만 맞춰지므로 글이 바뀌면 색인과 함께 다시 학습해야 한다.
#
#   bash pipeline/student/run.sh            (레포 루트에서, 인자 없음)
#
# 환경변수
#   PYTHON / NODE        인터프리터 (기본 python3 / node)
#   ARTICLES             문서 JSON 디렉토리 (기본 data/articles)
#   INDEX                doc_meta.json이 있는 색인 = 자산 출력 디렉토리 (기본 data/index)
#   STUDENT_WORK         중간 산출물·체크포인트·리포트 (기본 data/student)
#   STUDENT_EVAL_QUERIES 직접 쓴 평가 쿼리 [{text, relevant:[id]}] (선택 — dev/test로 반씩 나뉜다)
#   STUDENT_OOV_QUERIES  코퍼스 밖 단어 평가 쿼리, 같은 형식 (선택)
#   STUDENT_SEEDS        앙상블 멤버 수 (기본 2)
#   STUDENT_EVAL=0       평가 생략
#   KEEP_TRAIN_DATA=1    학습셋(학습 쿼리 + 교사 쿼리 임베딩)을 남김 — 기본은 학습 후 삭제
#   BUILD_LEXICON=1      코퍼스 밖 확장 어휘를 다시 만듦(자막·위키 후보 다운로드 → 교사 임베딩 →
#                        고정 입력 기저 → 묶음). 처음 한 번, 이후엔 드물게만
set -euo pipefail

PYTHON=${PYTHON:-python3}
NODE=${NODE:-node}
# 상대 경로 인터프리터(.venv/bin/python 등)도 cd 이후 동작하도록 절대 경로로 고정
[[ "$PYTHON" == */* && "$PYTHON" != /* ]] && PYTHON="$PWD/$PYTHON"
[[ "$NODE" == */* && "$NODE" != /* ]] && NODE="$PWD/$NODE"
HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/../.." && pwd)
ARTICLES=${ARTICLES:-$ROOT/data/articles}
INDEX=${INDEX:-$ROOT/data/index}
WORK=${STUDENT_WORK:-$ROOT/data/student}
mkdir -p "$WORK"

EVAL_ARGS=()
[[ -n "${STUDENT_EVAL_QUERIES:-}" ]] && EVAL_ARGS+=(--eval "$STUDENT_EVAL_QUERIES")
[[ -n "${STUDENT_OOV_QUERIES:-}" ]] && EVAL_ARGS+=(--oov "$STUDENT_OOV_QUERIES")

echo "==> student 1/4: vocabulary + training queries + eval sets"
"$NODE" "$HERE/build_queries.mjs" --articles "$ARTICLES" --index "$INDEX" --work "$WORK" ${EVAL_ARGS[@]+"${EVAL_ARGS[@]}"}

echo "==> student 2/4: teacher embeddings"
"$PYTHON" "$HERE/embed_teacher.py" --articles "$ARTICLES" --index "$INDEX" --work "$WORK"

if [[ "${BUILD_LEXICON:-0}" == "1" ]]; then
  echo "==> student 2b: extension lexicon (out-of-corpus words, fixed input basis)"
  "$NODE" "$HERE/build_lexicon_candidates.mjs" --src "$WORK/lexicon/src" --out "$WORK/lexicon/candidates.json"
  "$PYTHON" "$HERE/build_lexicon.py" --work "$WORK" --out "$INDEX"
fi

echo "==> student 3/4: distillation + asset export"
"$PYTHON" "$HERE/train.py" --work "$WORK" --index "$INDEX" --seeds "${STUDENT_SEEDS:-2}"

if [[ "${STUDENT_EVAL:-1}" != "0" ]]; then
  echo "==> student 4/4: evaluation report"
  "$PYTHON" "$HERE/evaluate.py" --work "$WORK" --index "$INDEX"
fi

# 학습셋은 남기지 않는다(체크포인트·리포트·교사 문서/어휘 벡터만 유지 — train.py --export-only 재생성에 충분)
if [[ "${KEEP_TRAIN_DATA:-0}" != "1" ]]; then
  rm -f "$WORK/train_queries.json" "$WORK/teacher_q_train.npy"
  echo "    removed training set (KEEP_TRAIN_DATA=1 to keep)"
fi
