#!/usr/bin/env bash
# CI용 전체 재학습 — 서버 접근 없이 공개 API만으로 인덱스 일체를 다시 만든다.
# 레포 루트에서 실행. 로컬 검증: PYTHON=.venv-경로 bash pipeline/ci_refresh.sh
set -euo pipefail

PYTHON=${PYTHON:-python3}
REPO=$(pwd)
WORK=${WORK_DIR:-/tmp/index-work}

# 파이프라인 스크립트가 가정하는 배치(<ROOT>/scripts + data + miseskorea-frontend)를 조립
rm -rf "$WORK"
mkdir -p "$WORK/data"
cp -r "$REPO/pipeline" "$WORK/scripts"
ln -sfn "$REPO" "$WORK/miseskorea-frontend"
cd "$WORK"

echo "==> export (public API)"
$PYTHON scripts/export_articles_api.py

echo "==> embeddings (full re-embed: base for graph, small for search)"
$PYTHON scripts/embed_articles.py base
$PYTHON scripts/embed_articles.py small

echo "==> graph + index (node2vec retrain + related BFS)"
$PYTHON scripts/build_graph.py
(cd scripts && $PYTHON build_index.py)

echo "==> admin graph layout"
$PYTHON scripts/graph_layout.py
$PYTHON scripts/build_viz.py

echo "==> copy into repo"
cp data/index/doc_meta.json data/index/doc_vectors_i8.bin \
   data/index/vector_scales.json data/index/tags.json \
   "$REPO/static/search-index/"
cp data/index/related_links.json "$REPO/src/lib/server/data/"

echo "==> fallback assets"
(cd "$REPO" && node pipeline/build_terms.js)
$PYTHON "$REPO/pipeline/build_fallback_vectors.py"

echo "==> source marker"
$PYTHON "$REPO/pipeline/check_source.py" --write

echo "==> done"
