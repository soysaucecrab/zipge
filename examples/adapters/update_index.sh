#!/usr/bin/env bash
# One-command index refresh: DB fetch -> export -> embeddings -> graph -> index -> student encoder -> frontend copy.
# Usage:
#   scripts/update_index.sh            # full run (prompts for the server SSH password)
#   SKIP_FETCH=1 scripts/update_index.sh   # reuse the existing local data/data.db
set -euo pipefail
cd "$(dirname "$0")/.."

PY=.venv/bin/python
FRONTEND=miseskorea-frontend

if [[ "${SKIP_FETCH:-}" != "1" ]]; then
  echo "==> fetching data.db from server (read-only)"
  scp miseskorea@167.179.89.118:pocketbase/pb_data/data.db data/data.db
fi

echo "==> exporting articles"
$PY scripts/export_articles.py

echo "==> embedding (base: graph kNN / small: search index)"
$PY scripts/embed_articles.py base
$PY scripts/embed_articles.py small

echo "==> building graph"
$PY scripts/build_graph.py

echo "==> building index (node2vec retrain + related BFS included)"
(cd scripts && ../$PY build_index.py)

echo "==> graph layout + admin viz"
$PY scripts/graph_layout.py
$PY scripts/build_viz.py

echo "==> student encoder (bge-m3 teacher, GPU recommended)"
[[ -e node_modules ]] || ln -s "$FRONTEND/node_modules" node_modules   # garu-ko for the Node steps
PYTHON=$PY bash scripts/student/run.sh

echo "==> copying into frontend"
cp data/index/doc_meta.json data/index/tags.json data/index/student_* \
   "$FRONTEND/static/search-index/"
cp data/index/related_links.json "$FRONTEND/src/lib/server/data/"

echo "==> keyword fallback dictionary (WASM 차단 환경용 사전)"
(cd "$FRONTEND" && node pipeline/build_terms.js)

echo "==> done. next steps:"
echo "    1) cd $FRONTEND && git diff --stat   # review"
echo "    2) commit & push the branch, then rebuild/deploy the image"
