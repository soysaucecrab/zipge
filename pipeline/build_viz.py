"""Build the graph visualization in two forms.

1. data/graph_viz.html — data inlined (artifact / local file viewing)
2. miseskorea-frontend/static/admin/ — fetch mode: graph.html + graph-data.json.
   The page polls graph-data.json for changes, so redeploying a fresh
   index refreshes any open admin screen automatically.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
tpl = (ROOT / "scripts" / "viz_template.html").read_text(encoding="utf-8")
data = (ROOT / "data" / "graph" / "layout.json").read_text(encoding="utf-8")

out_path = ROOT / "data" / "graph_viz.html"
out_path.write_text(tpl.replace("/*__DATA__*/null", data), encoding="utf-8")
print(f"{out_path} ({out_path.stat().st_size // 1024}KB)")

admin = ROOT / "miseskorea-frontend" / "static" / "admin"
if admin.parent.exists():
    admin.mkdir(exist_ok=True)
    page = "<!doctype html>\n<html lang=\"ko\">\n<meta charset=\"utf-8\">\n" + tpl
    (admin / "graph.html").write_text(page, encoding="utf-8")
    (admin / "graph-data.json").write_text(data, encoding="utf-8")
    print(f"{admin}/graph.html + graph-data.json")
