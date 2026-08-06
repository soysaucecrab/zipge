"""새 글 감지 — 인덱스에 실제로 들어가는 내용 필드만 해시해 비교한다.

  python3 pipeline/check_source.py          # 변경 있으면 "changed", 없으면 "unchanged" 출력
  python3 pipeline/check_source.py --write  # 현재 해시를 저장 (재학습 마지막 단계)

해시 파일(static/search-index/source_hash.txt)은 인덱스와 함께 PR에 실려,
"이 인덱스가 어느 시점의 글 내용으로 만들어졌나"의 마커가 된다.

updated 타임스탬프는 쓰지 않는다 — 조회수 증가만으로도 갱신되어 매시간
헛발동한다. slug 변경·조회수 변동·썸네일 교체는 의도적으로 무시하고
(재학습 사유 아님), 제목·본문·태그·게시판·분류·미리보기의 실제 변화와
글 추가·삭제만 잡는다.
"""
import hashlib
import json
import sys
import urllib.parse
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
MARKER = REPO / "static" / "search-index" / "source_hash.txt"
API = "https://miseskorea.org/api/collections/articles/records"

CONTENT_FIELDS = ["id", "title", "content", "tags", "board", "category", "preview"]


def current_hash() -> str:
    lines, page = [], 1
    while True:
        q = urllib.parse.urlencode({
            "filter": "(status='published')",
            "fields": ",".join(CONTENT_FIELDS),
            "sort": "id", "page": page, "perPage": 200,
        })
        req = urllib.request.Request(f"{API}?{q}", headers={"User-Agent": "index-refresh"})
        with urllib.request.urlopen(req, timeout=120) as r:
            data = json.load(r)
        for item in data["items"]:
            payload = json.dumps({k: item.get(k) for k in CONTENT_FIELDS},
                                 ensure_ascii=False, sort_keys=True)
            lines.append(hashlib.sha256(payload.encode()).hexdigest())
        if page >= data["totalPages"]:
            break
        page += 1
    return hashlib.sha256("\n".join(sorted(lines)).encode()).hexdigest()


def main():
    h = current_hash()
    if "--write" in sys.argv:
        MARKER.write_text(h + "\n", encoding="utf-8")
        print(f"wrote {h[:12]}")
        return
    stored = MARKER.read_text(encoding="utf-8").strip() if MARKER.exists() else ""
    print("changed" if h != stored else "unchanged")


if __name__ == "__main__":
    main()
