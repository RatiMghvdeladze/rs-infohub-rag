import json
import time
import re
import requests
import urllib3
from pathlib import Path

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

API_URL = "https://infohubapi.rs.ge/api/documents"
SITE_BASE = "https://infohub.rs.ge"

MAX_PAGES = 1030
TAKE = 10

OUT_JSONL = Path("data/raw_docs.jsonl")   # ინახავს ხაზ-ხაზ JSON ობიექტებს
STATE_FILE = Path("data/state.json")     # ინახავს რომელ გვერდზე ვართ გაჩერებული

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) InfoHub-DL/1.0",
    "Accept": "application/json, text/plain, */*",
    "Origin": "https://infohub.rs.ge",
    "Referer": "https://infohub.rs.ge/",
    "languagecode": "ka",
}

UUID_RE = re.compile(
    r"^[0-9a-fA-F]{8}-"
    r"[0-9a-fA-F]{4}-"
    r"[0-9a-fA-F]{4}-"
    r"[0-9a-fA-F]{4}-"
    r"[0-9a-fA-F]{12}$"
)

def extract_items(data):
    if data is None:
        return []
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for key in ["items", "data", "results", "result", "content"]:
            val = data.get(key)
            if isinstance(val, list):
                return val
        for key in ["payload", "response", "model"]:
            val = data.get(key)
            if isinstance(val, dict):
                for k2 in ["items", "data", "results", "content"]:
                    v2 = val.get(k2)
                    if isinstance(v2, list):
                        return v2
    return []

def find_uuid_anywhere(obj):
    if isinstance(obj, dict):
        for v in obj.values():
            u = find_uuid_anywhere(v)
            if u:
                return u
    elif isinstance(obj, list):
        for v in obj:
            u = find_uuid_anywhere(v)
            if u:
                return u
    elif isinstance(obj, str):
        s = obj.strip()
        if UUID_RE.match(s):
            return s
    return None

def build_best_public_url(item):
    uuid = find_uuid_anywhere(item)
    if uuid:
        return f"{SITE_BASE}/ka/workspace/document/{uuid}", uuid

    name_link = item.get("nameLink")
    if isinstance(name_link, str) and name_link.strip():
        return f"{SITE_BASE}{name_link.strip()}", None

    doc_id = item.get("id")
    if doc_id is not None:
        return f"{SITE_BASE}/ka/document/{doc_id}", None

    return f"{SITE_BASE}/ka", None

def fetch_page(skip, take, retries=5):
    params = {"skip": skip, "take": take, "species": "NewDocument"}
    backoff = 1.0
    for attempt in range(1, retries + 1):
        try:
            r = requests.get(API_URL, params=params, headers=HEADERS, timeout=40, verify=False)
            if r.status_code == 200:
                return r.json()
            print(f"⚠️ HTTP {r.status_code} | {r.url}")
        except Exception as e:
            print(f"⚠️ fetch error (attempt {attempt}/{retries}): {e}")

        time.sleep(backoff)
        backoff = min(backoff * 2, 20)

    return None

def load_state():
    if STATE_FILE.exists():
        try:
            return json.loads(STATE_FILE.read_text(encoding="utf-8"))
        except:
            pass
    return {"next_page": 1}

def save_state(next_page):
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    STATE_FILE.write_text(json.dumps({"next_page": next_page}, ensure_ascii=False, indent=2), encoding="utf-8")

def main():
    OUT_JSONL.parent.mkdir(parents=True, exist_ok=True)

    state = load_state()
    start_page = int(state.get("next_page", 1))

    print(f"🚀 Start from page={start_page} / {MAX_PAGES} | TAKE={TAKE}")
    print(f"📄 Writing to: {OUT_JSONL}")

    with OUT_JSONL.open("a", encoding="utf-8") as out:
        for page in range(start_page, MAX_PAGES + 1):
            skip = (page - 1) * TAKE
            print(f"\n⏳ Page {page}/{MAX_PAGES} (skip={skip})...")

            data = fetch_page(skip, TAKE)
            items = extract_items(data)
            if not items:
                print("✅ empty page/items → stopping early.")
                save_state(page + 1)
                break

            added = 0
            for item in items:
                if not isinstance(item, dict):
                    continue

                doc_id = item.get("id")
                title = item.get("name") or item.get("title") or ""
                raw_html = item.get("additionalDescription") or item.get("description") or item.get("text") or ""

                url, uuid = build_best_public_url(item)

                doc = {
                    "id": doc_id,
                    "uuid": uuid,
                    "title": title,
                    "text_html": raw_html,
                    "url": url,
                    "species": item.get("species"),
                    "createDate": item.get("createDate"),
                    "updateDate": item.get("updateDate"),
                }

                out.write(json.dumps(doc, ensure_ascii=False) + "\n")
                added += 1

            print(f"✅ wrote {added} docs")
            save_state(page + 1)

            # rate-limit: საჭირო হলে გაზარდე 0.6-1.2
            time.sleep(0.4)

    print("\n🎉 Done (or stopped early).")
    print(f"Next run will continue from: {json.loads(STATE_FILE.read_text(encoding='utf-8'))['next_page']}")

if __name__ == "__main__":
    main()