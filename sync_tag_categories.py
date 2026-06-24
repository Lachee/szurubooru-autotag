#!/usr/bin/env python3
"""
Sync tag categories from tag_categories.json into a szurubooru instance.

Usage:
    TOKEN=xxx python3 sync_tag_categories.py

Environment variables:
    SZURU_URL    - base URL, e.g. http://10.0.50.10:8033
    SZURU_USER   - username
    TOKEN        - API token or password
"""

import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from szurubooru import make_headers

RESET  = "\033[0m"
BOLD   = "\033[1m"
DIM    = "\033[2m"
CYAN   = "\033[36m"
GREEN  = "\033[32m"
YELLOW = "\033[33m"
RED    = "\033[31m"
GRAY   = "\033[90m"

BASE_URL     = os.getenv("SZURU_URL",  "http://10.0.50.10:8033")
API_USERNAME = os.getenv("SZURU_USER", "lachee")
API_TOKEN    = os.getenv("TOKEN")

# Map numeric category IDs from tag_categories.json to szurubooru category names
CATEGORY_MAP = {
    0: "default",    # general
    1: "artist",
    2: "artist",     # colorist → artist
    3: "copyright",
    4: "character",
    5: "species",
    6: "default",    # invalid/meta
    7: "meta",
    8: "lore",
    9: "accessory",
}

# Categories to create if missing, with a color and sort order
CATEGORY_DEFAULTS = {
    "default":   {"color": "#ffffff", "order": 0},
    "artist":    {"color": "#f4ac42", "order": 1},
    "copyright": {"color": "#dd55dd", "order": 2},
    "character": {"color": "#55dd55", "order": 3},
    "species":   {"color": "#ed5d1f", "order": 4},
    "meta":      {"color": "#aaaaaa", "order": 5},
    "lore":      {"color": "#0055dd", "order": 6},
    "accessory": {"color": "#dd5577", "order": 7},
}

BATCH_SIZE = 100


def api_request(headers: dict, method: str, url: str, body: dict | None = None):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as ex:
        text = ex.read().decode("utf-8", errors="replace")
        print(f"\n{RED}HTTP {ex.code}{RESET}: {text[:300]}", file=sys.stderr)
        return None


def fetch_szuru_categories(base: str, headers: dict) -> dict:
    result = api_request(headers, "GET", f"{base}/api/tag-categories")
    return {c["name"]: c for c in (result or {}).get("results", [])}


def ensure_categories(base: str, headers: dict, needed: set[str], existing: dict) -> None:
    for name in needed:
        if name in existing:
            continue
        defaults = CATEGORY_DEFAULTS.get(name, {"color": "#aaaaaa", "order": 99})
        print(f"  {YELLOW}Creating category{RESET} {BOLD}{name!r}{RESET} ({defaults['color']})")
        api_request(headers, "POST", f"{base}/api/tag-categories", {
            "name": name,
            "color": defaults["color"],
            "order": defaults["order"],
        })


def fetch_tags_page(base: str, headers: dict, offset: int) -> dict:
    params = urllib.parse.urlencode({
        "offset": offset,
        "limit": BATCH_SIZE,
        "fields": "names,category,version",
    })
    return api_request(headers, "GET", f"{base}/api/tags?{params}") or {}


def update_tag_category(base: str, headers: dict, name: str, version: int, category: str):
    return api_request(headers, "PUT", f"{base}/api/tag/{urllib.parse.quote(name, safe='')}", {
        "version": version,
        "category": category,
    })


def main():
    if not API_TOKEN:
        print(f"{RED}TOKEN env var is required{RESET}", file=sys.stderr)
        sys.exit(1)

    headers = make_headers(API_USERNAME, API_TOKEN)
    base = BASE_URL.rstrip("/")

    print(f"{CYAN}{BOLD}Loading tag_categories.json...{RESET}")
    with open("tag_categories.json", encoding="utf-8") as f:
        tag_categories: dict[str, int] = json.load(f)
    print(f"  {len(tag_categories):,} tags loaded")

    needed_szuru_cats = set(CATEGORY_MAP[v] for v in tag_categories.values() if v in CATEGORY_MAP)

    print(f"\n{CYAN}{BOLD}Fetching existing szurubooru categories...{RESET}")
    existing_cats = fetch_szuru_categories(base, headers)
    print(f"  Found: {', '.join(existing_cats) or '(none)'}")

    ensure_categories(base, headers, needed_szuru_cats, existing_cats)

    # First page to get total count
    print(f"\n{CYAN}{BOLD}Fetching tags from szurubooru...{RESET}")
    first = fetch_tags_page(base, headers, 0)
    total = first.get("total", 0)
    print(f"  {total:,} tags in instance\n")

    updated = 0
    skipped = 0
    unknown = 0
    offset = 0
    page_results = first.get("results", [])

    while True:
        for tag in page_results:
            name = tag["names"][0]
            current_cat = tag.get("category", "default")
            version = tag["version"]

            # normalise spaces → underscores to match our lookup key
            lookup = name.replace("_", " ")
            cat_id = tag_categories.get(lookup)
            if cat_id is None:
                unknown += 1
                continue

            desired_cat = CATEGORY_MAP.get(cat_id, "default")
            if desired_cat == current_cat:
                skipped += 1
                continue

            result = update_tag_category(base, headers, name, version, desired_cat)
            if result:
                updated += 1
                print(f"  {DIM}{name!r}{RESET} {GRAY}{current_cat} → {RESET}{GREEN}{desired_cat}{RESET}")
            else:
                print(f"  {RED}failed{RESET} {name!r}")

        offset += len(page_results)
        print(f"{DIM}  {offset}/{total} processed — {updated} updated, {skipped} unchanged, {unknown} unknown{RESET}", end="\r")

        if offset >= total:
            break
        page_results = fetch_tags_page(base, headers, offset).get("results", [])
        if not page_results:
            break

    print(f"\n\n{GREEN}{BOLD}Done.{RESET} {updated} updated, {skipped} already correct, {unknown} not in vocab.")


if __name__ == "__main__":
    main()
