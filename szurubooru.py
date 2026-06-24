#!/usr/bin/env python3
"""
Find posts without tags in a szurubooru instance, 10 at a time.

Usage:
    python3 find-untagged-posts.py [URL] [--user USER] [--token TOKEN] [--batch N]

Environment variables (alternative to flags):
    SZURU_URL    - base URL of the instance, e.g. http://localhost:6666
    SZURU_USER   - username
    SZURU_TOKEN  - API token (from user settings) or password
"""

import argparse
import base64
import json
import os
import sys
import urllib.error
import urllib.parse
import urllib.request


def make_headers(user: str, token: str) -> dict:
    creds = base64.b64encode(f"{user}:{token}".encode()).decode()
    return {
        "Authorization": f"Token {creds}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }


def fetch_untagged(base_url: str, headers: dict, offset: int, limit: int) -> dict:
    params = urllib.parse.urlencode({
        "query": "tag-count:0",
        "offset": offset,
        "limit": limit,
        "fields": "id,thumbnailUrl,tags,score,creationTime",
    })
    url = f"{base_url.rstrip('/')}/api/posts?{params}"
    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as ex:
        body = ex.read().decode("utf-8", errors="replace")
        print(f"HTTP {ex.code}: {body[:300]}", file=sys.stderr)
        sys.exit(1)
    except urllib.error.URLError as ex:
        print(f"Connection error: {ex.reason}", file=sys.stderr)
        sys.exit(1)


def print_batch(posts: list, base_url: str, offset: int) -> None:
    base = base_url.rstrip("/")
    for i, post in enumerate(posts, start=offset + 1):
        post_id = post["id"]
        score = post.get("score", 0)
        created = (post.get("creationTime") or "")[:10]
        url = f"{base}/post/{post_id}"
        print(f"  [{i:>4}] #{post_id:>6}  score={score:>4}  created={created}  {url}")

def main() -> None:
    parser = argparse.ArgumentParser(description="Find untagged posts in szurubooru.")
    parser.add_argument("url", nargs="?", default=os.environ.get("SZURU_URL", ""),
                        help="Base URL of the szurubooru instance")
    parser.add_argument("--user", default=os.environ.get("SZURU_USER", ""),
                        help="Username")
    parser.add_argument("--token", default=os.environ.get("SZURU_TOKEN", ""),
                        help="API token or password")
    parser.add_argument("--batch", type=int, default=10,
                        help="Number of posts per page (default: 10)")
    args = parser.parse_args()

    if not args.url:
        parser.error("URL is required (arg or SZURU_URL env var)")
    if not args.user or not args.token:
        parser.error("--user and --token are required (or SZURU_USER / SZURU_TOKEN env vars)")

    headers = make_headers(args.user, args.token)
    offset = 0
    batch = args.batch

    # fetch first page to get total
    data = fetch_untagged(args.url, headers, 0, batch)
    total = data.get("total", 0)

    if total == 0:
        print("No untagged posts found.")
        return

    print(f"Found {total} untagged post(s). Showing {batch} at a time.\n")

    while True:
        if offset > 0:
            data = fetch_untagged(args.url, headers, offset, batch)

        posts = data.get("results", [])
        if not posts:
            print("No more posts.")
            break

        end = min(offset + len(posts), total)
        print(f"--- Posts {offset + 1}–{end} of {total} ---")
        print_batch(posts, args.url, offset)

        offset += len(posts)
        if offset >= total:
            print("\nAll untagged posts listed.")
            break

        try:
            choice = input("\n[Enter] next batch, [q] quit: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if choice == "q":
            break


if __name__ == "__main__":
    main()
