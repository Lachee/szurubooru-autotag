from taggerine.inference_tagger_standalone import Tagger, _fmt_json
from szurubooru import update_post_tags, fetch_posts, fetch_tag_implications
import base64
import json
import os

RESET  = "\033[0m"
BOLD   = "\033[1m"
DIM    = "\033[2m"
CYAN   = "\033[36m"
GREEN  = "\033[32m"
YELLOW = "\033[33m"
GRAY   = "\033[90m"

# API
LIMIT = 10
RANGE_BATCH_LIMIT = 100

BASE_URL   = os.getenv("SZURU_URL", "http://localhost:8033")
USER       = os.getenv("SZURU_USER", "")
TOKEN      = os.getenv("SZURU_TOKEN", "")
DEVICE     = os.getenv("DEVICE", "cuda")
CHECKPOINT = os.getenv("CHECKPOINT", "taggerine/tagger_proto.safetensors")
VOCAB      = os.getenv("VOCAB", "taggerine/tagger_vocab_with_categories_and_alias_updated.json")

_threshold_env = os.getenv("THRESHOLD", "0.98")
THRESHOLD = float(_threshold_env) if _threshold_env else None
TOPK      = int(os.getenv("TOPK", "50"))

def resolve_implications(tags: list[str], cache: dict, base_url: str, headers: dict) -> list[str]:
    resolved = set(tags)
    queue = list(tags)
    while queue:
        tag = queue.pop()
        if tag not in cache:
            cache[tag] = fetch_tag_implications(base_url, headers, tag)
        for implied in cache[tag]:
            if implied not in resolved:
                resolved.add(implied)
                queue.append(implied)
    return sorted(resolved)


def tag_posts(query: str, batch_limit: int, description: str) -> None:
    creds = base64.b64encode(f"{USER}:{TOKEN}".encode()).decode()
    auth_header = {
        "Authorization": f"Token {creds}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }

    tagger = None
    topk, threshold = (
        (None, THRESHOLD) if THRESHOLD else (TOPK, None)
    )
    implications_cache: dict[str, list[str]] = {}

    total = None
    processed = 0
    offset = 0

    while True:
        response = fetch_posts(BASE_URL, auth_header, offset, batch_limit, query=query)

        if total is None:
            total = response["total"]
            print(f"{CYAN}{BOLD}Found {total} {description}.{RESET}")
            if total == 0:
                print(f"{GREEN}{BOLD}Nothing to tag.{RESET}")
                return
            # initialise the tagger with all the good sshtuff, lazily so we
            # don't pay startup cost when there's nothing to do
            tagger = Tagger(
                checkpoint_path=CHECKPOINT,
                vocab_path=VOCAB,
                device=DEVICE,
                max_size=1024,
            )

        posts = response["results"]
        if not posts:
            break

        for post in posts:
            processed += 1
            print(f"{DIM}{processed}/{total}{RESET} {BOLD}#{post['id']}{RESET} {GRAY}{post['thumbnailUrl']}{RESET} ...", end=" ", flush=True)

            results = tagger.predict(f"{BASE_URL.rstrip('/')}/{post['thumbnailUrl']}", topk=topk, threshold=threshold)
            tags = [t.replace(" ", "_") for t, _ in results]
            print(f"{DIM}{GREEN}({len(results)} tags)", end=" ", flush=True)

            tags = resolve_implications(tags, implications_cache, BASE_URL, auth_header)
            print(f"{DIM}{YELLOW}({len(tags) - len(results)} implied){RESET}", end=" ", flush=True)

            update_post_tags(BASE_URL, auth_header, post["id"], tags)
            print(f"{YELLOW}- {len(tags)} total tags{RESET}")

        offset += len(posts)
        if offset >= total:
            break

    print(f"\n{GREEN}{BOLD}Done.{RESET}")


def main(redo: bool = False, start: int | None = None, end: int | None = None):
    parts = []
    if start is not None:
        parts.append(f"id:{start}..{end}")
    if redo:
        parts.append("tag-count:1..")
    elif start is None:
        parts.append("tag-count:0")
    query = " ".join(parts)

    range_desc = f" in range {start}..{end}" if start is not None else ""
    description = f"{'already-tagged' if redo else 'untagged'} post(s){range_desc} to {'redo' if redo else 'tag'}"

    batch_limit = RANGE_BATCH_LIMIT if (redo or start is not None) else LIMIT
    tag_posts(query=query, batch_limit=batch_limit, description=description)

def fix_implications():
    creds = base64.b64encode(f"{USER}:{TOKEN}".encode()).decode()
    headers = {
        "Authorization": f"Token {creds}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }

    data = fetch_posts(BASE_URL, headers, 0, 1, query="tag-count:1..")
    total = data.get("total", 0)
    print(f"{CYAN}{BOLD}Checking implications on {total} tagged post(s)...{RESET}")

    implications_cache: dict[str, list[str]] = {}
    offset = 0
    checked = 0
    updated = 0

    while offset < total:
        data = fetch_posts(BASE_URL, headers, offset, LIMIT, query="tag-count:1..")
        posts = data.get("results", [])
        if not posts:
            break

        for post in posts:
            post_id = post["id"]
            current = [t["names"][0] for t in post.get("tags", [])]
            resolved = resolve_implications(current, implications_cache, BASE_URL, headers)
            new_implied = set(resolved) - set(current)
            if new_implied:
                update_post_tags(BASE_URL, headers, post_id, resolved)
                print(f"  {BOLD}#{post_id}{RESET} {GREEN}+{len(new_implied)} implied{RESET} {GRAY}{', '.join(sorted(new_implied))}{RESET}")
                updated += 1
            checked += 1

        offset += len(posts)
        print(f"{DIM}{checked}/{total}{RESET}", end="\r", flush=True)

    print(f"\n{GREEN}{BOLD}Done.{RESET} Checked {checked}, updated {updated}.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Auto-tag szurubooru posts.")
    parser.add_argument("--fix-implications", action="store_true",
                        help="Resolve missing tag implications on already-tagged posts")
    parser.add_argument("--redo", action="store_true",
                        help="Redo posts that already have tags instead of only untagged ones")
    parser.add_argument("--start", type=int, default=None,
                        help="Start post ID (inclusive). Tags posts id:start..end instead of scanning the whole instance")
    parser.add_argument("--end", type=int, default=None,
                        help="End post ID (inclusive). Requires --start")
    args = parser.parse_args()

    if args.start is None and args.end is not None:
        parser.error("--start and --end must be given together")
    if args.start is not None and args.end is None:
        parser.error("--start and --end must be given together")
    if args.start is not None and args.end is not None and args.start > args.end:
        parser.error("--start must be <= --end")

    if args.fix_implications:
        fix_implications()
    else:
        main(redo=args.redo, start=args.start, end=args.end)