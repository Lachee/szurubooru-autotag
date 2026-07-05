import serve
from szurubooru.api import fetch_posts, update_post_tags

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


def tag_posts(query: str, batch_limit: int, description: str) -> None:
    serve._headers = serve._make_headers()

    tagger = None
    total = None
    processed = 0
    offset = 0

    while True:
        response = fetch_posts(serve.BASE_URL, serve._headers, offset, batch_limit, query=query)

        if total is None:
            total = response["total"]
            print(f"{CYAN}{BOLD}Found {total} {description}.{RESET}")
            if total == 0:
                print(f"{GREEN}{BOLD}Nothing to tag.{RESET}")
                return
            # initialise the tagger with all the good sshtuff, lazily so we
            # don't pay startup cost when there's nothing to do
            tagger = serve.load_model()

        posts = response["results"]
        if not posts:
            break

        for post in posts:
            processed += 1
            print(f"{DIM}{processed}/{total}{RESET} {BOLD}#{post['id']}{RESET} {GRAY}{post['thumbnailUrl']}{RESET} ...", end=" ", flush=True)

            url = f"{serve.BASE_URL.rstrip('/')}/{post['thumbnailUrl']}"
            image = serve._download_image(url)
            results = tagger.tag(image)
            tags = [tag.name for tag in results.tags]
            print(f"{DIM}{GREEN}({len(tags)} tags)", end=" ", flush=True)

            tags = serve._resolve_implications(tags)
            print(f"{DIM}{YELLOW}({len(tags) - len(results.tags)} implied){RESET}", end=" ", flush=True)

            update_post_tags(serve.BASE_URL, serve._headers, post["id"], tags)
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
    serve._headers = serve._make_headers()

    data = fetch_posts(serve.BASE_URL, serve._headers, 0, 1, query="tag-count:1..")
    total = data.get("total", 0)
    print(f"{CYAN}{BOLD}Checking implications on {total} tagged post(s)...{RESET}")

    offset = 0
    checked = 0
    updated = 0

    while offset < total:
        data = fetch_posts(serve.BASE_URL, serve._headers, offset, LIMIT, query="tag-count:1..")
        posts = data.get("results", [])
        if not posts:
            break

        for post in posts:
            post_id = post["id"]
            current = [t["names"][0] for t in post.get("tags", [])]
            resolved = serve._resolve_implications(current)
            new_implied = set(resolved) - set(current)
            if new_implied:
                update_post_tags(serve.BASE_URL, serve._headers, post_id, resolved)
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
