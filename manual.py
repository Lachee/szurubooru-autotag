from taggerine.inference_tagger_standalone import Tagger, _fmt_json
from szurubooru import fetch_untagged, update_post_tags, fetch_posts, fetch_tag_implications
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


def main():
    # Load all images that need tagging
    creds = base64.b64encode(f"{USER}:{TOKEN}".encode()).decode()
    auth_header = {
        "Authorization": f"Token {creds}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }

    response = fetch_untagged(BASE_URL, auth_header,  0, LIMIT)

    total = len(response["results"])
    print(f"{CYAN}{BOLD}Found {response['total']} post(s) to tag. Going to tag {total}.{RESET}")
    if total == 0:
        print(f"{GREEN}{BOLD}No untagged posts found.{RESET}")
        return

    # initialise the tagger with all the good sshtuff
    tagger = Tagger(
        checkpoint_path=CHECKPOINT,
        vocab_path=VOCAB,
        device=DEVICE,
        max_size=1024,
    )

    topk, threshold = (
        (None, THRESHOLD) if THRESHOLD else (TOPK, None)
    )

    implications_cache: dict[str, list[str]] = {}

    for i, post in enumerate(response["results"], 1):
        print(f"{DIM}{i}/{total}{RESET} {BOLD}#{post['id']}{RESET} {GRAY}{post['thumbnailUrl']}{RESET} ...", end=" ", flush=True)

        results = tagger.predict(f"{BASE_URL.rstrip('/')}/{post['thumbnailUrl']}", topk=topk, threshold=threshold)
        tags = [t.replace(" ", "_") for t, _ in results]
        print(f"{DIM}{GREEN}({len(results)} tags)", end=" ", flush=True)

        tags = resolve_implications(tags, implications_cache, BASE_URL, {
            "Authorization": f"Token {creds}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        })
        print(f"{DIM}{YELLOW}({len(tags) - len(results)} implied){RESET}", end=" ", flush=True)

        update_post_tags(BASE_URL, auth_header, post["id"], tags)
        print(f"{YELLOW}- {len(tags)} total tags{RESET}")

    print(f"\n{GREEN}{BOLD}Done.{RESET}")

def fix_implications():
    creds = base64.b64encode(f"{API_USERNAME}:{API_TOKEN}".encode()).decode()
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
    import sys
    if "--fix-implications" in sys.argv:
        fix_implications()
    else:
        main()