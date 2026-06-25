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
BASE_URL = "http://10.0.50.10:8033/"
API_USERNAME = "lachee"
API_TOKEN = os.getenv("TOKEN")
LIMIT = 10

# Tagging
#  cpu, cuda, ipu, xpu, mkldnn, opengl, opencl, ideep, hip, ve, fpga, maia, xla, lazy, vulkan, mps, meta, hpu, mtia, privateuseone
DEVICE = 'cuda'
TOPK = 50
THRESHOLD = 0.98 # 0.85

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
    creds = base64.b64encode(f"{API_USERNAME}:{API_TOKEN}".encode()).decode()
    auth_header = {
        "Authorization": f"Token {creds}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }

    response = fetch_untagged(BASE_URL, auth_header,  0, LIMIT)

    total = len(response["results"])
    print(f"{CYAN}{BOLD}Found {response['total']} post(s) to tag. Going to tag {total}.{RESET}")

    # initialise the tagger with all the good sshtuff
    tagger = Tagger(
        checkpoint_path='taggerine/tagger_proto.safetensors',
        vocab_path='taggerine/tagger_vocab_with_categories_and_alias_updated.json',
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

if __name__ == "__main__":
    main()