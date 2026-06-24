from taggerine.inference_tagger_standalone import Tagger, _fmt_json
from szurubooru import fetch_untagged, update_post_tags, fetch_posts
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
LIMIT = 50

# Tagging
#  cpu, cuda, ipu, xpu, mkldnn, opengl, opencl, ideep, hip, ve, fpga, maia, xla, lazy, vulkan, mps, meta, hpu, mtia, privateuseone
DEVICE = 'cuda'
TOPK = 50
THRESHOLD = 0.98 # 0.85

def main():
    # Load all images that need tagging
    creds = base64.b64encode(f"{API_USERNAME}:{API_TOKEN}".encode()).decode()
    response = fetch_untagged(BASE_URL, {
        "Authorization": f"Token {creds}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    },  0, LIMIT)
    print(f"{CYAN}{BOLD}Found {response['total']} post(s) to tag.{RESET}")

    # INitialize the tagger
    tagger = Tagger(
        checkpoint_path='taggerine/tagger_proto.safetensors',
        vocab_path='taggerine/tagger_vocab_with_categories_and_alias_updated.json',
        device=DEVICE,
        max_size=1024,
    )

    topk, threshold = (
        (None, THRESHOLD) if THRESHOLD else (TOPK, None)
    )

    # Format the images
    all_results = []
    total = len(response["results"])
    for i, post in enumerate(response["results"], 1):
        print(f"{DIM}{i}/{total}{RESET} {BOLD}#{post['id']}{RESET} {GRAY}{post['thumbnailUrl']}{RESET} ...", end=" ", flush=True)
        src = f"{BASE_URL.rstrip('/')}/{post['thumbnailUrl']}"
        results = tagger.predict(src, topk=topk, threshold=threshold)
        print(f"{GREEN}({len(results)} tags){RESET}", end=" ", flush=True)
        tags = [ t.replace(" ", "_") for t, _ in results ]
        update_post_tags(BASE_URL, {
            "Authorization": f"Token {creds}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }, post["id"], tags)
        print(f"{DIM}uploaded{RESET}")

    print(f"\n{GREEN}{BOLD}Done.{RESET}")

if __name__ == "__main__":
    main()