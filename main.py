from taggerine.inference_tagger_standalone import Tagger, _fmt_json
from szurubooru import fetch_untagged
import base64
import json
import os

# API
BASE_URL = "http://10.0.50.10:8033/"
API_USERNAME = "lachee"
API_TOKEN = os.getenv("TOKEN")

# Tagging
#  cpu, cuda, ipu, xpu, mkldnn, opengl, opencl, ideep, hip, ve, fpga, maia, xla, lazy, vulkan, mps, meta, hpu, mtia, privateuseone
DEVICE = 'cuda'
TOPK = 30
THRESHOLD = 0.9

def main():
    # Load all images that need tagging
    creds = base64.b64encode(f"{API_USERNAME}:{API_TOKEN}".encode()).decode()
    response = fetch_untagged(BASE_URL, {
        "Authorization": f"Token {creds}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    },  0, 15)
    print(f"Found {response['total']} untagged post(s).")

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
    for post in response["results"]:
        print(f"Tagging #{post['id']}: {post["thumbnailUrl"]}...")
        src = f"{BASE_URL.rstrip('/')}/{post["thumbnailUrl"]}"
        results = tagger.predict(src, topk=topk, threshold=threshold)
        all_results.append(_fmt_json(src, results))

    print(json.dumps(all_results, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()