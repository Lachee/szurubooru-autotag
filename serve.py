#!/usr/bin/env python3
"""
Webhook server for szurubooru autotagger.

Listens for post_create webhooks from szurubooru, queues new posts,
and tags them in the background without blocking incoming requests.

Environment variables:
    SZURU_URL    - szurubooru base URL  (default: http://localhost:8033)
    SZURU_USER   - username
    SZURU_TOKEN  - API token or password
    DEVICE       - torch device: cuda, cpu, etc.  (default: cuda)
    THRESHOLD    - tag confidence threshold  (default: 0.98)
    TOPK         - top-k tags when THRESHOLD is unset  (default: 50)
    CHECKPOINT   - .safetensors checkpoint path  (default: taggerine/tagger_proto.safetensors)
    VOCAB        - vocab JSON path  (default: taggerine/tagger_vocab_with_categories_and_alias_updated.json)
    HOST         - bind address  (default: 0.0.0.0)
    PORT         - bind port  (default: 8000)
"""

import asyncio
import base64
import logging
import os
from contextlib import asynccontextmanager
import uvicorn
from fastapi import FastAPI, Request, Response
from szurubooru.api import PostNotFoundError, fetch_tag_implications, get_post, update_post_tags
from models.tagger import Tagger
from PIL import Image
import requests
from io import BytesIO

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

BASE_URL        = os.getenv("SZURU_URL", "http://localhost:8033")
USER            = os.getenv("SZURU_USER", "")
TOKEN           = os.getenv("SZURU_TOKEN", "")
DEVICE          = os.getenv("DEVICE", "cuda")
HOST            = os.getenv("HOST", "0.0.0.0")
PORT            = int(os.getenv("PORT", "8000"))
RETRY_NOT_FOUND = int(os.getenv("RETRY_NOT_FOUND", "5"))

MODEL           = os.getenv("MODEL", "pixai")
THRESHOLD = float(os.getenv("THRESHOLD", "0.95"))

_headers: dict = {}
_tagger: Tagger | None = None
_queue: asyncio.Queue[int] = asyncio.Queue()
_implications_cache: dict[str, list[str]] = {}
_processed = 0
_failed = 0

def _make_headers() -> dict:
    creds = base64.b64encode(f"{USER}:{TOKEN}".encode()).decode()
    return {
        "Authorization": f"Token {creds}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }


def _resolve_implications(tags: list[str]) -> list[str]:
    resolved = set(tags)
    pending = list(tags)
    while pending:
        tag = pending.pop()
        if tag not in _implications_cache:
            _implications_cache[tag] = fetch_tag_implications(BASE_URL, _headers, tag)
        for implied in _implications_cache[tag]:
            if implied not in resolved:
                resolved.add(implied)
                pending.append(implied)
    return sorted(resolved)

def _download_image(url : str) -> Image.Image:
    response = requests.get(url)
    return Image.open(BytesIO(response.content))

def load_model(model: str = MODEL, device: str = DEVICE, threshold: float = THRESHOLD) -> Tagger:
    tagger: Tagger
    if model == "pixai":
        from models.pixai import Model
        tagger = Model(device=device, threshold_general=threshold, threshold_character=0.75)
    elif model == "taggerine":
        from models.taggerine import Model
        tagger = Model()
    elif model == "smillingwolf" or model == "smillingwolf-vit":
        from models.smillingwolf import Model
        tagger = Model(
            repo_id="SmilingWolf/wd-eva02-large-tagger-v3" if model == "smillingwolf" else "SmilingWolf/wd-vit-large-tagger-v3",
        )
    else:
        raise RuntimeError(f"Unknown model: {model}")

    tagger.load()
    return tagger


def _tag_post(post_id: int) -> None:
    global _processed, _failed
    log.info("Tagging post #%d", post_id)
    try:
        post = get_post(BASE_URL, _headers, post_id)
        url = f"{BASE_URL.rstrip('/')}/{post['thumbnailUrl']}"
        image = _download_image(url)

        results = _tagger.tag(image)
        tags = _resolve_implications([tag.name for tag in results.tags])
        update_post_tags(BASE_URL, _headers, post_id, tags)
        _processed += 1
        log.info("Post #%d: %d tags applied", post_id, len(tags))
    except PostNotFoundError:
        raise
    except SystemExit as e:
        # szurubooru.py calls sys.exit() on API errors — catch so the server keeps running
        _failed += 1
        log.error("Post #%d: szurubooru API error (exit code %s)", post_id, e.code)
    except Exception:
        _failed += 1
        log.exception("Post #%d: unexpected error", post_id)


async def _worker() -> None:
    loop = asyncio.get_running_loop()
    while True:
        post_id = await _queue.get()
        log.info("Dequeued post #%d  (%d remaining)", post_id, _queue.qsize())
        not_found = False
        try:
            await loop.run_in_executor(None, _tag_post, post_id)
        except PostNotFoundError:
            log.warning(f"Post #%d not found, retrying in {RETRY_NOT_FOUND}s", post_id)
            not_found = True
        except asyncio.CancelledError:
            _queue.task_done()
            raise
        finally:
            _queue.task_done()

        if not_found:
            await asyncio.sleep(RETRY_NOT_FOUND)
            await _queue.put(post_id)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _tagger, _headers
    if not USER or not TOKEN:
        raise RuntimeError("SZURU_USER and SZURU_TOKEN environment variables are required")

    _headers = _make_headers()
    loop = asyncio.get_running_loop()

    _tagger = await loop.run_in_executor(
        None,
        load_model,
    )

    log.info("Tagger ready. Starting queue worker.")
    task = asyncio.create_task(_worker())
    yield
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass


app = FastAPI(lifespan=lifespan)

@app.post("/webhook")
async def webhook(request: Request) -> Response:
    try:
        body = await request.json()
    except Exception:
        return Response(status_code=400)

    if body.get("type") != "post" or body.get("operation") != "created":
        return Response(status_code=200)

    post_id = body.get("id")
    if not post_id:
        log.warning("Webhook: no post ID in payload: %s", body)
        return Response(status_code=422)

    await _queue.put(int(post_id))
    log.info("Queued post #%d  (queue size: %d)", post_id, _queue.qsize())
    return Response(status_code=200)


@app.get("/health")
async def health() -> dict:
    return {
        "status": "ok",
        "queue_size": _queue.qsize(),
        "processed": _processed,
        "failed": _failed,
    }


if __name__ == "__main__":
    uvicorn.run(app, host=HOST, port=PORT)
