from __future__ import annotations

from pathlib import Path
from typing import Any
import importlib.util
import json
import os
import tempfile

from huggingface_hub import snapshot_download
from PIL import Image

from models.tagger import Tagger, TagResult, Tag


class Model(Tagger):
    """
    Taggerine wrapper with the same shape as the SmilingWolf wrapper:

        tagger = Model()
        tagger.load()
        result = tagger.tag(image)

    TagResult shape:

        TagResult(
            tags=[{"name": "tag_name", "category": "optional_category", "probability": 0.99}],
            ratings={}
        )
    """

    repo_id: str
    cache_dir: str

    checkpoint_filename: str
    vocab_filename: str
    device: str
    max_size: int
    threshold: float | None
    topk: int

    model_dir: Path | None
    checkpoint_path: Path | None
    vocab_path: Path | None
    inference_path: Path | None

    tagger: Any
    tag_categories: dict[str, Any]

    def __init__(
        self,
        repo_id: str = "lodestones/taggerine",
        cache_dir: str = "models/.cache",
        checkpoint_filename: str = "tagger_proto.safetensors",
        vocab_filename: str = "tagger_vocab_with_categories_and_alias_updated.json",
        device: str = "cuda",
        max_size: int = 1024,
        threshold: float | None = 0.98,
        topk: int = 50,
    ):
        self.repo_id = repo_id
        self.cache_dir = cache_dir
        self.checkpoint_filename = checkpoint_filename
        self.vocab_filename = vocab_filename
        self.device = device
        self.max_size = max_size
        self.threshold = threshold
        self.topk = topk

        self.model_dir = None
        self.checkpoint_path = None
        self.vocab_path = None
        self.inference_path = None

        self.tagger = None
        self.tag_categories = {}

    def load(self) -> None:
        print(f"Loading Taggerine model from {self.repo_id}")

        downloaded_dir = snapshot_download(
            repo_id=self.repo_id,
            local_dir=f"{self.cache_dir}/{self.repo_id}",
            allow_patterns=[
                self.checkpoint_filename,
                self.vocab_filename,
                "inference_tagger_standalone.py",
                "requirements.txt",
            ],
        )

        self.model_dir = Path(downloaded_dir)

        self.checkpoint_path = self.model_dir / self.checkpoint_filename
        self.vocab_path = self.model_dir / self.vocab_filename
        self.inference_path = self.model_dir / "inference_tagger_standalone.py"

        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {self.checkpoint_path}")

        if not self.vocab_path.exists():
            raise FileNotFoundError(f"Vocab not found: {self.vocab_path}")

        if not self.inference_path.exists():
            raise FileNotFoundError(f"Inference script not found: {self.inference_path}")

        self.tag_categories = self._load_tag_categories(self.vocab_path)

        Tagger = self._load_tagger_class(self.inference_path)

        self.tagger = Tagger(
            checkpoint_path=str(self.checkpoint_path),
            vocab_path=str(self.vocab_path),
            device=self.device,
            max_size=self.max_size,
        )

        print(
            "Taggerine ready "
            f"(device={self.device}, max_size={self.max_size}, "
            f"threshold={self.threshold}, topk={self.topk})"
        )

    def tag(
        self,
        image: Image.Image,
        threshold: float | None = None,
        topk: int | None = None,
    ) -> TagResult:
        """
        Tag a PIL image.

        Taggerine's standalone predictor expects a path or URL,
        so this writes the PIL image to a temporary PNG.
        """
        if self.tagger is None:
            raise RuntimeError("Model is not loaded. Call load() before tag().")

        temp_path = self._write_temp_image(image)

        try:
            return self.tag_path(temp_path, threshold=threshold, topk=topk)
        finally:
            try:
                os.unlink(temp_path)
            except FileNotFoundError:
                pass

    def tag_url(
        self,
        url: str,
        threshold: float | None = None,
        topk: int | None = None,
    ) -> TagResult:
        if self.tagger is None:
            raise RuntimeError("Model is not loaded. Call load() before tag_url().")

        return self._predict(url, threshold=threshold, topk=topk)

    def tag_path(
        self,
        path: str | Path,
        threshold: float | None = None,
        topk: int | None = None,
    ) -> TagResult:
        if self.tagger is None:
            raise RuntimeError("Model is not loaded. Call load() before tag_path().")

        return self._predict(str(path), threshold=threshold, topk=topk)

    def _predict(
        self,
        source: str,
        threshold: float | None = None,
        topk: int | None = None,
    ) -> TagResult:
        resolved_threshold = self.threshold if threshold is None else threshold
        resolved_topk = self.topk if topk is None else topk

        predict_topk, predict_threshold = (
            (None, resolved_threshold)
            if resolved_threshold is not None
            else (resolved_topk, None)
        )

        raw_results = self.tagger.predict(
            source,
            topk=predict_topk,
            threshold=predict_threshold,
        )

        tags: list[Tag] = []
        ratings: dict[str, float] = {}

        for item in raw_results:
            tag_name, probability = self._parse_prediction(item)

            normalized_name = self.normalize_tag(tag_name)
            category = self._get_category(tag_name)

            if self._is_rating_category(category):
                ratings[normalized_name] = probability
                continue

            tags.append(Tag(name=normalized_name, category=str(category), probability=probability))
        return TagResult(
            tags=tags,
            ratings=ratings,
        )

    def _parse_prediction(self, item: Any) -> tuple[str, float]:
        """
        Supports:

            ("tag name", 0.991)

        Also tolerates dict-like result shapes.
        """
        if isinstance(item, dict):
            tag_name = item.get("tag") or item.get("name") or item.get("label")
            probability = (
                item.get("score")
                or item.get("probability")
                or item.get("confidence")
            )

            if tag_name is None or probability is None:
                raise ValueError(f"Unrecognized prediction dict shape: {item}")

            return str(tag_name), float(probability)

        if isinstance(item, (list, tuple)) and len(item) >= 2:
            return str(item[0]), float(item[1])

        raise ValueError(f"Unrecognized prediction shape: {item}")

    def _load_tagger_class(self, inference_path: Path) -> Any:
        spec = importlib.util.spec_from_file_location(
            "taggerine_inference_tagger_standalone",
            inference_path,
        )

        if spec is None or spec.loader is None:
            raise ImportError(f"Could not load inference module from {inference_path}")

        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        if not hasattr(module, "Tagger"):
            raise ImportError(f"No Tagger class found in {inference_path}")

        return module.Tagger

    def _load_tag_categories(self, vocab_path: Path) -> dict[str, Any]:
        with open(vocab_path, "r", encoding="utf-8") as f:
            vocab = json.load(f)

        tag_categories = (
            vocab.get("tag2category")
            or vocab.get("tag_to_category")
            or vocab.get("categories")
            or {}
        )

        if not isinstance(tag_categories, dict):
            return {}

        return tag_categories

    def _get_category(self, tag_name: str) -> str | None:
        normalized = self.normalize_tag(tag_name)
        spaced = tag_name.replace("_", " ")

        category = (
            self.tag_categories.get(tag_name)
            or self.tag_categories.get(normalized)
            or self.tag_categories.get(spaced)
        )

        if category is None:
            return None

        if isinstance(category, list):
            return ",".join(str(value) for value in category)

        return str(category)

    def _is_rating_category(self, category: str | None) -> bool:
        if category is None:
            return False

        return category.lower() in {
            "rating",
            "ratings",
            "safety",
            "9",
        }

    def _write_temp_image(self, image: Image.Image) -> str:
        image = self._prepare_pil_image(image)

        with tempfile.NamedTemporaryFile(
            mode="wb",
            suffix=".png",
            delete=False,
        ) as f:
            image.save(f, format="PNG")
            return f.name
