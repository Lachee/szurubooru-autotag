from __future__ import annotations

from pathlib import Path
from typing import Any
import hashlib
import json
import os
import subprocess
import sys
import tempfile

from huggingface_hub import snapshot_download
from PIL import Image

from models.tagger import Tagger, TagResult, Tag


class Model(Tagger):
    """
    PixAI Tagger v0.9 wrapper.

    Same public shape as the SmilingWolf/Taggerine wrappers:

        tagger = Model()
        tagger.load()
        result = tagger.tag(image)

    Returned TagResult shape:

        TagResult(
            tags=[
                {
                    "name": "tag_name",
                    "category": "general" | "character" | "copyright",
                    "probability": 0.99,
                }
            ],
            ratings={}
        )

    Notes:
        - pixai-labs/pixai-tagger-v0.9 is a gated Hugging Face repo.
        - Accept the repo terms in your browser first.
        - Then run `hf auth login` or set `HF_TOKEN`.
    """

    repo_id: str
    cache_dir: str
    weights_filename: str
    tags_filename: str
    character_ip_map_filename: str

    device: str
    image_size: int
    threshold_general: float
    threshold_character: float
    topk: int | None
    include_ip_tags: bool

    model_dir: Path | None
    weights_path: Path | None
    tags_path: Path | None
    character_ip_map_path: Path | None

    model: Any
    transform: Any

    tag_map: dict[str, int]
    index_to_tag_map: dict[int, str]
    character_ip_mapping: dict[str, list[str]]
    gen_tag_count: int
    character_tag_count: int

    def __init__(
        self,
        repo_id: str = "pixai-labs/pixai-tagger-v0.9",
        cache_dir: str = "models/.cache",
        weights_filename: str = "model_v0.9.pth",
        tags_filename: str = "tags_v0.9_13k.json",
        character_ip_map_filename: str = "char_ip_map.json",
        device: str = "cuda",
        image_size: int = 448,
        threshold_general: float = 0.30,
        threshold_character: float = 0.75,
        topk: int | None = 128,
        include_ip_tags: bool = False,
    ):
        self.repo_id = repo_id
        self.cache_dir = cache_dir
        self.weights_filename = weights_filename
        self.tags_filename = tags_filename
        self.character_ip_map_filename = character_ip_map_filename

        self.device = device
        self.image_size = image_size
        self.threshold_general = threshold_general
        self.threshold_character = threshold_character
        self.topk = topk
        self.include_ip_tags = include_ip_tags

        self.model_dir = None
        self.weights_path = None
        self.tags_path = None
        self.character_ip_map_path = None

        self.model = None
        self.transform = None

        self.tag_map = {}
        self.index_to_tag_map = {}
        self.character_ip_mapping = {}
        self.gen_tag_count = 0
        self.character_tag_count = 0

    def load(self) -> None:
        print(f"Loading PixAI Tagger model from {self.repo_id}")

        downloaded_dir = snapshot_download(
            repo_id=self.repo_id,
            local_dir=f"{self.cache_dir}/{self.repo_id}",
            allow_patterns=[
                self.weights_filename,
                self.tags_filename,
                self.character_ip_map_filename,
                "README.md",
            ],
            token=os.getenv("HF_TOKEN") or None,
        )

        self.model_dir = Path(downloaded_dir)

        self.weights_path = self.model_dir / self.weights_filename
        self.tags_path = self.model_dir / self.tags_filename
        self.character_ip_map_path = self.model_dir / self.character_ip_map_filename

        self._validate_downloaded_files()

        torch = self._import_torch()
        transforms = self._import_torchvision_transforms()

        self._load_tag_metadata()
        self.model = self._load_model(torch)
        self.transform = transforms.Compose(
            [
                transforms.Resize((self.image_size, self.image_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.5, 0.5, 0.5],
                    std=[0.5, 0.5, 0.5],
                ),
            ]
        )

        print(
            "PixAI Tagger ready "
            f"(device={self.device}, image_size={self.image_size}, "
            f"threshold_general={self.threshold_general}, "
            f"threshold_character={self.threshold_character}, topk={self.topk})"
        )

    def tag(
        self,
        image: Image.Image,
        threshold: float | None = None,
        topk: int | None = None,
    ) -> TagResult:
        """
        Tag a PIL image.

        The single `threshold` argument is supported for compatibility with the
        generic Tagger interface. When provided, it overrides both general and
        character thresholds.
        """
        if self.model is None or self.transform is None:
            raise RuntimeError("Model is not loaded. Call load() before tag().")

        threshold_general = self.threshold_general if threshold is None else threshold
        threshold_character = self.threshold_character if threshold is None else threshold
        resolved_topk = self.topk if topk is None else topk

        image = self._prepare_pil_image(image)

        torch = self._import_torch()

        with torch.inference_mode():
            x = self.transform(image).unsqueeze(0)

            if self.device == "cuda":
                x = x.pin_memory().to(self.device, non_blocking=True)
            else:
                x = x.to(self.device)

            probs = self.model(x)[0].detach().float().cpu()

        tags = self._build_tag_items(
            probs=probs,
            threshold_general=threshold_general,
            threshold_character=threshold_character,
            topk=resolved_topk,
        )

        return TagResult(
            tags=tags,
            ratings={},
        )

    def tag_path(
        self,
        path: str | Path,
        threshold: float | None = None,
        topk: int | None = None,
    ) -> TagResult:
        image = Image.open(path)
        image.load()

        return self.tag(
            image,
            threshold=threshold,
            topk=topk,
        )

    def tag_url(
        self,
        url: str,
        threshold: float | None = None,
        topk: int | None = None,
    ) -> TagResult:
        """
        Convenience method for URL images.

        Requires `requests`, which is installed by `_install_requirements_if_needed`.
        """
        try:
            import io
            import requests
        except ModuleNotFoundError as e:
            raise ModuleNotFoundError(
                f"{e}\n\nInstall requests with:\n"
                f"  {sys.executable} -m pip install requests"
            ) from e

        response = requests.get(
            url,
            timeout=30,
            headers={"User-Agent": "Mozilla/5.0"},
        )
        response.raise_for_status()

        image = Image.open(io.BytesIO(response.content))
        image.load()

        return self.tag(
            image,
            threshold=threshold,
            topk=topk,
        )

    def _build_tag_items(
        self,
        probs: Any,
        threshold_general: float,
        threshold_character: float,
        topk: int | None,
    ) -> list[Tag]:
        candidates: list[Tag] = []

        general_end = self.gen_tag_count
        character_start = self.gen_tag_count
        character_end = self.gen_tag_count + self.character_tag_count

        for index in range(0, general_end):
            probability = float(probs[index])

            if probability < threshold_general:
                continue

            candidates.append(Tag(name=self.normalize_tag(self.index_to_tag_map[index]), category="general", probability=probability))

        character_probabilities_by_tag: dict[str, float] = {}
        for index in range(character_start, character_end):
            probability = float(probs[index])

            if probability < threshold_character:
                continue

            tag_name = self.normalize_tag(self.index_to_tag_map[index])
            character_probabilities_by_tag[tag_name] = probability

            candidates.append(Tag(name=tag_name, category="character", probability=probability))

        if self.include_ip_tags:
            candidates.extend(
                self._build_ip_tag_items(character_probabilities_by_tag)
            )

        candidates.sort(
            key=lambda item : item.probability,
            reverse=True,
        )

        if topk is not None and topk > 0:
            candidates = candidates[:topk]

        return candidates

    def _build_ip_tag_items(
        self,
        character_probabilities_by_tag: dict[str, float],
    ) -> list[Tag]:
        """
        PixAI's local script infers IP/copyright tags from detected characters.

        These probabilities are derived from the strongest linked character, not
        direct model logits.
        """
        ip_probabilities: dict[str, float] = {}

        for character_tag, character_probability in character_probabilities_by_tag.items():
            mapped_ips = (
                self.character_ip_mapping.get(character_tag)
                or self.character_ip_mapping.get(character_tag.replace("_", " "))
                or []
            )

            for ip_tag in mapped_ips:
                normalized_ip = self.normalize_tag(ip_tag)
                previous = ip_probabilities.get(normalized_ip, 0.0)
                ip_probabilities[normalized_ip] = max(previous, character_probability)

        return [
            Tag(name=ip_tag, category="copyright", probability=probability)
            for ip_tag, probability in ip_probabilities.items()
        ]

    def _load_model(self, torch: Any) -> Any:
        timm = self._import_timm()

        class TaggingHead(torch.nn.Module):
            def __init__(self, input_dim: int, num_classes: int):
                super().__init__()
                self.head = torch.nn.Sequential(
                    torch.nn.Linear(input_dim, num_classes)
                )

            def forward(self, x: Any) -> Any:
                logits = self.head(x)
                return torch.sigmoid(logits)

        if self.weights_path is None:
            raise RuntimeError("weights_path was not initialized")

        device = self._resolve_device(torch)

        base_model_repo = "hf_hub:SmilingWolf/wd-eva02-large-tagger-v3"
        encoder = timm.create_model(base_model_repo, pretrained=False)
        encoder.reset_classifier(0)

        decoder = TaggingHead(1024, self.gen_tag_count + self.character_tag_count)
        model = torch.nn.Sequential(encoder, decoder)

        try:
            state = torch.load(
                str(self.weights_path),
                map_location=device,
                weights_only=True,
            )
        except TypeError:
            # Older torch versions do not support weights_only.
            state = torch.load(
                str(self.weights_path),
                map_location=device,
            )

        model.load_state_dict(state)
        model.to(device).eval()

        self.device = device

        return model

    def _load_tag_metadata(self) -> None:
        if self.tags_path is None:
            raise RuntimeError("tags_path was not initialized")

        with open(self.tags_path, "r", encoding="utf-8") as f:
            tag_info = json.load(f)

        self.tag_map = tag_info["tag_map"]
        tag_split = tag_info["tag_split"]

        self.gen_tag_count = int(tag_split["gen_tag_count"])
        self.character_tag_count = int(tag_split["character_tag_count"])

        self.index_to_tag_map = {
            int(index): tag
            for tag, index in self.tag_map.items()
        }

        if self.character_ip_map_path is not None and self.character_ip_map_path.exists():
            with open(self.character_ip_map_path, "r", encoding="utf-8") as f:
                raw_mapping = json.load(f)

            self.character_ip_mapping = {
                self.normalize_tag(character_tag): [
                    self.normalize_tag(ip_tag)
                    for ip_tag in ip_tags
                ]
                for character_tag, ip_tags in raw_mapping.items()
                if isinstance(ip_tags, list)
            }
        else:
            self.character_ip_mapping = {}

    def _validate_downloaded_files(self) -> None:
        if self.weights_path is None:
            raise RuntimeError("weights_path was not initialized")

        if self.tags_path is None:
            raise RuntimeError("tags_path was not initialized")

        if self.character_ip_map_path is None:
            raise RuntimeError("character_ip_map_path was not initialized")

        missing = [
            path
            for path in [
                self.weights_path,
                self.tags_path,
                self.character_ip_map_path,
            ]
            if not path.exists()
        ]

        if missing:
            missing_lines = "\n".join(f"  - {path}" for path in missing)

            raise FileNotFoundError(
                "PixAI model files were not found after download:\n"
                f"{missing_lines}\n\n"
                "If this is a gated repository, accept the model terms on "
                "Hugging Face and authenticate first:\n"
                "  hf auth login\n\n"
                "Or set:\n"
                "  export HF_TOKEN=..."
            )

    def _resolve_device(self, torch: Any) -> str:
        requested = self.device.lower().strip()

        if requested == "cuda" and not torch.cuda.is_available():
            print("CUDA requested but unavailable; falling back to CPU")
            return "cpu"

        return requested

    def _prepare_pil_image(self, image: Image.Image) -> Image.Image:
        image.load()

        if image.mode in ("RGBA", "LA") or "transparency" in image.info:
            image = image.convert("RGBA")

            background = Image.new("RGB", image.size, (255, 255, 255))
            background.paste(image, mask=image.split()[3])

            return background

        if image.mode == "P":
            image = image.convert("RGBA")

            background = Image.new("RGB", image.size, (255, 255, 255))
            background.paste(image, mask=image.split()[3])

            return background

        return image.convert("RGB")

    @staticmethod
    def _import_torch() -> Any:
        try:
            import torch

            return torch
        except ModuleNotFoundError as e:
            raise ModuleNotFoundError(
                f"{e}\n\nInstall torch with:\n"
                f"  {sys.executable} -m pip install torch torchvision"
            ) from e

    @staticmethod
    def _import_torchvision_transforms() -> Any:
        try:
            import torchvision.transforms as transforms

            return transforms
        except ModuleNotFoundError as e:
            raise ModuleNotFoundError(
                f"{e}\n\nInstall torchvision with:\n"
                f"  {sys.executable} -m pip install torchvision"
            ) from e

    @staticmethod
    def _import_timm() -> Any:
        try:
            import timm

            return timm
        except ModuleNotFoundError as e:
            raise ModuleNotFoundError(
                f"{e}\n\nInstall timm with:\n"
                f"  {sys.executable} -m pip install timm"
            ) from e

    @staticmethod
    def normalize_tag(tag: str) -> str:
        return tag.strip().replace(" ", "_")

