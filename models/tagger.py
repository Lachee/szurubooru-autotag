from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from PIL import Image

@dataclass()
class Tag:
    name : str
    category : str
    probability: float

@dataclass()
class TagResult:
    tags: list[Tag]
    ratings: dict[str, float]


class Tagger(ABC):
    @abstractmethod
    def load(self) -> None:
        pass

    @abstractmethod
    def tag(self, image: Image.Image, *args, **kwargs) -> TagResult:
        raise NotImplementedError

    @staticmethod
    def normalize_tag(tag: str) -> str:
        return tag.strip().replace(" ", "_")

    def _prepare_pil_image(self, image: Image.Image) -> Image.Image:
        if image.mode in ("RGBA", "LA") or "transparency" in image.info:
            image = image.convert("RGBA")

            background = Image.new("RGB", image.size, (255, 255, 255))
            background.paste(image, mask=image.split()[3])

            return background

        return image.convert("RGB")
