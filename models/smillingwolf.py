import csv

import numpy as np
from huggingface_hub import hf_hub_download
import onnxruntime as ort
from PIL import Image

from models.tagger import Tagger, TagResult, Tag


class Model(Tagger):

    repo_id: str
    cache_dir: str

    general_tags: list[tuple[int, str]]
    character_tags: list[tuple[int, str]]
    rating_tags: list[tuple[int, str]]

    session: ort.InferenceSession | None = None
    target_size = 448
    is_nchw = False

    def __init__(self,
                 repo_id: str = "SmilingWolf/wd-eva02-large-tagger-v3",
                 cache_dir: str = "models/.cache",
     ):
        self.repo_id = repo_id
        self.cache_dir = cache_dir
        self.general_tags = []
        self.character_tags = []
        self.rating_tags = []
        self.session = None
        self.target_size = 448
        self.is_nchw = False

    def load(self):
        print(f"Loading model {self.repo_id}")

        # Setup the Tags
        csv_path = hf_hub_download(
            repo_id=self.repo_id,
            local_dir=f"{self.cache_dir}/{self.repo_id}",
            filename="selected_tags.csv"
        )
        if csv_path:
            self.general_tags = []
            self.character_tags = []
            self.rating_tags = []

            with open(csv_path, "r", encoding="utf-8") as f:
                reader = csv.reader(f)
                next(reader)  # skip header row

                for i, row in enumerate(reader):
                    if not row:
                        continue

                    tag_name = row[1]
                    category = int(row[2])

                    if category == 0:
                        self.general_tags.append((i, tag_name))
                    elif category == 4:
                        self.character_tags.append((i, tag_name))
                    elif category == 9:
                        self.rating_tags.append((i, tag_name))

        # Set up ONNX runtime session
        model_path = hf_hub_download(
            repo_id=self.repo_id,
            local_dir=f"{self.cache_dir}/{self.repo_id}",
            filename="model.onnx",
        )
        if model_path:
            self.session = ort.InferenceSession(model_path)
            input_shape = self.session.get_inputs()[0].shape

            # Dynamically find the model's expected resolution and layout: NCHW vs NHWC
            if len(input_shape) >= 4 and input_shape[1] == 3:
                self.target_size = input_shape[2]
                self.is_nchw = True
            elif len(input_shape) >= 4 and input_shape[3] == 3:
                self.target_size = input_shape[1]
                self.is_nchw = False
            else:
                self.target_size = 448
                self.is_nchw = False
        else:
            self.session = None
            self.target_size = 448
            self.is_nchw = False

    def tag(self, image: Image.Image, threshold: float = 0.35) -> TagResult:
        if self.session is None:
            raise RuntimeError("Model session is not initialized. Call load() before tag().")

        input_name = self.session.get_inputs()[0].name

        image = self._prepare_pil_image(image)

        # Pad image to make it square
        width, height = image.size
        size = max(width, height)

        padded = Image.new("RGB", (size, size), (255, 255, 255))
        padded.paste(
            image,
            (
                (size - width) // 2,
                (size - height) // 2,
            ),
        )

        # Resize to expected model size
        try:
            resample_filter = Image.Resampling.LANCZOS
        except AttributeError:
            resample_filter = Image.ANTIALIAS

        resized = padded.resize(
            (self.target_size, self.target_size),
            resample_filter,
        )

        # Convert to BGR array for SmilingWolf models
        img_array = np.array(resized, dtype=np.float32)
        img_array = img_array[:, :, ::-1]  # RGB -> BGR

        if self.is_nchw:
            img_array = img_array.transpose((2, 0, 1))

        img_array = np.expand_dims(img_array, axis=0)

        # Run ONNX model prediction
        outputs = self.session.run(None, {input_name: img_array})
        scores = outputs[0][0]

        # Filter predictions above threshold
        result_tags: list[Tag] = []

        for idx, tag_name in self.general_tags:
            prob = float(scores[idx])

            if prob >= threshold:
                result_tags.append(Tag(name=tag_name, category="general", probability=prob))

        for idx, tag_name in self.character_tags:
            prob = float(scores[idx])

            if prob >= threshold:
                result_tags.append(Tag(name=tag_name, category="character", probability=prob))

        # Get safety ratings
        ratings: dict[str, float] = {}

        for idx, tag_name in self.rating_tags:
            ratings[tag_name] = float(scores[idx])

        return TagResult(
            tags=result_tags,
            ratings=ratings,
        )
