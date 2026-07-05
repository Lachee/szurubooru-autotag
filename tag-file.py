#!/usr/bin/env python3

import argparse
import csv
from pathlib import Path

from PIL import Image

from models.taggerine import Model as Taggerine
from models.smillingwolf import Model as SmillingWolf
from models.pixai import Model as PixAI


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--path")
    parser.add_argument("--threshold", type=float, default=0.7)
    parser.add_argument("--csv", dest="csv_path", type=Path)
    parser.add_argument(
        "--model",
        choices=("taggerine", "smillingwolf", "smillingwolf-vit", "pixai"),
        default="pixai",
    )
    return parser.parse_args()


def print_table(tags: list[dict], ratings: dict[str, float]) -> None:
    tags = sorted(tags, key=lambda tag: tag["probability"], reverse=True)

    name_width = max(len("Name"), *(len(tag["name"]) for tag in tags)) if tags else len("Name")
    category_values = [tag.get("category") or "-" for tag in tags]
    category_width = max(len("Category"), *(len(category) for category in category_values)) if tags else len("Category")
    probability_width = len("Probability")

    print(f"\r\n{'Name':<{name_width}}  {'Category':<{category_width}}  {'Probability':>{probability_width}}")
    print(f"{'-' * name_width}  {'-' * category_width}  {'-' * probability_width}")
    for tag, category in zip(tags, category_values):
        print(
            f"{tag['name']:<{name_width}}  "
            f"{category:<{category_width}}  "
            f"{(tag['probability']*100):>{probability_width}.4f} %"
        )

    if ratings:
        rating = max(ratings, key=ratings.get)
        print(f"\r\nRating: {rating}  ({(ratings[rating]*100):.2f} % sure)")


def write_csv(output_path: Path, tags: list[dict], ratings: dict[str, float]) -> None:
    rows: list[dict[str, str]] = []

    for tag in sorted(tags, key=lambda tag: tag["probability"], reverse=True):
        rows.append(
            {
                "kind": "tag",
                "name": tag["name"],
                "category": tag.get("category") or "-",
                "probability": f"{tag['probability']:.6f}",
            }
        )

    for name, probability in sorted(ratings.items(), key=lambda item: item[1], reverse=True):
        rows.append(
            {
                "kind": "rating",
                "name": name,
                "category": "-",
                "probability": f"{probability:.6f}",
            }
        )

    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["kind", "name", "category", "probability"])
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    if args.model == "smillingwolf":
        tagger = SmillingWolf(repo_id="SmilingWolf/wd-eva02-large-tagger-v3")
    elif args.model == "smillingwolf-vit":
        tagger = SmillingWolf(repo_id="SmilingWolf/wd-vit-large-tagger-v3")
    elif args.model == "pixai":
        tagger = PixAI(
            device="cuda",
            threshold_general=0.30,
            threshold_character=0.75,
            topk=128,
        )
    else:
        tagger = Taggerine()
    tagger.load()

    image = Image.open(args.path)
    result = tagger.tag(image, threshold=args.threshold)

    if args.csv_path:
        write_csv(args.csv_path, result.tags, result.ratings)
        print(f"Wrote CSV to {args.csv_path}")
        return

    print_table(result.tags, result.ratings)


if __name__ == "__main__":
    main()
