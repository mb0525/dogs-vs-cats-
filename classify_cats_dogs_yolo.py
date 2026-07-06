"""Classify a folder of cat/dog images with a pretrained YOLO model."""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

from ultralytics import YOLO


CAT_CLASSES = {
    "tabby",
    "tiger cat",
    "Persian cat",
    "Siamese cat",
    "Egyptian cat",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("cat-or-dog-image-classification-challenge/data_mixed"),
        help="Folder containing JPG images.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("cat_dog_predictions.csv"),
        help="Destination CSV file.",
    )
    parser.add_argument("--model", default="yolo11n-cls.pt")
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--limit", type=int, help="Only process the first N images.")
    return parser.parse_args()


def numeric_sort_key(path: Path) -> tuple[int, int | str]:
    try:
        return (0, int(path.stem))
    except ValueError:
        return (1, path.name.lower())


def class_ids(names: dict[int, str]) -> tuple[list[int], list[int]]:
    normalized_names = {
        index: name.replace("_", " ") for index, name in names.items()
    }
    cat_ids = [
        index for index, name in normalized_names.items() if name in CAT_CLASSES
    ]

    by_name = {name: index for index, name in normalized_names.items()}
    try:
        dog_start = by_name["Chihuahua"]
        dog_end = by_name["Mexican hairless"]
    except KeyError as error:
        raise RuntimeError("The model does not use the expected ImageNet labels.") from error
    dog_ids = list(range(dog_start, dog_end + 1))

    if len(cat_ids) != 5 or not dog_ids:
        raise RuntimeError("Could not identify the ImageNet cat and dog classes.")
    return cat_ids, dog_ids


def main() -> None:
    args = parse_args()
    images = sorted(args.input.glob("*.jpg"), key=numeric_sort_key)
    if args.limit is not None:
        images = images[: args.limit]
    if not images:
        raise SystemExit(f"No JPG images found in: {args.input.resolve()}")

    model = YOLO(args.model)
    cat_ids, dog_ids = class_ids(model.names)
    rows: list[dict[str, str | int | float]] = []
    started = time.perf_counter()

    for batch_start in range(0, len(images), args.batch):
        batch_paths = images[batch_start : batch_start + args.batch]
        predictions = model.predict(
            source=[str(path) for path in batch_paths],
            imgsz=224,
            device="cpu",
            verbose=False,
        )

        if len(predictions) != len(batch_paths):
            raise RuntimeError(
                f"YOLO returned {len(predictions)} results for "
                f"{len(batch_paths)} input images."
            )

        for image_path, result in zip(batch_paths, predictions, strict=True):
            probabilities = result.probs.data.cpu()
            cat_score = float(probabilities[cat_ids].sum())
            dog_score = float(probabilities[dog_ids].sum())
            animal_score = cat_score + dog_score
            dog_probability = dog_score / animal_score if animal_score else 0.5
            label = "dog" if dog_probability >= 0.5 else "cat"
            confidence = max(dog_probability, 1.0 - dog_probability)
            rows.append(
                {
                    "image": image_path.name,
                    "img": int(image_path.stem) if image_path.stem.isdigit() else image_path.stem,
                    "label": label,
                    "is_dog": round(dog_probability, 6),
                    "confidence": round(confidence, 6),
                    "cat_score": round(cat_score, 6),
                    "dog_score": round(dog_score, 6),
                }
            )

        count = len(rows)
        if count % 100 < args.batch or count == len(images):
            elapsed = time.perf_counter() - started
            print(
                f"Processed {count}/{len(images)} images "
                f"({count / elapsed:.1f} images/s)",
                flush=True,
            )

    rows.sort(key=lambda row: (0, int(row["img"])) if isinstance(row["img"], int) else (1, str(row["img"])))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8-sig") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    cats = sum(row["label"] == "cat" for row in rows)
    dogs = len(rows) - cats
    print(f"Done: {cats} cats, {dogs} dogs")
    print(f"CSV: {args.output.resolve()}")


if __name__ == "__main__":
    main()
