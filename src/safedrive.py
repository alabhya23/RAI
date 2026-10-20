from __future__ import annotations

import json
import random
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd
from PIL import ImageDraw


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LABELS = PROJECT_ROOT / "bdd100k_labels_release" / "bdd100k" / "labels" / "bdd100k_labels_images_val.json"
DEFAULT_IMAGES = PROJECT_ROOT / "bdd100k" / "bdd100k" / "images" / "100k"


def load_validation_records(labels_path: Path) -> list[dict[str, Any]]:
    """Read BDD100K image labels into compact records for the dashboard."""
    try:
        import ijson
    except ImportError:
        with labels_path.open("r", encoding="utf-8") as source:
            raw_records = json.load(source)
        return _normalize_records(raw_records)
    else:
        with labels_path.open("rb") as source:
            return _normalize_records(ijson.items(source, "item"))


def _normalize_records(raw_records) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for raw in raw_records:
        attributes = raw.get("attributes") or {}
        people = []
        for label in raw.get("labels") or []:
            if label.get("category") != "person":
                continue
            box = label.get("box2d") or {}
            try:
                coordinates = [float(box[key]) for key in ("x1", "y1", "x2", "y2")]
            except (KeyError, TypeError, ValueError):
                continue
            if coordinates[2] <= coordinates[0] or coordinates[3] <= coordinates[1]:
                continue
            people.append(
                {
                    "box": coordinates,
                    "occluded": bool((label.get("attributes") or {}).get("occluded", False)),
                    "truncated": bool((label.get("attributes") or {}).get("truncated", False)),
                }
            )
        records.append(
            {
                "name": raw.get("name", ""),
                "timeofday": attributes.get("timeofday", "undefined"),
                "weather": attributes.get("weather", "undefined"),
                "scene": attributes.get("scene", "undefined"),
                "people": people,
            }
        )
    return records


def summarize_attributes(records: list[dict[str, Any]]) -> pd.DataFrame:
    rows = [
        {"timeofday": row["timeofday"], "weather": row["weather"], "scene": row["scene"]}
        for row in records
    ]
    return pd.DataFrame(rows)


def match_image(images_root: Path, image_name: str) -> Path | None:
    """Find an image by basename across the extracted BDD100K split folders."""
    direct_candidates = [
        images_root / "val" / image_name,
        images_root / image_name,
    ]
    for candidate in direct_candidates:
        if candidate.is_file():
            return candidate
    for candidate in images_root.glob(f"*/{image_name}"):
        if candidate.is_file():
            return candidate
    return None


def draw_boxes(image, boxes: list[dict[str, Any]], color=(38, 102, 83), label="person"):
    draw = ImageDraw.Draw(image)
    for item in boxes:
        x1, y1, x2, y2 = item["box"]
        draw.rectangle((x1, y1, x2, y2), outline=color, width=3)
        confidence = item.get("confidence")
        text = f"{label} {confidence:.2f}" if confidence is not None else label
        draw.text((x1, max(0, y1 - 15)), text, fill=color)
    return image


def intersection_over_union(first: list[float], second: list[float]) -> float:
    left = max(first[0], second[0])
    top = max(first[1], second[1])
    right = min(first[2], second[2])
    bottom = min(first[3], second[3])
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    first_area = max(0.0, first[2] - first[0]) * max(0.0, first[3] - first[1])
    second_area = max(0.0, second[2] - second[0]) * max(0.0, second[3] - second[1])
    union = first_area + second_area - intersection
    return intersection / union if union else 0.0


def detection_metrics(
    ground_truth: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
    iou_threshold: float = 0.5,
) -> dict[str, float | int]:
    matched_truth: set[int] = set()
    true_positives = 0
    false_positives = 0
    for prediction in sorted(predictions, key=lambda item: item.get("confidence", 0.0), reverse=True):
        candidates = [
            (intersection_over_union(prediction["box"], target["box"]), index)
            for index, target in enumerate(ground_truth)
            if index not in matched_truth
        ]
        best_iou, best_index = max(candidates, default=(0.0, -1))
        if best_iou >= iou_threshold:
            matched_truth.add(best_index)
            true_positives += 1
        else:
            false_positives += 1
    false_negatives = len(ground_truth) - true_positives
    precision = true_positives / (true_positives + false_positives) if true_positives + false_positives else 0.0
    recall = true_positives / (true_positives + false_negatives) if true_positives + false_negatives else 0.0
    return {
        "TP": true_positives,
        "FP": false_positives,
        "FN": false_negatives,
        "Precision": precision,
        "Recall": recall,
        "FNR": false_negatives / (true_positives + false_negatives) if true_positives + false_negatives else 0.0,
    }


def evaluate_records(
    records: list[dict[str, Any]],
    images_root: Path,
    detector: Any,
    confidence: float = 0.25,
    limit: int = 40,
    iou_threshold: float = 0.5,
) -> pd.DataFrame:
    """Evaluate a deterministic sample balanced across daytime and night."""
    eligible = [row for row in records if row["timeofday"] in {"daytime", "night"}]
    selected: list[dict[str, Any]] = []
    per_group = max(1, limit // 2)
    sampler = random.Random(42)
    for group in ("daytime", "night"):
        group_rows = [row for row in eligible if row["timeofday"] == group]
        selected.extend(sampler.sample(group_rows, min(per_group, len(group_rows))))
    if len(selected) < limit:
        selected_ids = {row["name"] for row in selected}
        selected.extend([row for row in eligible if row["name"] not in selected_ids][: limit - len(selected)])

    evaluated: list[dict[str, Any]] = []
    for row in selected:
        image_path = match_image(images_root, row["name"])
        if image_path is None:
            continue
        result = detector.predict(str(image_path), conf=confidence, verbose=False)[0]
        predictions = [
            {"box": box.xyxy[0].cpu().tolist(), "confidence": float(box.conf[0])}
            for box in result.boxes
            if int(box.cls[0]) == 0
        ]
        evaluated.append({**row, "predictions": predictions})

    output = []
    for group in ("daytime", "night"):
        group_rows = [row for row in evaluated if row["timeofday"] == group]
        aggregate = Counter()
        for row in group_rows:
            scores = detection_metrics(row["people"], row["predictions"], iou_threshold)
            aggregate.update({key: int(scores[key]) for key in ("TP", "FP", "FN")})
        tp, fp, fn = aggregate["TP"], aggregate["FP"], aggregate["FN"]
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        output.append(
            {
                "Condition": group,
                "Frames evaluated": len(group_rows),
                "Ground-truth persons": tp + fn,
                "TP": tp,
                "FP": fp,
                "FN": fn,
                "Precision": precision,
                "Recall": recall,
                "FNR": fn / (tp + fn) if tp + fn else 0.0,
            }
        )
    return pd.DataFrame(output)


def apply_center_occlusion(image, fraction: float):
    result = image.copy()
    width, height = result.size
    occlusion_width = int(width * fraction)
    occlusion_height = int(height * fraction)
    left = (width - occlusion_width) // 2
    top = (height - occlusion_height) // 2
    ImageDraw.Draw(result).rectangle(
        (left, top, left + occlusion_width, top + occlusion_height), fill=(24, 37, 34)
    )
    return result