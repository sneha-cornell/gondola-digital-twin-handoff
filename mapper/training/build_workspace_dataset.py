from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
BACKEND_DIR = ROOT_DIR / "backend"
DEFAULT_WORKSPACE_DIR = BACKEND_DIR / "workspace"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}

if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from detector import ObjectDetector  # noqa: E402
from product_classes import ClassRegistry  # noqa: E402


@dataclass(frozen=True)
class DatasetItem:
    job_id: str
    image_path: Path
    image_rel: Path
    label_path: Path | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a YOLO dataset from backend/workspace job images."
    )
    parser.add_argument(
        "--workspace-dir",
        default=str(DEFAULT_WORKSPACE_DIR),
        help="Workspace root containing job folders with images/ subdirectories.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Output dataset root. Will contain images/, labels/, and dataset.yaml.",
    )
    parser.add_argument(
        "--job",
        action="append",
        dest="jobs",
        help="Specific job ID(s) to include. Omit to include every workspace job.",
    )
    parser.add_argument(
        "--source-label-dir",
        default="labels",
        help="Optional per-job label directory name to copy from if manual YOLO labels already exist.",
    )
    parser.add_argument(
        "--pseudo-label",
        action="store_true",
        help="Bootstrap missing labels with the current detector.",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.10,
        help="Detector confidence threshold for pseudo-label generation.",
    )
    parser.add_argument(
        "--val-fraction",
        type=float,
        default=0.20,
        help="Fraction of images routed to val split using a stable hash split.",
    )
    parser.add_argument(
        "--copy-mode",
        choices=("copy", "symlink"),
        default="symlink",
        help="How to place images into the dataset root.",
    )
    parser.add_argument(
        "--include-empty-labels",
        action="store_true",
        help="Write empty label files for images with no objects. Useful for hard negatives.",
    )
    parser.add_argument(
        "--class-labels",
        action="store_true",
        help=(
            "Write per-SKU class ids from the identified detections in each job's "
            "detections/ + recognition_results.json instead of a single generic class. "
            "Unidentified detections are dropped from the labels."
        ),
    )
    parser.add_argument(
        "--min-identity-confidence",
        type=float,
        default=0.65,
        help="Minimum identity confidence for a detection to contribute a class label (with --class-labels).",
    )
    return parser.parse_args()


def collect_items(
    workspace_dir: Path,
    source_label_dir_name: str,
    selected_jobs: set[str] | None,
) -> list[DatasetItem]:
    items: list[DatasetItem] = []
    for job_dir in sorted(path for path in workspace_dir.iterdir() if path.is_dir()):
        if selected_jobs is not None and job_dir.name not in selected_jobs:
            continue

        images_dir = job_dir / "images"
        if not images_dir.exists():
            continue

        source_label_dir = job_dir / source_label_dir_name
        for image_path in sorted(path for path in images_dir.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES):
            label_path = source_label_dir / f"{image_path.stem}.txt"
            items.append(
                DatasetItem(
                    job_id=job_dir.name,
                    image_path=image_path,
                    image_rel=Path(job_dir.name) / image_path.name,
                    label_path=label_path if label_path.exists() else None,
                )
            )
    return items


def stable_split(item: DatasetItem, val_fraction: float) -> str:
    digest = hashlib.sha1(str(item.image_rel).encode("utf-8")).hexdigest()
    bucket = int(digest[:8], 16) / 0xFFFFFFFF
    return "val" if bucket < val_fraction else "train"


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def link_or_copy(src: Path, dst: Path, mode: str) -> None:
    ensure_parent(dst)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if mode == "symlink":
        dst.symlink_to(src.resolve())
    else:
        shutil.copy2(src, dst)


def image_size(image_path: Path) -> tuple[int, int]:
    from PIL import Image

    with Image.open(image_path) as image:
        return image.size


def detector_boxes(detector: ObjectDetector, image_path: Path) -> list[list[float]]:
    model = detector._get_model()
    predict_kwargs: dict = {
        "source": str(image_path),
        "conf": detector.confidence,
        "verbose": False,
    }
    if detector.class_ids is not None:
        predict_kwargs["classes"] = detector.class_ids
    results = model.predict(**predict_kwargs)
    return [box.xyxy[0].tolist() for box in results[0].boxes]


def write_yolo_labels(
    label_path: Path,
    image_path: Path,
    boxes_xyxy: list[list[float]],
    class_ids: list[int] | None = None,
) -> int:
    ensure_parent(label_path)
    width, height = image_size(image_path)
    lines: list[str] = []
    for index, (x1, y1, x2, y2) in enumerate(boxes_xyxy):
        class_id = class_ids[index] if class_ids is not None else 0
        cx = ((x1 + x2) * 0.5) / width
        cy = ((y1 + y2) * 0.5) / height
        bw = max(0.0, x2 - x1) / width
        bh = max(0.0, y2 - y1) / height
        cx = max(0.0, min(1.0, cx))
        cy = max(0.0, min(1.0, cy))
        bw = max(0.0, min(1.0, bw))
        bh = max(0.0, min(1.0, bh))
        lines.append(f"{class_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
    label_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return len(lines)


def load_class_boxes(
    job_dir: Path,
    registry: ClassRegistry,
    min_identity_confidence: float,
) -> dict[str, list[tuple[int, list[float]]]]:
    """Join a job's cached detections (bboxes) with its recognition results
    (class identities) and return per-image lists of (class_id, bbox_xyxy).
    Detections without a confident catalog identity are dropped."""
    detections_by_id: dict[str, dict] = {}
    detections_dir = job_dir / "detections"
    if detections_dir.exists():
        for path in sorted(detections_dir.glob("*.json")):
            payload = json.loads(path.read_text(encoding="utf-8"))
            items = payload if isinstance(payload, list) else payload.get("detections") or []
            for det in items:
                if det.get("id"):
                    detections_by_id[det["id"]] = det

    results_path = job_dir / "recognition_results.json"
    if not results_path.exists():
        return {}
    products = json.loads(results_path.read_text(encoding="utf-8")).get("products", [])

    boxes_by_image: dict[str, list[tuple[int, list[float]]]] = {}
    for product in products:
        if not product.get("exact_match"):
            continue
        confidence = float(product.get("identity_confidence") or 0.0)
        if confidence < min_identity_confidence:
            continue
        name = product.get("class_name") or product.get("recognized_product_name")
        resolved = registry.resolve(name)
        if resolved is None:
            continue
        detection = detections_by_id.get(product.get("id"))
        if detection is None or not detection.get("bbox"):
            continue
        class_id, _ = resolved
        boxes_by_image.setdefault(detection["image_name"], []).append(
            (class_id, [float(v) for v in detection["bbox"]])
        )
    return boxes_by_image


def copy_manual_label(src: Path, dst: Path) -> int:
    ensure_parent(dst)
    shutil.copy2(src, dst)
    count = 0
    for line in src.read_text(encoding="utf-8").splitlines():
        if line.strip():
            count += 1
    return count


def write_dataset_yaml(output_dir: Path, names: dict[int, str] | None = None) -> Path:
    names = names or {0: "product"}
    yaml_path = output_dir / "dataset.yaml"
    name_lines = [f"  {class_id}: {name}" for class_id, name in sorted(names.items())]
    yaml_path.write_text(
        "\n".join(
            [
                f"path: {output_dir.resolve()}",
                "train: images/train",
                "val: images/val",
                "",
                f"nc: {max(names) + 1}",
                "names:",
                *name_lines,
                "",
            ]
        ),
        encoding="utf-8",
    )
    return yaml_path


def main() -> None:
    args = parse_args()
    workspace_dir = Path(args.workspace_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    selected_jobs = set(args.jobs) if args.jobs else None
    items = collect_items(workspace_dir, args.source_label_dir, selected_jobs)
    if not items:
        raise SystemExit("No workspace images found for the requested jobs.")

    detector: ObjectDetector | None = None
    if args.pseudo_label:
        detector = ObjectDetector(confidence=args.confidence)
        # Scene-level OCR is unnecessary for bootstrapping detection labels.
        detector._ocr_crop_enabled = False

    registry: ClassRegistry | None = None
    class_boxes_by_job: dict[str, dict[str, list[tuple[int, list[float]]]]] = {}
    if args.class_labels:
        registry = ClassRegistry()
        for job_id in sorted({item.job_id for item in items}):
            class_boxes_by_job[job_id] = load_class_boxes(
                workspace_dir / job_id,
                registry,
                args.min_identity_confidence,
            )

    pseudo_cache_dir = output_dir / "_pseudo_labels"
    pseudo_cache_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "images": 0,
        "train_images": 0,
        "val_images": 0,
        "manual_labeled_images": 0,
        "class_labeled_images": 0,
        "pseudo_labeled_images": 0,
        "empty_images": 0,
        "objects": 0,
    }

    for item in items:
        split = stable_split(item, args.val_fraction)
        image_dst = output_dir / "images" / split / item.image_rel
        label_dst = output_dir / "labels" / split / item.image_rel.with_suffix(".txt")
        link_or_copy(item.image_path, image_dst, args.copy_mode)

        class_boxes = class_boxes_by_job.get(item.job_id, {}).get(item.image_path.name)

        label_count = 0
        if item.label_path is not None:
            label_count = copy_manual_label(item.label_path, label_dst)
            summary["manual_labeled_images"] += 1
        elif class_boxes:
            label_count = write_yolo_labels(
                label_dst,
                item.image_path,
                [bbox for _, bbox in class_boxes],
                class_ids=[class_id for class_id, _ in class_boxes],
            )
            summary["class_labeled_images"] += 1
        elif detector is not None:
            cache_path = pseudo_cache_dir / item.image_rel.with_suffix(".json")
            ensure_parent(cache_path)
            if cache_path.exists():
                payload = json.loads(cache_path.read_text(encoding="utf-8"))
                boxes_xyxy = payload.get("boxes_xyxy", [])
            else:
                boxes_xyxy = detector_boxes(detector, item.image_path)
                cache_path.write_text(
                    json.dumps(
                        {
                            "job_id": item.job_id,
                            "image_name": item.image_path.name,
                            "confidence": detector.confidence,
                            "boxes_xyxy": boxes_xyxy,
                        },
                        indent=2,
                    ),
                    encoding="utf-8",
                )
            if boxes_xyxy or args.include_empty_labels:
                label_count = write_yolo_labels(label_dst, item.image_path, boxes_xyxy)
            summary["pseudo_labeled_images"] += 1
        elif args.include_empty_labels:
            write_yolo_labels(label_dst, item.image_path, [])

        summary["images"] += 1
        summary["objects"] += label_count
        summary[f"{split}_images"] += 1
        if label_count == 0:
            summary["empty_images"] += 1

    yaml_path = write_dataset_yaml(
        output_dir,
        names=registry.yolo_names() if registry is not None else None,
    )

    print(f"Dataset root: {output_dir}")
    print(f"Dataset YAML: {yaml_path}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
