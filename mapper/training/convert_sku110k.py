"""
Convert SKU-110K CSV annotations to YOLO format and upload to GCS.

Usage:
    python convert_sku110k.py \
        --data_dir /path/to/SKU110K_fixed \
        --output_dir ./sku110k \
        --bucket <YOUR_BUCKET>          # omit to skip GCS upload

SKU-110K CSV columns (no header):
    image_name, x1, y1, x2, y2, class, image_width, image_height

YOLO label format (one file per image, one detection per line):
    <class_index> <cx> <cy> <w> <h>    (all values normalized 0-1)
All products are mapped to class 0.
"""
from __future__ import annotations

import argparse
import csv
import os
from collections import defaultdict
from pathlib import Path


def convert_split(
    csv_path: Path,
    images_src: Path,
    images_dst: Path,
    labels_dst: Path,
) -> int:
    images_dst.mkdir(parents=True, exist_ok=True)
    labels_dst.mkdir(parents=True, exist_ok=True)

    rows_by_image: dict[str, list[list[str]]] = defaultdict(list)
    with csv_path.open(newline="", encoding="utf-8") as fh:
        for row in csv.reader(fh):
            if len(row) < 8:
                continue
            rows_by_image[row[0]].append(row)

    converted = 0
    for image_name, rows in rows_by_image.items():
        src = images_src / image_name
        if not src.exists():
            print(f"  WARNING: image not found — {src}")
            continue

        img_w = float(rows[0][6])
        img_h = float(rows[0][7])
        if img_w <= 0 or img_h <= 0:
            continue

        lines: list[str] = []
        for row in rows:
            x1, y1, x2, y2 = float(row[1]), float(row[2]), float(row[3]), float(row[4])
            cx = ((x1 + x2) / 2) / img_w
            cy = ((y1 + y2) / 2) / img_h
            w  = (x2 - x1) / img_w
            h  = (y2 - y1) / img_h
            cx = max(0.0, min(1.0, cx))
            cy = max(0.0, min(1.0, cy))
            w  = max(0.0, min(1.0, w))
            h  = max(0.0, min(1.0, h))
            lines.append(f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

        label_file = labels_dst / (Path(image_name).stem + ".txt")
        label_file.write_text("\n".join(lines) + "\n", encoding="utf-8")

        dst = images_dst / image_name
        if not dst.exists():
            import shutil
            shutil.copy2(src, dst)

        converted += 1

    return converted


def upload_to_gcs(local_dir: Path, bucket: str, prefix: str = "sku110k") -> None:
    try:
        from google.cloud import storage
    except ImportError:
        print("google-cloud-storage not installed — skipping GCS upload.")
        return

    client = storage.Client()
    bucket_obj = client.bucket(bucket)
    total = 0
    for path in sorted(local_dir.rglob("*")):
        if path.is_dir():
            continue
        blob_name = f"{prefix}/{path.relative_to(local_dir).as_posix()}"
        bucket_obj.blob(blob_name).upload_from_filename(str(path))
        total += 1
        if total % 500 == 0:
            print(f"  uploaded {total} files...")
    print(f"  done — {total} files uploaded to gs://{bucket}/{prefix}/")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True, help="Path to SKU110K_fixed root")
    parser.add_argument("--output_dir", default="./sku110k")
    parser.add_argument("--bucket", default="", help="GCS bucket name (skips upload if empty)")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    images_root = data_dir / "images"

    splits = {
        "train": data_dir / "annotations" / "annotations_train.csv",
        "val":   data_dir / "annotations" / "annotations_val.csv",
    }
    for split, csv_path in splits.items():
        if not csv_path.exists():
            print(f"Skipping {split} — CSV not found at {csv_path}")
            continue
        print(f"Converting {split}...")
        n = convert_split(
            csv_path=csv_path,
            images_src=images_root,
            images_dst=output_dir / "images" / split,
            labels_dst=output_dir / "labels" / split,
        )
        print(f"  {n} images converted")

    if args.bucket:
        print(f"Uploading to gs://{args.bucket}/sku110k ...")
        upload_to_gcs(output_dir, args.bucket)


if __name__ == "__main__":
    main()
