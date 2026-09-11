"""One-command SKU-110K finetune.

Wraps the convert + train flow into a single entry point. Assumes you have
already downloaded SKU-110K_fixed (the dataset is gated and not auto-fetched).

Typical use:

    python training/finetune_sku110k.py \
        --data-dir /path/to/SKU110K_fixed \
        --output-dir training/sku110k_yolo \
        --model yolo11l.pt \
        --epochs 60

The script:
  1. Converts SKU-110K CSV annotations to YOLO format (idempotent — skipped
     when label files already exist).
  2. Writes a local dataset.yaml pointing at the converted output.
  3. Runs ``train_custom.py`` with sane defaults.
  4. Copies the resulting ``best.pt`` to ``backend/models/best.pt`` so the
     production detector picks it up automatically.

Pass ``--no-copy`` to skip the final copy if you want to vet the run first.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
TRAINING_DIR = ROOT_DIR / "training"
BACKEND_MODELS_DIR = ROOT_DIR / "backend" / "models"


def _have_converted(output_dir: Path) -> bool:
    train_labels = output_dir / "labels" / "train"
    val_labels = output_dir / "labels" / "val"
    return (
        train_labels.exists()
        and val_labels.exists()
        and any(train_labels.glob("*.txt"))
        and any(val_labels.glob("*.txt"))
    )


def _write_dataset_yaml(output_dir: Path) -> Path:
    yaml_path = output_dir / "dataset.yaml"
    yaml_path.write_text(
        "\n".join(
            [
                f"path: {output_dir.resolve()}",
                "train: images/train",
                "val:   images/val",
                "",
                "nc: 1",
                "names:",
                "  0: product",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return yaml_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Convert + finetune YOLO on SKU-110K.")
    parser.add_argument(
        "--data-dir",
        required=True,
        help="Path to the downloaded SKU110K_fixed directory "
             "(containing images/ and annotations/).",
    )
    parser.add_argument(
        "--output-dir",
        default=str(TRAINING_DIR / "sku110k_yolo"),
        help="Where to materialize the YOLO-format dataset.",
    )
    parser.add_argument(
        "--model",
        default="yolo11l.pt",
        help="Base checkpoint. yolo11l.pt is the default sweet spot; "
             "yolo11x.pt for max recall, yolo11m.pt for faster training.",
    )
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--name", default="sku110k_finetune")
    parser.add_argument(
        "--device",
        default="auto",
        help="auto, cpu, mps, or CUDA device id.",
    )
    parser.add_argument(
        "--no-copy",
        action="store_true",
        help="Do not copy the resulting best.pt into backend/models/.",
    )
    parser.add_argument(
        "--reconvert",
        action="store_true",
        help="Re-run convert_sku110k.py even if YOLO labels already exist.",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not data_dir.exists():
        print(f"ERROR: --data-dir does not exist: {data_dir}", file=sys.stderr)
        return 2

    # 1. Convert (idempotent unless --reconvert).
    if args.reconvert or not _have_converted(output_dir):
        print(f"[1/3] converting SKU-110K CSV → YOLO ({output_dir})")
        rc = subprocess.call(
            [
                sys.executable,
                str(TRAINING_DIR / "convert_sku110k.py"),
                "--data_dir",
                str(data_dir),
                "--output_dir",
                str(output_dir),
            ]
        )
        if rc != 0:
            print(f"convert_sku110k.py failed with exit code {rc}", file=sys.stderr)
            return rc
    else:
        print(f"[1/3] YOLO labels already present at {output_dir} — skipping convert.")

    # 2. Write a local dataset.yaml that points to the converted output.
    yaml_path = _write_dataset_yaml(output_dir)
    print(f"[2/3] dataset.yaml: {yaml_path}")

    # 3. Train.
    runs_dir = TRAINING_DIR / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    train_cmd = [
        sys.executable,
        str(TRAINING_DIR / "train_custom.py"),
        "--data",
        str(yaml_path),
        "--model",
        args.model,
        "--epochs",
        str(args.epochs),
        "--imgsz",
        str(args.imgsz),
        "--batch",
        str(args.batch),
        "--name",
        args.name,
        "--device",
        args.device,
        "--project",
        str(runs_dir),
    ]
    print(f"[3/3] training: {' '.join(train_cmd)}")
    rc = subprocess.call(train_cmd)
    if rc != 0:
        print(f"train_custom.py failed with exit code {rc}", file=sys.stderr)
        return rc

    # 4. Promote the trained checkpoint to backend/models/best.pt.
    if args.no_copy:
        print("--no-copy set; skipping promotion.")
        return 0

    weight_path = runs_dir / args.name / "weights" / "best.pt"
    if not weight_path.exists():
        print(
            f"WARNING: trained best.pt not found at {weight_path}; not promoting.",
            file=sys.stderr,
        )
        return 0

    BACKEND_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    target = BACKEND_MODELS_DIR / "best.pt"
    if target.exists():
        backup = BACKEND_MODELS_DIR / "best.previous.pt"
        shutil.copy2(target, backup)
        print(f"backed up existing best.pt → {backup}")
    shutil.copy2(weight_path, target)
    print(f"promoted {weight_path} → {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
