from __future__ import annotations

import argparse
from pathlib import Path

import torch
from ultralytics import YOLO


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = ROOT_DIR / "backend" / "models" / "best.pt"
# When best.pt is missing, default to the YOLO11-Large COCO backbone. It is the
# best recall / speed tradeoff to fine-tune for dense shelf scenes. Switch to
# yolo11x.pt for maximum recall, or yolo11m.pt for faster training/inference.
FALLBACK_BACKBONE = "yolo11l.pt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a custom shelf-product YOLO detector.")
    parser.add_argument("--data", required=True, help="Path to YOLO dataset.yaml.")
    parser.add_argument(
        "--model",
        default=str(DEFAULT_MODEL if DEFAULT_MODEL.exists() else FALLBACK_BACKBONE),
        help="Base checkpoint to fine-tune from.",
    )
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--project", default=str(ROOT_DIR / "training" / "runs"))
    parser.add_argument("--name", default="custom_shelf")
    parser.add_argument("--device", default="auto", help="auto, cpu, mps, or CUDA device id.")
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def resolve_device(value: str) -> str | int:
    if value != "auto":
        try:
            return int(value)
        except ValueError:
            return value

    if torch.cuda.is_available():
        return 0
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)

    model = YOLO(args.model)
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        device=device,
        project=args.project,
        name=args.name,
        exist_ok=True,
        resume=args.resume,
        optimizer="auto",
        patience=args.patience,
        save_period=10,
        plots=True,
        # Keep augmentations moderate; shelf facings are dense and perspective-sensitive.
        mosaic=0.4,
        mixup=0.05,
        degrees=6.0,
        translate=0.08,
        scale=0.35,
        fliplr=0.5,
        hsv_h=0.015,
        hsv_s=0.5,
        hsv_v=0.3,
    )


if __name__ == "__main__":
    main()
