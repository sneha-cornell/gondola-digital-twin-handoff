"""
Vertex AI training entry point for YOLO11l fine-tuned on SKU-110K.

Vertex AI automatically sets AIP_MODEL_DIR to a GCS path like
gs://<bucket>/model/  — ultralytics writes best.pt there so you can
download it after the job completes.
"""
from __future__ import annotations

import os

from ultralytics import YOLO

DATA_YAML   = os.getenv("DATA_YAML",   "/gcs/<YOUR_BUCKET>/sku110k/sku110k.yaml")
BASE_MODEL  = os.getenv("BASE_MODEL",  "yolo11l.pt")
EPOCHS      = int(os.getenv("EPOCHS",  "80"))
IMG_SIZE    = int(os.getenv("IMG_SIZE", "1024"))
BATCH       = int(os.getenv("BATCH",   "8"))

# AIP_MODEL_DIR is injected by Vertex AI and points to a GCS path.
# Fall back to ./runs for local testing.
OUTPUT_DIR  = os.getenv("AIP_MODEL_DIR", "./runs")

model = YOLO(BASE_MODEL)

model.train(
    data=DATA_YAML,
    epochs=EPOCHS,
    imgsz=IMG_SIZE,
    batch=BATCH,
    device=0,
    project=OUTPUT_DIR,
    name="grocery",
    exist_ok=True,
    # Good defaults for dense retail shelf detection
    mosaic=1.0,
    mixup=0.1,
    degrees=5.0,
    translate=0.1,
    scale=0.5,
    fliplr=0.5,
    hsv_h=0.015,
    hsv_s=0.7,
    hsv_v=0.4,
    patience=15,       # early stopping
    save_period=10,    # checkpoint every 10 epochs
    plots=True,
)
