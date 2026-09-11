"""
Submit a Vertex AI Custom Container Training job to fine-tune YOLO11l
on the SKU-110K dataset.

Prerequisites:
  pip install google-cloud-aiplatform
  gcloud auth application-default login

Fill in the four CONFIG values below before running.
"""
from __future__ import annotations

from google.cloud import aiplatform

# ── Configuration ─────────────────────────────────────────────────────────────
GCP_PROJECT   = "<YOUR_PROJECT>"    # e.g. "my-gcp-project"
GCP_REGION    = "<REGION>"          # e.g. "us-central1"
GCS_BUCKET    = "<YOUR_BUCKET>"     # e.g. "my-training-bucket" (no gs:// prefix)
IMAGE_URI     = (
    f"{GCP_REGION}-docker.pkg.dev/{GCP_PROJECT}/yolo/grocery-trainer:latest"
)
# ──────────────────────────────────────────────────────────────────────────────

MACHINE_TYPE       = "n1-standard-8"
ACCELERATOR_TYPE   = "NVIDIA_TESLA_T4"
ACCELERATOR_COUNT  = 1

# Hyperparameters passed as env vars so you can override without rebuilding the image.
ENV_VARS = {
    "DATA_YAML":  f"/gcs/{GCS_BUCKET}/sku110k/sku110k.yaml",
    "BASE_MODEL": "yolo11l.pt",
    "EPOCHS":     "80",
    "IMG_SIZE":   "1024",
    "BATCH":      "8",
}


def main() -> None:
    aiplatform.init(project=GCP_PROJECT, location=GCP_REGION)

    job = aiplatform.CustomContainerTrainingJob(
        display_name="yolo11l-grocery",
        container_uri=IMAGE_URI,
        # Pass hyperparameters without rebuilding the Docker image
        environment_variables=ENV_VARS,
    )

    print(f"Submitting training job …")
    print(f"  Image : {IMAGE_URI}")
    print(f"  Output: gs://{GCS_BUCKET}/model/")

    model = job.run(
        model_display_name="yolo11l-grocery",
        base_output_dir=f"gs://{GCS_BUCKET}/model",
        machine_type=MACHINE_TYPE,
        accelerator_type=ACCELERATOR_TYPE,
        accelerator_count=ACCELERATOR_COUNT,
        replica_count=1,
        enable_web_access=False,
        # Mount the GCS bucket so the container can read dataset files
        # via /gcs/<bucket> without downloading them first.
        boot_disk_type="pd-ssd",
        boot_disk_size_gb=100,
        sync=True,   # change to False to fire-and-forget
    )

    print("\nJob complete.")
    if model:
        print(f"Registered model: {model.resource_name}")
    print(
        f"\nDownload trained weights:\n"
        f"  gsutil cp gs://{GCS_BUCKET}/model/grocery/weights/best.pt backend/\n"
        f"\nThen set in your environment:\n"
        f"  YOLO_MODEL=best.pt"
    )


if __name__ == "__main__":
    main()
