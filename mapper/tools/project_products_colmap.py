import json
import sys
from pathlib import Path

import numpy as np

BACKEND = Path("/home/ec2-user/SageMaker/product_detection1/backend")
sys.path.insert(0, str(BACKEND))

from colmap_text import load_text_model
from geometry import detect_shelf_planes, project_detections_to_shelves


REPO = Path("/home/ec2-user/SageMaker/product_detection1")
TEXT_DIR = REPO / "backend/workspace/iphone16-2/text"
JOB_DIR = REPO / "backend/workspace/iphone16-2_named_high_recall_v4_job"
OUT = REPO / "backend/workspace/iphone16-2_named_high_recall_v4_job/colmap_product_projection.json"


def fix_path(path_value):
    if not path_value:
        return None
    path = Path(path_value)
    if path.exists():
        return str(path)
    marker = "product_detection1/"
    text = str(path_value)
    if marker in text:
        candidate = REPO / text.split(marker, 1)[1]
        if candidate.exists():
            return str(candidate)
    return str(path_value)


def load_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}


def load_detections() -> list[dict]:
    detections = []
    for path in sorted((JOB_DIR / "detections").glob("*.json")):
        payload = load_json(path)
        for det in payload.get("detections", []):
            item = dict(det)
            name_payload = load_json(JOB_DIR / "product_names" / f"{item['id']}.json")
            product_name = (
                name_payload.get("product_name")
                or name_payload.get("catalog_candidate")
                or name_payload.get("candidate")
            )
            if product_name:
                item["label"] = product_name
                item["display_label"] = product_name
                item["recognized_product_name"] = product_name
                item["product_identity_label"] = True
                item["product_identity_confidence"] = float(
                    name_payload.get("catalog_confidence") or name_payload.get("confidence") or 0.0
                )
                item["product_identity_reason"] = name_payload.get("reason")
                item["product_identity_source"] = name_payload.get("source")
            item["crop_path"] = fix_path(
                item.get("crop_path") or name_payload.get("selected_crop_path") or item.get("raw_crop_path")
            )
            item["raw_crop_path"] = fix_path(item.get("raw_crop_path"))
            detections.append(item)
    return detections


def main():
    reconstruction = load_text_model(TEXT_DIR)
    points = np.array([pt.xyz for pt in reconstruction.points3d.values()], dtype=float)
    hint_up = np.median(
        np.array([img.rotation.T[:, 1] for img in reconstruction.images.values()], dtype=float),
        axis=0,
    )
    shelves = detect_shelf_planes(points, hint_up=hint_up)
    detections = load_detections()
    products = project_detections_to_shelves(reconstruction, shelves, detections)
    payload = {
        "text_dir": str(TEXT_DIR),
        "job_dir": str(JOB_DIR),
        "camera_count": len(reconstruction.cameras),
        "image_count": len(reconstruction.images),
        "point_count": len(reconstruction.points3d),
        "shelf_count": len(shelves),
        "detection_count": len(detections),
        "projected_product_count": len(products),
        "shelves": shelves,
        "products": products,
    }
    OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({k: payload[k] for k in [
        "camera_count",
        "image_count",
        "point_count",
        "shelf_count",
        "detection_count",
        "projected_product_count",
    ]}, indent=2))
    print(OUT)


if __name__ == "__main__":
    main()
