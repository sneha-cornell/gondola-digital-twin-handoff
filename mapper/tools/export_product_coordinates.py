import csv
import json
from pathlib import Path


SOURCE = Path(
    "/home/ec2-user/SageMaker/product_detection1/backend/workspace/"
    "iphone16-2_named_high_recall_v4_job/blender_colmap_exact_placements_v2.json"
)
OUT_JSON = Path(
    "/home/ec2-user/SageMaker/product_detection1/backend/workspace/"
    "iphone16-2_named_high_recall_v4_job/product_coordinates_v2.json"
)
OUT_CSV = Path(
    "/home/ec2-user/SageMaker/product_detection1/backend/workspace/"
    "iphone16-2_named_high_recall_v4_job/product_coordinates_v2.csv"
)


def main():
    payload = json.loads(SOURCE.read_text(encoding="utf-8"))
    rows = []
    for index, placement in enumerate(payload["placements"], start=1):
        x, y, z = placement["location"]
        rows.append(
            {
                "index": index,
                "object": placement["object"],
                "product_name": placement.get("product_name") or "",
                "shelf_object": placement["shelf_object"],
                "x": x,
                "y": y,
                "z": z,
                "cluster_view_count": placement["cluster_view_count"],
                "placement_source": placement["placement_source"],
                "detection_id": placement["detection_id"],
                "image_name": placement["image_name"],
                "crop_file": placement["crop_file"],
            }
        )

    OUT_JSON.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    with OUT_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()) if rows else [])
        writer.writeheader()
        writer.writerows(rows)

    print(f"count={len(rows)}")
    print(OUT_JSON)
    print(OUT_CSV)


if __name__ == "__main__":
    main()
