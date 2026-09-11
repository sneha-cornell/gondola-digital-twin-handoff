from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fill unknown single-best layout slots from nearby strict COLMAP placements."
    )
    parser.add_argument("--single-best-json", required=True)
    parser.add_argument("--strict-json", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument(
        "--max-distance",
        type=float,
        default=0.08,
        help="Maximum same-shelf horizontal distance, in Blender units/meters.",
    )
    parser.add_argument(
        "--min-strict-view-count",
        type=int,
        default=2,
        help="Require this many supporting views for strict placements used to fill slots.",
    )
    return parser.parse_args()


def load_payload(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or "placements" not in payload:
        raise SystemExit(f"Expected placement payload with 'placements': {path}")
    return payload


def shelf_axis(shelf_name: str) -> int:
    return 1 if shelf_name.startswith(("Unit2", "Unit4")) else 0


def is_fill_target(item: dict[str, Any]) -> bool:
    status = str(item.get("product_identity_status") or "")
    return status in {"unknown", "inferred_unknown_slot"}


def strict_candidate(item: dict[str, Any], min_view_count: int) -> bool:
    if item.get("product_identity_status") != "recognized":
        return False
    if not str(item.get("product_name") or "").strip() or item.get("product_name") == "Unknown":
        return False
    return int(item.get("cluster_view_count") or 0) >= min_view_count


def merge(single_best: dict[str, Any], strict: dict[str, Any], max_distance: float, min_view_count: int) -> dict[str, Any]:
    output = deepcopy(single_best)
    placements = output["placements"]

    strict_by_shelf: dict[str, list[tuple[int, dict[str, Any]]]] = defaultdict(list)
    for index, item in enumerate(strict.get("placements") or []):
        if strict_candidate(item, min_view_count):
            strict_by_shelf[str(item.get("shelf_object") or "")].append((index, item))

    used_strict: set[tuple[str, int]] = set()
    filled = []
    skipped_no_candidate = 0
    skipped_far = 0

    for item in placements:
        if not is_fill_target(item):
            continue
        shelf = str(item.get("shelf_object") or "")
        axis = shelf_axis(shelf)
        loc = item.get("location") or [0.0, 0.0, 0.0]
        coord = float(loc[axis])
        candidates = []
        for strict_index, strict_item in strict_by_shelf.get(shelf, []):
            key = (shelf, strict_index)
            if key in used_strict:
                continue
            strict_loc = strict_item.get("location") or [0.0, 0.0, 0.0]
            distance = abs(float(strict_loc[axis]) - coord)
            candidates.append((distance, strict_index, strict_item))
        candidates.sort(key=lambda row: (row[0], -int(row[2].get("cluster_view_count") or 0)))
        if not candidates:
            skipped_no_candidate += 1
            continue
        distance, strict_index, strict_item = candidates[0]
        if distance > max_distance:
            skipped_far += 1
            continue

        original = {
            "product_name": item.get("product_name"),
            "product_identity_status": item.get("product_identity_status"),
            "detection_id": item.get("detection_id"),
            "crop_file": item.get("crop_file"),
            "placement_source": item.get("placement_source"),
        }
        item["original_layout_identity"] = original
        item["product_name"] = strict_item.get("product_name")
        item["product_identity_status"] = "recognized"
        item["detection_id"] = strict_item.get("detection_id")
        item["image_name"] = strict_item.get("image_name")
        item["crop_file"] = strict_item.get("crop_file")
        item["merged_from_strict_detection_id"] = strict_item.get("detection_id")
        item["merged_from_strict_image_name"] = strict_item.get("image_name")
        item["merged_from_strict_cluster_view_count"] = strict_item.get("cluster_view_count")
        item["merged_match_distance"] = round(float(distance), 6)
        item["placement_source"] = "single_best_layout_filled_from_strict_colmap"
        used_strict.add((shelf, strict_index))
        filled.append(item)

    statuses = Counter(str(item.get("product_identity_status") or "") for item in placements)
    sources = Counter(str(item.get("placement_source") or "") for item in placements)
    output["merge_from_strict"] = {
        "strict_json": "provided",
        "max_distance": max_distance,
        "min_strict_view_count": min_view_count,
        "filled_unknown_slots": len(filled),
        "remaining_unknown_slots": statuses.get("unknown", 0) + statuses.get("inferred_unknown_slot", 0),
        "skipped_no_candidate": skipped_no_candidate,
        "skipped_far": skipped_far,
        "status_counts": dict(statuses),
        "placement_source_counts": dict(sources),
    }
    output["placed_products"] = sum(
        1
        for item in placements
        if item.get("product_identity_status") in {"recognized", "unknown", "inferred_unknown_slot"}
    )
    output["recognized_products"] = statuses.get("recognized", 0)
    output["unknown_products"] = statuses.get("unknown", 0)
    output["inferred_unknown_slots"] = statuses.get("inferred_unknown_slot", 0)
    output["empty_slots"] = statuses.get("empty_slot", 0)
    return output


def main() -> int:
    args = parse_args()
    single_best_path = Path(args.single_best_json)
    strict_path = Path(args.strict_json)
    output_path = Path(args.output_json)

    merged = merge(
        load_payload(single_best_path),
        load_payload(strict_path),
        max_distance=args.max_distance,
        min_view_count=args.min_strict_view_count,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(merged, indent=2), encoding="utf-8")
    stats = merged["merge_from_strict"]
    print(f"filled_unknown_slots={stats['filled_unknown_slots']}")
    print(f"remaining_unknown_slots={stats['remaining_unknown_slots']}")
    print(f"recognized_products={merged['recognized_products']}")
    print(f"empty_slots={merged['empty_slots']}")
    print(f"wrote={output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
