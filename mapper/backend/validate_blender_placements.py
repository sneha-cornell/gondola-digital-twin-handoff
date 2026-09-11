from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a review report for image-derived Blender product placements."
    )
    parser.add_argument("placements_json", help="Placement JSON generated for Blender.")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for placement_review.{md,csv,json}. Defaults to the placement JSON directory.",
    )
    parser.add_argument(
        "--near-duplicate-distance",
        type=float,
        default=0.08,
        help="Shelf-axis distance below which same-name neighbors are flagged.",
    )
    parser.add_argument(
        "--tight-spacing-distance",
        type=float,
        default=0.045,
        help="Shelf-axis distance below which any neighboring placements are flagged.",
    )
    return parser.parse_args()


def load_placements(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        return payload, list(payload.get("placements") or [])
    return {}, list(payload)


def shelf_unit(shelf_name: str) -> str:
    if "_Shelf_" in shelf_name:
        return shelf_name.split("_Shelf_", 1)[0]
    return shelf_name.split("_", 1)[0]


def shelf_axis(shelf_name: str) -> str:
    unit = shelf_unit(shelf_name)
    if unit in {"Unit2", "Unit4"}:
        return "y"
    return "x"


def location(item: dict[str, Any]) -> tuple[float, float, float]:
    raw = item.get("location")
    if isinstance(raw, list | tuple) and len(raw) >= 3:
        return float(raw[0]), float(raw[1]), float(raw[2])
    return (
        float(item.get("x") or 0.0),
        float(item.get("y") or 0.0),
        float(item.get("z") or 0.0),
    )


def product_name(item: dict[str, Any]) -> str:
    name = str(item.get("product_name") or "").strip()
    return name or "Unknown"


def is_unknown(item: dict[str, Any]) -> bool:
    status = str(item.get("product_identity_status") or "").lower()
    return status in {"unknown", "inferred_unknown_slot"} or product_name(item).lower() == "unknown"


def median(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(statistics.median(values))


def robust_outlier(value: float, values: list[float], min_count: int = 8) -> bool:
    if len(values) < min_count:
        return False
    q1, q3 = statistics.quantiles(values, n=4)[0], statistics.quantiles(values, n=4)[2]
    iqr = q3 - q1
    if iqr <= 1e-9:
        return abs(value - median(values)) > 0.08
    return value < q1 - 3.0 * iqr or value > q3 + 3.0 * iqr


def base_record(index: int, item: dict[str, Any]) -> dict[str, Any]:
    x, y, z = location(item)
    return {
        "index": index,
        "detection_id": item.get("detection_id") or "",
        "image_name": item.get("image_name") or "",
        "product_name": product_name(item),
        "shelf_object": item.get("shelf_object") or "",
        "x": x,
        "y": y,
        "z": z,
        "cluster_view_count": int(item.get("cluster_view_count") or 0),
        "placement_source": item.get("placement_source") or "",
        "crop_file": item.get("crop_file") or "",
    }


def build_report(
    payload: dict[str, Any],
    placements: list[dict[str, Any]],
    near_duplicate_distance: float,
    tight_spacing_distance: float,
) -> dict[str, Any]:
    method = str(payload.get("method") or "")
    is_single_best_layout = method.startswith("single best image")
    records = [base_record(i, item) for i, item in enumerate(placements)]
    flags: list[set[str]] = [set() for _ in records]
    notes: list[list[str]] = [[] for _ in records]

    for i, item in enumerate(placements):
        status = str(item.get("product_identity_status") or "").lower()
        if is_unknown(item):
            flags[i].add("unknown_product")
        if status == "inferred_unknown_slot":
            flags[i].add("inferred_unknown_slot")
        if status == "empty_slot":
            flags[i].add("empty_slot")
        if not is_single_best_layout and records[i]["cluster_view_count"] <= 1:
            flags[i].add("single_view")
        if not is_single_best_layout and records[i]["placement_source"] != "ray":
            flags[i].add("fallback_source")

    by_shelf: dict[str, list[int]] = defaultdict(list)
    for i, rec in enumerate(records):
        by_shelf[str(rec["shelf_object"])].append(i)

    for shelf, indexes in by_shelf.items():
        axis = shelf_axis(shelf)
        coord = axis
        ordered = sorted(indexes, key=lambda idx: float(records[idx][coord]))
        for left, right in zip(ordered, ordered[1:]):
            distance = abs(float(records[right][coord]) - float(records[left][coord]))
            same_name = records[left]["product_name"] == records[right]["product_name"]
            if distance < tight_spacing_distance:
                flags[left].add("tight_neighbor")
                flags[right].add("tight_neighbor")
                detail = f"{records[right]['detection_id']} at {distance:.3f}m"
                notes[left].append(f"tight neighbor {detail}")
                detail = f"{records[left]['detection_id']} at {distance:.3f}m"
                notes[right].append(f"tight neighbor {detail}")
            if same_name and distance < near_duplicate_distance:
                flags[left].add("near_duplicate_same_name")
                flags[right].add("near_duplicate_same_name")

        xs = [float(records[idx]["x"]) for idx in indexes]
        ys = [float(records[idx]["y"]) for idx in indexes]
        zs = [float(records[idx]["z"]) for idx in indexes]
        z_med = median(zs)
        for idx in indexes:
            if robust_outlier(float(records[idx]["x"]), xs) or robust_outlier(float(records[idx]["y"]), ys):
                flags[idx].add("shelf_xy_outlier")
            if abs(float(records[idx]["z"]) - z_med) > 0.08:
                flags[idx].add("shelf_z_outlier")

    weights = {
        "unknown_product": 4,
        "fallback_source": 3,
        "single_view": 2,
        "near_duplicate_same_name": 2,
        "tight_neighbor": 1,
        "shelf_xy_outlier": 2,
        "shelf_z_outlier": 2,
        "inferred_unknown_slot": 3,
        "empty_slot": 1,
    }

    risky: list[dict[str, Any]] = []
    spacing_only = 0
    primary_review_flags = {
        "unknown_product",
        "fallback_source",
        "single_view",
        "shelf_xy_outlier",
        "shelf_z_outlier",
        "inferred_unknown_slot",
        "empty_slot",
    }
    for i, rec in enumerate(records):
        rec_flags = sorted(flags[i])
        risk_score = sum(weights.get(flag, 1) for flag in rec_flags)
        needs_primary_review = bool(set(rec_flags) & primary_review_flags)
        if rec_flags and not needs_primary_review:
            spacing_only += 1
        if rec_flags and needs_primary_review:
            row = dict(rec)
            row["risk_score"] = risk_score
            row["flags"] = rec_flags
            row["notes"] = notes[i]
            risky.append(row)

    risky.sort(key=lambda row: (-int(row["risk_score"]), str(row["shelf_object"]), int(row["index"])))

    shelf_summary = []
    for shelf, indexes in sorted(by_shelf.items()):
        shelf_flags = Counter(flag for idx in indexes for flag in flags[idx])
        names = Counter(records[idx]["product_name"] for idx in indexes)
        shelf_summary.append(
            {
                "shelf_object": shelf,
                "count": len(indexes),
                "risky_count": sum(1 for idx in indexes if flags[idx] & primary_review_flags),
                "spacing_only_count": sum(
                    1 for idx in indexes if flags[idx] and not flags[idx] & primary_review_flags
                ),
                "flags": dict(shelf_flags),
                "top_products": names.most_common(8),
            }
        )
    shelf_summary.sort(key=lambda row: (-int(row["risky_count"]), -int(row["count"]), str(row["shelf_object"])))

    flag_counts = Counter(flag for rec_flags in flags for flag in rec_flags)
    product_counts = Counter(rec["product_name"] for rec in records)
    source_counts = Counter(rec["placement_source"] for rec in records)
    view_counts = Counter(rec["cluster_view_count"] for rec in records)

    return {
        "source_metadata": {
            key: payload.get(key)
            for key in (
                "method",
                "detections_with_crops_or_boxes",
                "unknown_detections",
                "shelf_hits",
                "direct_ray_hits",
                "observed_point_fallback_hits",
                "inferred_missing_shelf_hits",
                "clustered_products",
            )
            if key in payload
        },
        "summary": {
            "placements": len(records),
            "risky_placements": len(risky),
            "spacing_only_placements": spacing_only,
            "recognized": sum(1 for item in placements if not is_unknown(item)),
            "unknown": sum(1 for item in placements if is_unknown(item)),
            "inferred_unknown_slot": flag_counts.get("inferred_unknown_slot", 0),
            "empty_slot": flag_counts.get("empty_slot", 0),
            "single_view": flag_counts.get("single_view", 0),
            "fallback_source": flag_counts.get("fallback_source", 0),
            "near_duplicate_same_name": flag_counts.get("near_duplicate_same_name", 0),
            "tight_neighbor": flag_counts.get("tight_neighbor", 0),
            "shelf_xy_outlier": flag_counts.get("shelf_xy_outlier", 0),
            "shelf_z_outlier": flag_counts.get("shelf_z_outlier", 0),
        },
        "flag_counts": dict(flag_counts),
        "placement_sources": dict(source_counts),
        "cluster_view_counts": dict(sorted(view_counts.items())),
        "top_products": product_counts.most_common(25),
        "shelves": shelf_summary,
        "risky_placements": risky,
    }


def write_csv(path: Path, risky: list[dict[str, Any]]) -> None:
    fieldnames = [
        "risk_score",
        "flags",
        "index",
        "detection_id",
        "image_name",
        "product_name",
        "shelf_object",
        "x",
        "y",
        "z",
        "cluster_view_count",
        "placement_source",
        "crop_file",
        "notes",
    ]
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in risky:
            out = {key: row.get(key, "") for key in fieldnames}
            out["flags"] = ";".join(row.get("flags") or [])
            out["notes"] = " | ".join(row.get("notes") or [])
            writer.writerow(out)


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    summary = report["summary"]
    lines = [
        "# Blender Placement Review",
        "",
        "## Summary",
        "",
        f"- Placements: {summary['placements']}",
        f"- Primary review placements: {summary['risky_placements']}",
        f"- Spacing-only placements: {summary['spacing_only_placements']}",
        f"- Recognized: {summary['recognized']}",
        f"- Unknown: {summary['unknown']}",
        f"- Inferred unknown slots: {summary['inferred_unknown_slot']}",
        f"- Empty slots: {summary['empty_slot']}",
        f"- Single-view placements: {summary['single_view']}",
        f"- Fallback-source placements: {summary['fallback_source']}",
        f"- Near-duplicate same-name placements: {summary['near_duplicate_same_name']}",
        f"- Tight-neighbor placements: {summary['tight_neighbor']}",
        f"- Shelf XY outliers: {summary['shelf_xy_outlier']}",
        f"- Shelf Z outliers: {summary['shelf_z_outlier']}",
        "",
        "## Highest Risk Placements",
        "",
        "| Risk | Flags | Detection | Product | Shelf | Views | Source |",
        "| ---: | --- | --- | --- | --- | ---: | --- |",
    ]
    for row in report["risky_placements"][:100]:
        flags = ", ".join(row["flags"])
        product = str(row["product_name"]).replace("|", "\\|")
        lines.append(
            f"| {row['risk_score']} | {flags} | {row['detection_id']} | "
            f"{product} | {row['shelf_object']} | {row['cluster_view_count']} | {row['placement_source']} |"
        )

    lines.extend(
        [
            "",
            "## Shelves To Review First",
            "",
            "| Shelf | Placements | Primary Review | Spacing Only | Main Flags | Top Products |",
            "| --- | ---: | ---: | ---: | --- | --- |",
        ]
    )
    for shelf in report["shelves"][:30]:
        flag_text = ", ".join(f"{name}:{count}" for name, count in Counter(shelf["flags"]).most_common(5))
        top_text = ", ".join(f"{name} ({count})" for name, count in shelf["top_products"][:5])
        safe_top_text = top_text.replace("|", "\\|")
        lines.append(
            f"| {shelf['shelf_object']} | {shelf['count']} | {shelf['risky_count']} | "
            f"{shelf['spacing_only_count']} | {flag_text} | {safe_top_text} |"
        )

    lines.extend(
        [
            "",
            "## Top Products",
            "",
            "| Product | Count |",
            "| --- | ---: |",
        ]
    )
    for name, count in report["top_products"]:
        lines.append(f"| {str(name).replace('|', '\\|')} | {count} |")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    placements_path = Path(args.placements_json)
    output_dir = Path(args.output_dir) if args.output_dir else placements_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    payload, placements = load_placements(placements_path)
    report = build_report(
        payload,
        placements,
        near_duplicate_distance=args.near_duplicate_distance,
        tight_spacing_distance=args.tight_spacing_distance,
    )

    json_path = output_dir / "placement_review.json"
    csv_path = output_dir / "placement_review_risky.csv"
    md_path = output_dir / "placement_review.md"

    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_csv(csv_path, report["risky_placements"])
    write_markdown(md_path, report)

    summary = report["summary"]
    print(f"placements={summary['placements']}")
    print(f"risky_placements={summary['risky_placements']}")
    print(f"single_view={summary['single_view']}")
    print(f"fallback_source={summary['fallback_source']}")
    print(f"unknown={summary['unknown']}")
    print(f"wrote={md_path}")
    print(f"wrote={csv_path}")
    print(f"wrote={json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
