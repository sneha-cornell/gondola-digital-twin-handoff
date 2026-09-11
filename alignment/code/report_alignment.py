"""
Show what came out of align_products_to_gondola.py.

Prints a per-shelf occupancy table and an elevation view of each side, so a
misalignment is visible rather than buried in a residual: products should sit
in tidy rows just above their shelf lines, spread across the bay run. Products
stacked on the wrong shelf, bunched at one end, or floating between boards all
show up immediately.

    python3 report_alignment.py --aligned /tmp/demo_aligned.json

Pass --truth (from synthetic_scan.py) to also print recovery error against the
known transform.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def elevation(side_shelves: list[dict], products: list[dict], *, width: int = 78) -> list[str]:
    """One text row per shelf: product facings drawn at their position along X.

    Rows are ordered top shelf first, the way you'd see the fixture standing in
    front of it. A '#' means several facings fell in the same character cell.
    """
    if not side_shelves:
        return ["  (no shelves on this side)"]

    run = max(
        max(abs(shelf["bounds"]["u"][0]), abs(shelf["bounds"]["u"][1]))
        for shelf in side_shelves
    )
    by_shelf: dict[str, list[dict]] = defaultdict(list)
    for product in products:
        if product.get("shelf_id"):
            by_shelf[product["shelf_id"]].append(product)

    lines = []
    for shelf in sorted(side_shelves, key=lambda s: -s["height"]):
        cells = [" "] * width
        for product in by_shelf.get(shelf["id"], []):
            local_u = product["shelf_local"]["u"]
            column = int(round((local_u / run + 1.0) * 0.5 * (width - 1)))
            column = max(0, min(width - 1, column))
            cells[column] = "#" if cells[column] != " " else "o"
        count = len(by_shelf.get(shelf["id"], []))
        lines.append(f"  {shelf['height']:5.3f} m |{''.join(cells)}| {count:3d}")
    lines.append(f"          +{'-' * width}+")
    lines.append(f"          {-run:+.2f} m{' ' * (width - 16)}{run:+.2f} m")
    return lines


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--aligned", required=True, help="Output of align_products_to_gondola.py")
    parser.add_argument("--truth", default=None, help="Ground truth from synthetic_scan.py")
    parser.add_argument("--sample", type=int, default=6, help="How many product records to print")
    args = parser.parse_args()

    aligned = json.loads(Path(args.aligned).read_text(encoding="utf-8"))
    products = aligned.get("products", [])
    shelves = aligned.get("shelves", [])
    alignment = aligned.get("product_alignment", {})
    layout = aligned.get("layout", {})

    print("=" * 92)
    print(f"  ALIGNED PLANOGRAM  --  {aligned.get('job_id', '?')}")
    print("=" * 92)
    transform = alignment.get("transform", {})
    quality = alignment.get("quality", {})
    fit = alignment.get("shelf_height_fit", {})
    print(f"  fixture      {layout.get('total_bay_run_m', 0):.2f} m run x "
          f"{layout.get('gondola_depth_m', 0):.2f} m deep x "
          f"{layout.get('gondola_height_m', 0):.2f} m high"
          f"   (shelf heights: {layout.get('shelf_heights_source', '?')})")
    print(f"  scale        {transform.get('scale_meters_per_colmap_unit', 0):.6f} m per COLMAP unit"
          f"   (shelf-match RMS {fit.get('rms_m', 0) * 100:.1f} cm over "
          f"{fit.get('matched', 0)} shelves, {fit.get('scale_drift_vs_hint', 1) * 100 - 100:+.1f}% "
          f"vs the seed)")
    print(f"  seated       {quality.get('snapped_to_a_shelf', 0)}/{quality.get('product_count', 0)} "
          f"products on a board, median clearance "
          f"{(quality.get('median_clearance_m') or 0) * 100:.1f} cm")

    print()
    print("  PER-SHELF OCCUPANCY")
    print(f"  {'shelf':<22} {'side':<8} {'z (m)':>7} {'items':>6} {'clearance':>11} {'span used (m)':>16}")
    print(f"  {'-' * 22} {'-' * 8} {'-' * 7} {'-' * 6} {'-' * 11} {'-' * 16}")
    for shelf in shelves:
        seated = [p for p in products if p.get("shelf_id") == shelf["id"]]
        if not seated:
            continue
        clearances = [p["clearance_m"] for p in seated]
        us = [p["shelf_local"]["u"] for p in seated]
        print(f"  {shelf['id']:<22} {str(shelf.get('side_label')):<8} {shelf['height']:>7.3f} "
              f"{len(seated):>6} {np.mean(clearances) * 100:>8.1f} cm "
              f"{min(us):>+7.2f} .. {max(us):>+6.2f}")

    unseated = [p for p in products if not p.get("seated")]
    if unseated:
        print(f"\n  {len(unseated)} products did not land on any board "
              f"(kept, flagged `seated: false` -- these are the ones to look at):")
        for product in unseated[: args.sample]:
            position = product["position"]
            print(f"    {str(product['id']):<12} at "
                  f"({position[0]:+.2f}, {position[1]:+.2f}, {position[2]:+.2f})  "
                  f"{product.get('product_name') or '?'}")

    for side in ("Side A", "Side C", "Side B", "Side D"):
        side_shelves = [s for s in shelves if s.get("side_label") == side]
        seated_here = [
            p for p in products
            if p.get("side_label") == side and p.get("shelf_local")
        ]
        if not side_shelves or not seated_here:
            continue
        print()
        print(f"  ELEVATION -- {side}  ({len(seated_here)} facings, viewed head-on, top shelf first)")
        for line in elevation(side_shelves, seated_here):
            print(line)

    print()
    print("  SAMPLE PRODUCT RECORDS (model frame, meters)")
    for product in products[: args.sample]:
        local = product.get("shelf_local") or {}
        position = product["position"]
        print(f"    {str(product.get('product_name'))[:38]:<38} "
              f"({position[0]:+.3f}, {position[1]:+.3f}, {position[2]:+.3f})  "
              f"-> {str(product.get('shelf_id')):<20} "
              f"u={local.get('u', float('nan')):+.3f} clearance={product.get('clearance_m') or 0:.3f}")

    if args.truth:
        truth = json.loads(Path(args.truth).read_text(encoding="utf-8"))
        expected = np.array(truth["positions_model"], dtype=float)
        actual = np.array([p["position"] for p in products], dtype=float)
        if len(expected) == len(actual):
            errors = np.linalg.norm(actual - expected, axis=1)
            print()
            print("  VERSUS GROUND TRUTH")
            print(f"    scale      {transform.get('scale_meters_per_colmap_unit', 0):.6f} vs "
                  f"{truth['scale_meters_per_colmap_unit']:.6f} "
                  f"({100 * (transform.get('scale_meters_per_colmap_unit', 0) / truth['scale_meters_per_colmap_unit'] - 1):+.3f}%)")
            print(f"    position   median {np.median(errors) * 100:.2f} cm   "
                  f"p95 {np.percentile(errors, 95) * 100:.2f} cm   "
                  f"max {errors.max() * 100:.2f} cm")
            print("    (a 180 degree flip about up would show here as metre-scale error -- see "
                  "--flip)")
    print("=" * 92)


if __name__ == "__main__":
    main()
