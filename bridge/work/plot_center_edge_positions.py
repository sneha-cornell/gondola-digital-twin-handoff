"""Center + edge shelf-position plot for real bboxes on image 00014.

Extends plot_bbox_positions_from_colmap.py (same real colmap_text.py parser,
same points3D.txt, same hand-drawn bboxes) by keeping the FULL set of observed
3D points per box instead of collapsing straight to a median. That spread is
what lets each product report a left edge / right edge on the shelf-run axis,
not just a single center point.

Per product, per shelf row:
    background rejection = drop any observed 3D point more than
        BACKGROUND_REJECT_RADIUS world units from that box's own median point,
        before computing anything else. Several boxes span a gap between bags
        where the wall/ceiling behind the shelf is visible (see S1-P1, S1-P5
        in this image's real reconstruction) -- those pixels have real
        reconstructed 3D points, just meters behind the shelf, not on it.
        Checked across every box in this dataset: every genuine product
        cluster tops out at <=0.584 units from its own median; the two
        contaminated boxes have their real cluster end by 0.485 and then jump
        straight to 3.7+ / 4.7+ units.0.6 sits cleanly in that gap for both --
        it was picked by inspecting the sorted per-box distance distributions,
        not assumed.
    center = median of the surviving points' projection onto the row's fitted
             shelf-run axis (PCA on the row's product centers, COLMAP units)
    left / right edge = 5th / 95th percentile of that same projection
        (percentile on top of the background rejection above, as a second,
        milder guard against any remaining stray point)

Everything stays in COLMAP world units (arbitrary scale, not meters) --
this image has no metric anchor, so "meters" would be a fabricated number.
"""

from __future__ import annotations

import dataclasses
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

_real_dataclass = dataclasses.dataclass


def _compat_dataclass(*args, **kwargs):
    kwargs.pop("slots", None)
    return _real_dataclass(*args, **kwargs)


dataclasses.dataclass = _compat_dataclass
MAPPER_BACKEND = "/Users/snehamanimurugan/Downloads/dtsm_extract/digital-twin-shelf-mapper-main/backend"
sys.path.insert(0, MAPPER_BACKEND)
from colmap_text import load_text_model  # noqa: E402


DATA = Path("/Users/snehamanimurugan/Downloads/data")
SOURCE = DATA / "images" / "00014.jpg"
OUTPUT_IMAGE = Path("outputs/00014_center_edge_positions.png")
OUTPUT_JSON = Path("outputs/00014_center_edge_positions.json")
MIN_OBSERVED_POINTS = 8  # same floor as the mapper's observation-based shelf matcher
EDGE_PERCENTILE = 5.0    # 5th/95th -- robust edge, not the noisy raw min/max
BACKGROUND_REJECT_RADIUS = 0.6  # world units from the box's own median point;
# see module docstring -- verified against the sorted distance distribution
# of every box in this dataset before picking this number.

ROWS = [
    ("S1", 610, 1470, [(190, 520), (490, 820), (770, 1080), (1060, 1350), (1320, 1640), (1610, 2020)]),
    ("S2", 1490, 2040, [(180, 450), (440, 710), (700, 990), (980, 1260), (1240, 1530), (1510, 1840), (1820, 2120)]),
    ("S3", 2060, 2600, [(170, 450), (430, 710), (690, 965), (945, 1225), (1200, 1490), (1470, 1760), (1740, 2070)]),
    ("S4", 2620, 3120, [(180, 470), (450, 720), (700, 980), (960, 1240), (1220, 1510), (1490, 1780), (1760, 2070)]),
    ("S5", 3140, 3635, [(170, 490), (470, 760), (740, 1040), (1020, 1320), (1300, 1610), (1590, 1900), (1880, 2130)]),
]


def make_font(size: int):
    try:
        return ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial Bold.ttf", size)
    except OSError:
        return ImageFont.load_default()


def observed_points(reconstruction, image, bbox) -> np.ndarray:
    x1, y1, x2, y2 = bbox
    return np.array(
        [
            reconstruction.points3d[point_id].xyz
            for x, y, point_id in image.observations
            if x1 <= x <= x2 and y1 <= y <= y2 and point_id in reconstruction.points3d
        ],
        dtype=float,
    )


def reject_background(points: np.ndarray) -> tuple[np.ndarray, int]:
    """Drop points far from this box's own median -- see module docstring.

    One pass is enough here (verified against the real data): the background
    points sit 3.7-6.3 units out, the floor is 0.6, there is no ambiguous
    middle ground that would need a second, re-centered pass.
    """
    if len(points) < 3:
        return points, 0
    median = np.median(points, axis=0)
    distance = np.linalg.norm(points - median, axis=1)
    keep = distance <= BACKGROUND_REJECT_RADIUS
    return points[keep], int((~keep).sum())


def fit_row_axis(row_items: list[dict]) -> np.ndarray | None:
    """Same PCA-on-row-centers axis as the original script: the direction the
    row's product centers vary along most, in the COLMAP world frame."""
    valid = [item for item in row_items if item["observed_point_count"] >= MIN_OBSERVED_POINTS]
    if len(valid) < 2:
        return None
    centers = np.array([item["_center_3d"] for item in valid], dtype=float)
    centroid = np.median(centers, axis=0)
    _, _, right_vectors = np.linalg.svd(centers - centroid, full_matrices=False)
    axis = right_vectors[0]
    pixel_centers = np.array([item["image_center_px"][0] for item in valid], dtype=float)
    projected = (centers - centroid) @ axis
    slope, _intercept = np.polyfit(pixel_centers, projected, deg=1)
    if slope < 0:
        axis = -axis
    return axis


def project_edges(points: np.ndarray, axis: np.ndarray, centroid: np.ndarray) -> tuple[float, float, float]:
    """Center/left/right of one product's own point cloud, projected onto the
    row's shared axis -- so edges come from that product's real 3D spread,
    not an assumed width."""
    projected = (points - centroid) @ axis
    center = float(np.median(projected))
    left = float(np.percentile(projected, EDGE_PERCENTILE))
    right = float(np.percentile(projected, 100 - EDGE_PERCENTILE))
    return center, left, right


def main() -> None:
    reconstruction = load_text_model(DATA)
    image_pose = reconstruction.images["00014.jpg"]

    product_rows: list[dict] = []
    for shelf_id, top, bottom, intervals in ROWS:
        row_items = []
        for number, (left, right) in enumerate(intervals, start=1):
            bbox = [left, top + 60, right, bottom - 35]
            raw_points = observed_points(reconstruction, image_pose, bbox)
            points, dropped = reject_background(raw_points)
            item = {
                "id": f"{shelf_id}-P{number}",
                "shelf_id": shelf_id,
                "bbox": bbox,
                "image_center_px": [float((left + right) * 0.5), float((bbox[1] + bbox[3]) * 0.5)],
                "raw_observed_point_count": int(len(raw_points)),
                "background_points_rejected": dropped,
                "observed_point_count": int(len(points)),
                "status": "positioned" if len(points) >= MIN_OBSERVED_POINTS else "insufficient_3d_evidence",
            }
            if len(points):
                item["_center_3d"] = np.median(points, axis=0)
                item["_points"] = points
            row_items.append(item)

        axis = fit_row_axis(row_items)
        centroid = (
            np.median(np.array([it["_center_3d"] for it in row_items if it["observed_point_count"] >= MIN_OBSERVED_POINTS]), axis=0)
            if axis is not None else None
        )
        for item in row_items:
            if axis is not None and item["observed_point_count"] >= MIN_OBSERVED_POINTS:
                center, edge_left, edge_right = project_edges(item["_points"], axis, centroid)
                item["shelf_axis_colmap"] = [float(v) for v in axis]
                item["center_u"] = center
                item["edge_left_u"] = edge_left
                item["edge_right_u"] = edge_right
                item["width_u"] = edge_right - edge_left
                item["center_position_colmap"] = [float(v) for v in item["_center_3d"]]
            else:
                item["shelf_axis_colmap"] = [float(v) for v in axis] if axis is not None else None
                item["center_u"] = None
                item["edge_left_u"] = None
                item["edge_right_u"] = None
                item["width_u"] = None
                item["center_position_colmap"] = (
                    [float(v) for v in item["_center_3d"]] if "_center_3d" in item else None
                )
            item.pop("_center_3d", None)
            item.pop("_points", None)
        product_rows.append({"shelf_id": shelf_id, "products": row_items})

    positioned = [it for row in product_rows for it in row["products"] if it["status"] == "positioned"]
    payload = {
        "image_name": "00014.jpg",
        "coordinate_system": "COLMAP world units (arbitrary scale, not meters -- this image has no metric anchor)",
        "method": (
            "Real 3D points observed inside each bbox (colmap_text.py parser, points3D.txt observations). "
            f"Background rejection first: points more than {BACKGROUND_REJECT_RADIUS} world units from the "
            "box's own median are dropped (catches wall/ceiling geometry visible through gaps between bags "
            "-- verified against every box's real distance distribution before picking the radius). "
            "Per row: PCA on the surviving product centers gives a shared shelf-run axis. Per product: its "
            "own surviving points are projected onto the axis; center = median, edges = "
            f"{EDGE_PERCENTILE:.0f}th/{100 - EDGE_PERCENTILE:.0f}th percentile."
        ),
        "minimum_observed_points": MIN_OBSERVED_POINTS,
        "edge_percentile": EDGE_PERCENTILE,
        "background_reject_radius": BACKGROUND_REJECT_RADIUS,
        "positioned_product_count": len(positioned),
        "insufficient_3d_evidence_count": sum(1 for row in product_rows for it in row["products"] if it["status"] != "positioned"),
        "shelves": product_rows,
    }
    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    # --- render: photo with boxes, then a pure position graph below it -----
    source = Image.open(SOURCE).convert("RGBA")
    chart_h = 190 + len(product_rows) * 210
    canvas = Image.new("RGBA", (source.width, source.height + chart_h), (15, 24, 33, 255))
    canvas.alpha_composite(source, (0, 0))
    draw = ImageDraw.Draw(canvas)
    title_font, label_font, small_font, tiny_font = (
        make_font(42), make_font(27), make_font(21), make_font(17)
    )

    for row in product_rows:
        items = row["products"]
        top = items[0]["bbox"][1] - 60
        bottom = items[0]["bbox"][3] + 35
        draw.rectangle((105, top, 2140, bottom), outline=(0, 230, 210, 255), width=8)
        draw.text((125, top + 10), row["shelf_id"], font=label_font, fill=(0, 255, 225, 255))
        for item in items:
            x1, y1, x2, y2 = item["bbox"]
            valid = item["status"] == "positioned"
            color = (255, 190, 0, 255) if valid else (255, 75, 75, 255)
            draw.rounded_rectangle((x1, y1, x2, y2), radius=14, outline=color, width=6)
            cx, cy = item["image_center_px"]
            draw.ellipse((cx - 17, cy - 17, cx + 17, cy + 17), fill=color)
            draw.text((cx - 15, cy - 13), item["id"].split("P")[-1], font=small_font, fill=(0, 0, 0, 255))

    chart_top = source.height + 20
    draw.text((45, chart_top), "Shelf-position graph: center + edges from real observed 3D points",
              font=title_font, fill=(255, 255, 255, 255))
    draw.text((46, chart_top + 57),
              "amber whisker = 5th-95th percentile of that product's own 3D points (edges); "
              "dot = median (center); red = <8 observed points, not plotted",
              font=small_font, fill=(180, 220, 228, 255))

    axis_left, axis_right = 260, 2040
    for row_index, row in enumerate(product_rows):
        y = chart_top + 140 + row_index * 210
        items = row["products"]
        valid = [it for it in items if it["center_u"] is not None]
        draw.text((50, y - 12), row["shelf_id"], font=label_font, fill=(0, 240, 215, 255))
        draw.line((axis_left, y + 40, axis_right, y + 40), fill=(90, 115, 125, 255), width=3)

        if not valid:
            draw.text((axis_left, y + 20), "No positions with sufficient 3D evidence",
                      font=small_font, fill=(255, 100, 100, 255))
            continue

        lo = min(it["edge_left_u"] for it in valid)
        hi = max(it["edge_right_u"] for it in valid)
        span = max(hi - lo, 1e-9)
        pad = span * 0.06

        def to_px(u: float) -> float:
            return axis_left + (u - (lo - pad)) / (span + 2 * pad) * (axis_right - axis_left)

        for item in items:
            number = item["id"].split("P")[-1]
            if item["center_u"] is None:
                x = axis_left + 20 + int(number) * 55
                draw.ellipse((x - 15, y + 55, x + 15, y + 85), outline=(255, 90, 90, 255), width=4)
                draw.text((x - 6, y + 58), number, font=tiny_font, fill=(255, 130, 130, 255))
                continue
            xl, xc, xr = to_px(item["edge_left_u"]), to_px(item["center_u"]), to_px(item["edge_right_u"])
            # whisker = the product's real edge-to-edge extent on the shelf axis
            draw.line((xl, y + 55, xl, y + 85), fill=(255, 190, 0, 255), width=4)
            draw.line((xr, y + 55, xr, y + 85), fill=(255, 190, 0, 255), width=4)
            draw.line((xl, y + 70, xr, y + 70), fill=(255, 190, 0, 180), width=3)
            draw.ellipse((xc - 12, y + 58, xc + 12, y + 82), fill=(255, 190, 0, 255), outline=(0, 0, 0, 255))
            draw.text((xc - 6, y + 61), number, font=tiny_font, fill=(0, 0, 0, 255))

        draw.text((axis_left - 5, y + 95), f"{lo:+.3f}", font=small_font, fill=(180, 220, 228, 255))
        draw.text((axis_right - 130, y + 95), f"{hi:+.3f}  (COLMAP u)", font=small_font, fill=(180, 220, 228, 255))

    canvas.convert("RGB").save(OUTPUT_IMAGE, quality=95)
    total_dropped = sum(it["background_points_rejected"] for row in product_rows for it in row["products"])
    contaminated = [it["id"] for row in product_rows for it in row["products"] if it["background_points_rejected"] >= 5]
    print(f"Positioned {len(positioned)}/{sum(len(r['products']) for r in product_rows)} products "
          f"with center + edge coordinates")
    print(f"Background rejection: {total_dropped} points dropped across all boxes "
          f"(>={BACKGROUND_REJECT_RADIUS} units from that box's median)")
    if contaminated:
        print(f"  boxes with heavy contamination (>=5 points dropped): {', '.join(contaminated)}")
    print(OUTPUT_IMAGE)
    print(OUTPUT_JSON)


if __name__ == "__main__":
    main()
