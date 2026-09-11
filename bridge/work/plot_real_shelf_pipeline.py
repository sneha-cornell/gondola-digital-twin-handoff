"""Real mapper pipeline on a real photo: detect_shelf_planes + project_detections_to_shelves.

Unlike plot_center_edge_positions.py (which reimplemented a simplified version
of the mapper's bbox-observation logic), this script calls the mapper's ACTUAL
geometry.py functions, unmodified, imported directly:

    detect_shelf_planes()           -- RANSAC plane fitting + DBSCAN height
                                        clustering on the real sparse point
                                        cloud, exactly as main.py calls it
    project_detections_to_shelves() -- ray-cast each detection's bbox center
                                        from the camera, intersect shelf
                                        planes, cross-check against observed
                                        3D points inside the box, return each
                                        product's real p3d + shelf_local (u,v,n)

What is NOT real here, and why: the actual detector (backend/detector.py)
needs torch + ultralytics. Installing those was declined (cost/time), so the
bounding boxes below are the same hand-drawn stand-ins as the prior script --
clearly marked `detection_source: "manual_stand_in"` in the output. Everything
downstream of the boxes -- shelf-plane geometry, ray casting, shelf assignment,
shelf-local coordinates -- is the mapper's real, unmodified code running on
the real reconstruction.

One real problem found while wiring this up: feeding geometry.py's
detect_shelf_planes() the RAW, un-isolated sparse cloud (all 42,946 points --
the whole room, not just the gondola) found only 2 broad shelf planes, versus
the ~8 real bands the same scan gave up last week. That's because
detect_shelf_planes() in production never sees raw sparse points -- main.py
runs it on a DENSE reconstruction (DepthAnything + TSDF), which we don't have
here (also needs torch). So this script substitutes last week's real
isolation code instead: isolate_subject.py's proximity_mask +
viewpoint_diversity_mask + height_band_mask (unmodified, same defaults as
last week) strip the room down to just the fixture BEFORE handing the result
to geometry.py's detect_shelf_planes() -- the same two-repo chaining the user
asked for, not a new shortcut.

A second, smaller thing surfaced in the process: geometry.py's _normalize()
only guards an exact-zero norm, not a near-zero one, so a degenerate RANSAC
triple (near-duplicate sampled points) occasionally produces a near-infinite
unit vector and a RuntimeWarning (divide by zero / overflow in matmul).
Checked with np.seterr(all="raise") to confirm it's real, not incidental.
Harmless to the final result here -- NaN distances just always lose the
"best inlier count" comparison -- but it is a real latent bug in their code,
not mine.

Edges (this script's one addition on top of their code, same as last time):
project_detections_to_shelves returns one point per product, not an extent.
For each product, its own observed 3D points (mapper's real
_observed_points_in_bbox) are projected onto ITS ASSIGNED SHELF's real u-axis
(shelf["axes"]["u"], not a separately-fit axis) to get left/right edges --
after the same background-point rejection that caught real contamination
last time (wall/ceiling points visible through gaps between bags).

Caveat carried over from last week's PDF: geometry.py's shelf-matching
thresholds (SHELF_NORMAL_MIN_ALIGNMENT, OBSERVED_SHELF_MATCH_MAX_HEIGHT_DELTA_M
= 0.22 m, MAX_OVERFLOW_M = 0.28 m, MIN_SHELF_AREA_M2 = 0.20) are meter-sized
constants applied to this scan's raw COLMAP units. They only behave sanely
here because this particular scan's fitted scale happens to be close to
1 unit ~ 1 m (checked last week: 1.02-1.08 m/unit) -- a property of this scan,
not a guarantee geometry.py enforces.
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
GONDOLA_PIPELINE_CODE = "/Users/snehamanimurugan/Downloads/gondola_scan_pipeline/code"
sys.path.insert(0, MAPPER_BACKEND)
sys.path.insert(0, GONDOLA_PIPELINE_CODE)
from colmap_text import load_text_model  # noqa: E402
from geometry import (  # noqa: E402
    _observed_points_in_bbox,
    detect_shelf_planes,
    project_detections_to_shelves,
)
# Last week's isolation code (gondola_scan_pipeline), unmodified -- strips the
# raw sparse cloud down to just the fixture before geometry.py sees it.
from isolate_subject import (  # noqa: E402
    camera_centers_and_up,
    height_band_mask,
    parse_images_txt,
    parse_points3d_txt,
    proximity_mask,
    viewpoint_diversity_mask,
)

DATA = Path("/Users/snehamanimurugan/Downloads/data")
SOURCE = DATA / "images" / "00014.jpg"
OUTPUT_IMAGE = Path("outputs/00014_real_pipeline_positions.png")
OUTPUT_JSON = Path("outputs/00014_real_pipeline_positions.json")

BACKGROUND_REJECT_RADIUS = 0.6  # world units; validated against this dataset last time
EDGE_PERCENTILE = 5.0

# Same hand-drawn stand-in boxes as before -- NOT from a real detector.
# ultralytics/torch weren't installed (declined for cost); backend/models/best.pt
# exists in the repo and would replace this list entirely if run.
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


def hint_up_from_cameras(reconstruction) -> np.ndarray:
    """Exactly main.py's derivation: COLMAP camera +Y is image-down, so
    R.T[:, 1] is world-down; negate the sign-consensus average."""
    images = list(reconstruction.images.values())
    down_vectors = np.array([image.rotation.T[:, 1] for image in images], dtype=float)
    if len(down_vectors) < 2:
        return np.array([0.0, 0.0, 1.0])
    mean = down_vectors.mean(axis=0)
    signs = np.where(down_vectors @ mean >= 0, 1.0, -1.0)
    raw = (down_vectors * signs[:, None]).mean(axis=0)
    norm = float(np.linalg.norm(raw))
    return -(raw / norm) if norm > 1e-6 else np.array([0.0, 0.0, 1.0])


def reject_background(points: np.ndarray) -> tuple[np.ndarray, int]:
    if len(points) < 3:
        return points, 0
    median = np.median(points, axis=0)
    distance = np.linalg.norm(points - median, axis=1)
    keep = distance <= BACKGROUND_REJECT_RADIUS
    return points[keep], int((~keep).sum())


def isolate_fixture(hint_up: np.ndarray) -> np.ndarray:
    """Last week's real isolation chain (isolate_subject.py), same defaults,
    re-run here with colmap_text-compatible geometry (both parsers read the
    same points3D.txt / images.txt, same COLMAP coordinate convention)."""
    images = parse_images_txt(DATA / "images.txt")
    pts, _cols, tracks = parse_points3d_txt(DATA / "points3D.txt")
    cam_centers, _up, alignment = camera_centers_and_up(images)
    print(f"  isolate_subject up-vector alignment: {alignment:.3f} (1.0 = perfect)")

    prox_mask, _, prox_cutoff = proximity_mask(pts, tracks, cam_centers, keep_percentile=70.0)
    div_mask, _, div_cutoff = viewpoint_diversity_mask(pts, tracks, cam_centers, min_track_len=3, keep_percentile=40.0)
    combined = prox_mask & div_mask
    height_mask, band_lo, band_hi = height_band_mask(pts[combined], hint_up, density_frac=0.15)
    isolated = pts[combined][height_mask]
    print(f"  proximity (cutoff={prox_cutoff:.2f}) + diversity (cutoff={div_cutoff:.2f}) + "
          f"height band [{band_lo:.2f}, {band_hi:.2f}]: {len(pts)} -> {len(isolated)} points")
    return isolated


def main() -> None:
    reconstruction = load_text_model(DATA)
    image_pose = reconstruction.images["00014.jpg"]
    print(f"Loaded real reconstruction: {len(reconstruction.images)} cameras, "
          f"{len(reconstruction.points3d)} 3D points")

    hint_up = hint_up_from_cameras(reconstruction)
    print("Isolating the fixture (last week's isolate_subject.py, real code, unmodified):")
    isolated_points = isolate_fixture(hint_up)

    # --- REAL mapper code: shelf-plane detection, on the isolated cloud ------
    shelves = detect_shelf_planes(isolated_points, hint_up=hint_up)
    print(f"detect_shelf_planes() [real geometry.py]: found {len(shelves)} shelf plane(s)")
    for shelf in shelves:
        print(f"  {shelf['id']}: height={shelf['height']:.3f}  "
              f"{shelf['extents']['width']:.2f}x{shelf['extents']['depth']:.2f} units  "
              f"points={shelf['point_count']}")

    if not shelves:
        raise SystemExit(
            "detect_shelf_planes() found zero shelves on the real point cloud. "
            "Nothing downstream can run without at least one plane."
        )

    # --- build the manual stand-in detections list --------------------------
    detections = []
    row_lookup = {}
    for shelf_label, top, bottom, intervals in ROWS:
        for number, (left, right) in enumerate(intervals, start=1):
            det_id = f"{shelf_label}-P{number}"
            bbox = [float(left), float(top + 60), float(right), float(bottom - 35)]
            detections.append({
                "id": det_id,
                "image_name": "00014.jpg",
                "label": "product",
                "confidence": 1.0,  # manual box, not a model score
                "bbox": bbox,
            })
            row_lookup[det_id] = bbox

    # --- REAL mapper code: ray-cast + shelf assignment -----------------------
    products = project_detections_to_shelves(reconstruction, shelves, detections)
    print(f"project_detections_to_shelves() [real geometry.py]: "
          f"{len(products)}/{len(detections)} boxes assigned to a real shelf plane")

    products_by_id = {p["id"]: p for p in products}

    # --- this script's addition: edges from real observed points on the ------
    # --- REAL assigned shelf's own axis (not a separately-fit axis) ---------
    for product in products:
        shelf = next(s for s in shelves if s["id"] == product["shelf_id"])
        bbox = row_lookup[product["id"]]
        raw_points = np.array(_observed_points_in_bbox(reconstruction, image_pose, bbox), dtype=float)
        points, dropped = reject_background(raw_points) if len(raw_points) else (raw_points, 0)
        product["raw_observed_point_count"] = int(len(raw_points))
        product["background_points_rejected"] = dropped
        product["observed_point_count"] = int(len(points))
        axis_u = np.array(shelf["axes"]["u"], dtype=float)
        if len(points) >= 3:
            projected = points @ axis_u
            product["edge_left_u"] = float(np.percentile(projected, EDGE_PERCENTILE))
            product["edge_right_u"] = float(np.percentile(projected, 100 - EDGE_PERCENTILE))
            product["width_u"] = product["edge_right_u"] - product["edge_left_u"]
        else:
            product["edge_left_u"] = product["edge_right_u"] = product["width_u"] = None

    unassigned = [d["id"] for d in detections if d["id"] not in products_by_id]

    payload = {
        "image_name": "00014.jpg",
        "detection_source": "manual_stand_in",
        "detection_note": (
            "Bounding boxes are hand-drawn, not from backend/detector.py -- "
            "torch/ultralytics were not installed (cost declined). "
            "Everything from shelf-plane detection onward is the mapper's real, "
            "unmodified geometry.py code."
        ),
        "isolation_source": "gondola_scan_pipeline/code/isolate_subject.py [real, unmodified, last week's code]",
        "shelf_plane_source": "geometry.py detect_shelf_planes() [real, unmodified], run on the isolated cloud above",
        "shelf_assignment_source": "geometry.py project_detections_to_shelves() [real, unmodified]",
        "edge_source": (
            "this script: real observed 3D points per box, projected onto the assigned "
            "shelf's real axis_u, background-rejected (see BACKGROUND_REJECT_RADIUS), "
            f"{EDGE_PERCENTILE:.0f}th/{100 - EDGE_PERCENTILE:.0f}th percentile"
        ),
        "shelves_detected": len(shelves),
        "detections_submitted": len(detections),
        "products_assigned_to_a_shelf": len(products),
        "products_unassigned": len(unassigned),
        "unassigned_ids": unassigned,
        "shelves": [
            {
                "id": s["id"], "height": s["height"], "point_count": s["point_count"],
                "normal": s["normal"], "extents": s["extents"],
            }
            for s in shelves
        ],
        "products": products,
    }
    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    # --- render -------------------------------------------------------------
    source = Image.open(SOURCE).convert("RGBA")
    by_real_shelf: dict[str, list[dict]] = {}
    for product in products:
        by_real_shelf.setdefault(product["shelf_id"], []).append(product)

    chart_h = 220 + max(len(by_real_shelf), 1) * 220
    canvas = Image.new("RGBA", (source.width, source.height + chart_h), (15, 24, 33, 255))
    canvas.alpha_composite(source, (0, 0))
    draw = ImageDraw.Draw(canvas)
    title_font, label_font, small_font, tiny_font = (
        make_font(42), make_font(27), make_font(21), make_font(17)
    )

    id_to_bbox = row_lookup
    for shelf_label, top, bottom, intervals in ROWS:
        draw.rectangle((105, top, 2140, bottom), outline=(0, 230, 210, 255), width=6)
        draw.text((125, top + 10), shelf_label, font=label_font, fill=(0, 255, 225, 255))
        for number, (left, right) in enumerate(intervals, start=1):
            det_id = f"{shelf_label}-P{number}"
            bbox = id_to_bbox[det_id]
            product = products_by_id.get(det_id)
            color = (255, 190, 0, 255) if product is not None else (255, 75, 75, 255)
            draw.rounded_rectangle(tuple(bbox), radius=14, outline=color, width=6)
            cx = (bbox[0] + bbox[2]) / 2
            cy = bbox[1] + 40
            draw.ellipse((cx - 17, cy - 17, cx + 17, cy + 17), fill=color)
            draw.text((cx - 15, cy - 13), str(number), font=small_font, fill=(0, 0, 0, 255))
            real_shelf = product["shelf_id"] if product else "?"
            draw.text((bbox[0] + 2, bbox[3] - 26), real_shelf, font=tiny_font, fill=color)

    chart_top = source.height + 20
    draw.text((45, chart_top),
              "Real mapper pipeline: detect_shelf_planes() + project_detections_to_shelves()",
              font=title_font, fill=(255, 255, 255, 255))
    draw.text((46, chart_top + 57),
              f"{len(shelves)} real shelf planes detected from the sparse point cloud "
              f"(labels on boxes = the REAL shelf each was assigned to, not the drawn row)",
              font=small_font, fill=(180, 220, 228, 255))
    draw.text((46, chart_top + 90),
              "amber whisker = 5th-95th pct of real observed 3D points on the assigned shelf's real axis; "
              "red = not assigned to any real shelf plane",
              font=small_font, fill=(180, 220, 228, 255))

    axis_left, axis_right = 260, 2040
    sorted_shelf_ids = sorted(by_real_shelf.keys(), key=lambda sid: -next(s["height"] for s in shelves if s["id"] == sid))
    for row_index, shelf_id in enumerate(sorted_shelf_ids):
        items = by_real_shelf[shelf_id]
        shelf_height = next(s["height"] for s in shelves if s["id"] == shelf_id)
        y = chart_top + 150 + row_index * 220
        draw.text((50, y - 12), f"{shelf_id}  (z={shelf_height:.3f})", font=label_font, fill=(0, 240, 215, 255))
        draw.line((axis_left, y + 40, axis_right, y + 40), fill=(90, 115, 125, 255), width=3)

        valid = [it for it in items if it["edge_left_u"] is not None]
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
            label = item["id"].split("P")[-1]
            if item["edge_left_u"] is None:
                continue
            xl, xr = to_px(item["edge_left_u"]), to_px(item["edge_right_u"])
            xc = to_px(item["shelf_local"]["u"])
            draw.line((xl, y + 55, xl, y + 85), fill=(255, 190, 0, 255), width=4)
            draw.line((xr, y + 55, xr, y + 85), fill=(255, 190, 0, 255), width=4)
            draw.line((xl, y + 70, xr, y + 70), fill=(255, 190, 0, 180), width=3)
            draw.ellipse((xc - 12, y + 58, xc + 12, y + 82), fill=(255, 190, 0, 255), outline=(0, 0, 0, 255))
            draw.text((xc - 6, y + 61), label, font=tiny_font, fill=(0, 0, 0, 255))

        draw.text((axis_left - 5, y + 95), f"{lo:+.3f}", font=small_font, fill=(180, 220, 228, 255))
        draw.text((axis_right - 160, y + 95), f"{hi:+.3f}  (COLMAP u, real shelf axis)",
                  font=small_font, fill=(180, 220, 228, 255))

    canvas.convert("RGB").save(OUTPUT_IMAGE, quality=95)
    print(f"\n{len(products)}/{len(detections)} boxes assigned to a real detected shelf plane")
    print(OUTPUT_IMAGE)
    print(OUTPUT_JSON)


if __name__ == "__main__":
    main()
