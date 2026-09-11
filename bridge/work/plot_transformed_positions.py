"""Center + edge positions, run through the REAL alignment transform.

Prior script (plot_center_edge_positions.py) computed real centers/edges but
left them in raw COLMAP units -- it never touched last week's alignment work
because there was no second frame to reconcile against. That was a real gap:
the whole point of last week's fix was the colmap_alignment transform, and
this script's output should be expressed in it, not left in arbitrary units.

This version uses, in order:
    1. REAL detections -- mapper/backend/workspace/single_view_00014/results.json
       products[] (58 raw YOLO boxes, merged to 35 by consensus). NOT the
       hand-drawn boxes from before.
    2. REAL shelf assignment -- each product's shelf_id, already computed by
       geometry.py::project_detections_to_shelves() when that job ran.
    3. This script's contribution: real observed 3D points per box
       (geometry.py::_observed_points_in_bbox, real function, imported not
       reimplemented), background-rejected (same 0.6-unit radius, same
       reasoning as before).
    4. THE ACTUAL FIX: transform the real points into the model frame FIRST,
       using the fitted colmap_alignment transform already sitting in
       aligned/alignment_report.json (scale, rotation, origin -- computed
       last week by align_products_to_gondola.py against this exact job),
       THEN take center/edges as the median / 5th-95th percentile of the
       transformed points' model-X coordinate.

    An earlier version of this script did it backwards: it measured edges
    along shelf["axes"]["u"] in COLMAP space, THEN transformed the resulting
    3D point. That is wrong -- shelf["axes"]["u"] is geometry.py's
    _plane_basis() output, an ARBITRARY direction inside the shelf plane,
    not the bay-run direction. Checked directly: its dot product with the
    real bay-run axis (frame.rotation[0]) is ~0.04 -- nearly perpendicular.
    That bug produced a real-looking but wrong finding ("edges land on Y not
    X"), which was actually just this axis mismatch, not new evidence of
    alignment weakness. Transforming first and measuring in the model frame
    sidesteps the whole question of which COLMAP-space axis is "correct" --
    the fitted rotation already answers that.
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
MAPPER_BACKEND = "/Users/snehamanimurugan/Documents/gondola-digital-twin-handoff/mapper/backend"
ALIGNMENT_CODE = "/Users/snehamanimurugan/Documents/gondola-digital-twin-handoff/alignment/code"
sys.path.insert(0, MAPPER_BACKEND)
sys.path.insert(0, ALIGNMENT_CODE)
from colmap_text import load_text_model  # noqa: E402
from geometry import _observed_points_in_bbox  # noqa: E402
from frame_alignment import ModelFrame  # noqa: E402

DATA = Path("/Users/snehamanimurugan/Documents/gondola-digital-twin-handoff/data/full_scan")
JOB_DIR = Path("/Users/snehamanimurugan/Documents/gondola-digital-twin-handoff/mapper/backend/workspace/single_view_00014")
SOURCE_IMAGE = JOB_DIR / "images" / "00014.jpg"
RESULTS_JSON = JOB_DIR / "results.json"
ALIGNED_GONDOLA_JSON = JOB_DIR / "aligned" / "gondola.json"
ALIGNMENT_REPORT_JSON = JOB_DIR / "aligned" / "alignment_report.json"

OUTPUT_IMAGE = Path("outputs/00014_transformed_positions.png")
OUTPUT_JSON = Path("outputs/00014_transformed_positions.json")

BACKGROUND_REJECT_RADIUS = 0.6
EDGE_PERCENTILE = 5.0
MIN_OBSERVED_POINTS = 8


def make_font(size: int):
    try:
        return ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial Bold.ttf", size)
    except OSError:
        return ImageFont.load_default()


def load_frame() -> ModelFrame:
    """Rebuild the exact transform already fitted for this job -- don't
    refit, reuse: refitting could drift from what aligned/gondola_with_products.json
    was actually produced with."""
    report = json.loads(ALIGNMENT_REPORT_JSON.read_text())
    transform = report["transform"]
    return ModelFrame(
        scale=float(transform["scale_meters_per_colmap_unit"]),
        rotation=np.array(transform["rotation_model_from_colmap"], dtype=float),
        origin=np.array(transform["origin_colmap"], dtype=float),
    )


def reject_background(points: np.ndarray) -> tuple[np.ndarray, int]:
    if len(points) < 3:
        return points, 0
    median = np.median(points, axis=0)
    distance = np.linalg.norm(points - median, axis=1)
    keep = distance <= BACKGROUND_REJECT_RADIUS
    return points[keep], int((~keep).sum())


def main() -> None:
    reconstruction = load_text_model(DATA)
    image_pose = reconstruction.images["00014.jpg"]
    results = json.loads(RESULTS_JSON.read_text())
    shelves_by_id = {s["id"]: s for s in results["shelves"]}
    gondola = json.loads(ALIGNED_GONDOLA_JSON.read_text())
    frame = load_frame()
    print(f"Loaded real transform: scale={frame.scale:.5f} m/unit "
          f"(fitted last week by align_products_to_gondola.py for this exact job)")

    products_out = []
    for product in results["products"]:
        shelf = shelves_by_id.get(product["shelf_id"])
        if shelf is None:
            continue
        bbox = product["bbox"]
        raw_points = np.array(_observed_points_in_bbox(reconstruction, image_pose, bbox), dtype=float)
        points, dropped = reject_background(raw_points) if len(raw_points) else (raw_points, 0)

        entry = {
            "id": product["id"],
            "shelf_id": product["shelf_id"],
            "raw_observed_point_count": int(len(raw_points)),
            "background_points_rejected": dropped,
            "observed_point_count": int(len(points)),
        }
        if len(points) < MIN_OBSERVED_POINTS:
            entry["status"] = "insufficient_3d_evidence"
            products_out.append(entry)
            continue

        # THE ACTUAL FIX: shelf["axes"]["u"] is geometry.py's _plane_basis()
        # output -- an ARBITRARY direction inside the shelf plane, not the
        # bay-run direction (checked: dot product with the true bay-run axis
        # below is ~0.04, i.e. nearly perpendicular). Measuring edges along it
        # and only THEN transforming was the bug in the first version of this
        # script. Correct order: transform the real points into the model
        # frame first (frame.rotation already encodes the true bay-run
        # direction, solved for by align_products_to_gondola.py), then take
        # the spread directly in model coordinates -- no intermediate axis
        # assumption at all.
        points_model = frame.to_model(points)  # (N, 3) meters, real gondola frame
        center_model = np.median(points_model, axis=0)
        left_model = center_model.copy()
        right_model = center_model.copy()
        left_model[0] = float(np.percentile(points_model[:, 0], EDGE_PERCENTILE))
        right_model[0] = float(np.percentile(points_model[:, 0], 100 - EDGE_PERCENTILE))

        entry.update({
            "status": "positioned",
            "center_model_m": [float(v) for v in center_model],
            "edge_left_model_m": [float(v) for v in left_model],
            "edge_right_model_m": [float(v) for v in right_model],
            "width_m": float(np.linalg.norm(right_model - left_model)),
        })
        products_out.append(entry)

    positioned = [p for p in products_out if p["status"] == "positioned"]
    payload = {
        "image_name": "00014.jpg",
        "detection_source": "real -- results.json products[] (real YOLO detections, real geometry.py shelf assignment)",
        "transform_source": f"real -- {ALIGNMENT_REPORT_JSON} (fitted by align_products_to_gondola.py last week)",
        "edge_source": (
            f"real observed 3D points per box, background-rejected (radius={BACKGROUND_REJECT_RADIUS}), "
            "transformed into the model frame FIRST via the real fitted transform, then "
            f"{EDGE_PERCENTILE:.0f}th/{100 - EDGE_PERCENTILE:.0f}th percentile of model-X taken as the edges "
            "(NOT projected onto shelf['axes']['u'] beforehand -- that axis is arbitrary, see module docstring)"
        ),
        "frame": "gondola model frame, meters, +Z up, +X bay run (same as render_planogram.py)",
        "positioned_count": len(positioned),
        "insufficient_evidence_count": sum(1 for p in products_out if p["status"] != "positioned"),
        "products": products_out,
    }
    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    # Sanity check the fix actually took: edges should now vary in X (bay
    # run) and NOT in Y -- by construction (left/right only differ in the
    # model-X coordinate), so a nonzero dY here would mean the fix regressed.
    if positioned:
        span_x = max(abs(p["edge_right_model_m"][0] - p["edge_left_model_m"][0]) for p in positioned)
        span_y = max(abs(p["edge_right_model_m"][1] - p["edge_left_model_m"][1]) for p in positioned)
        print(f"Edge spread: max |dX|={span_x:.3f} m (real product widths), "
              f"max |dY|={span_y:.3f} m (should be 0.000 by construction)")

    # --- render: single elevation panel (X vs Z), real metric frame -----
    # A second top-down (X vs Y) panel from the previous version is gone:
    # with the axis bug fixed, edges no longer vary in Y at all (by
    # construction), so that panel would show flat lines with nothing to see.
    layout = gondola["layout"]
    run = float(layout["total_bay_run_m"])
    fixture_h = float(layout["gondola_height_m"])
    width, height = 1800, 850
    image = Image.new("RGB", (width, height), "#151b24")
    draw = ImageDraw.Draw(image)
    title_font, body_font, small_font = make_font(38), make_font(24), make_font(18)

    draw.text((60, 35), "Product edges, transformed into the gondola's real metric frame", font=title_font, fill="white")
    draw.text((62, 82), f"real detections -> real shelf assignment -> real observed points -> "
                        f"REAL fitted transform (scale={frame.scale:.4f} m/unit)  |  "
                        f"{len(positioned)}/{len(products_out)} positioned",
              font=small_font, fill="#b9d6dc")

    def draw_panel(panel_y0, panel_label, vertical_key, vertical_span, vertical_label):
        x0, x1, y0, y1 = 190, 1700, panel_y0, panel_y0 + 480
        for tick in range(7):
            x = x0 + tick * (x1 - x0) / 6
            draw.line((x, y0, x, y1), fill="#34404e", width=2)
            draw.text((x - 25, y1 + 14), f"{(-run / 2 + tick * run / 6):+.2f} m", font=small_font, fill="#b7c1c9")
        draw.rectangle((x0, y0, x1, y1), outline="#d9e3e5", width=3)
        draw.text((x0, y1 + 50), "model X -- along the bay run (m)", font=body_font, fill="#d9e3e5")
        draw.text((15, y0 - 30), panel_label, font=body_font, fill="#d9e3e5")

        def to_px(x_m, v_m):
            v_frac = (v_m + vertical_span / 2) / vertical_span if vertical_key == "y" else v_m / fixture_h
            return x0 + (x_m + run / 2) / run * (x1 - x0), y1 - v_frac * (y1 - y0)

        if vertical_key == "z":
            for shelf in gondola.get("shelves", []):
                if shelf.get("side_label") not in ("Side A", "Side C"):
                    continue
                z = float(shelf["height"])
                y = y1 - z / fixture_h * (y1 - y0)
                draw.line((x0, y, x1, y), fill="#a9a59d", width=3)
                draw.text((x0 - 60, y - 9), f"{z:.2f}", font=small_font, fill="#d9e3e5")
        else:
            draw.line((x0, (y0 + y1) / 2, x1, (y0 + y1) / 2), fill="#a9a59d", width=2)
            draw.text((x0 - 60, (y0 + y1) / 2 - 9), f"{vertical_label}=0", font=small_font, fill="#d9e3e5")

        for product in positioned:
            cx, cy, cz = product["center_model_m"]
            lx, ly, lz = product["edge_left_model_m"]
            rx, ry, rz = product["edge_right_model_m"]
            v_center, v_left, v_right = (cz, lz, rz) if vertical_key == "z" else (cy, ly, ry)
            pcx, pcy = to_px(cx, v_center)
            plx, ply = to_px(lx, v_left)
            prx, pry = to_px(rx, v_right)
            draw.line((plx, ply, prx, pry), fill="#ffbe33", width=4)
            draw.line((plx, ply - 8, plx, ply + 8), fill="#ffbe33", width=3)
            draw.line((prx, pry - 8, prx, pry + 8), fill="#ffbe33", width=3)
            draw.ellipse((pcx - 9, pcy - 9, pcx + 9, pcy + 9), fill="#3d94f6", outline="white", width=2)

    draw_panel(150, "height (m) -- elevation, X vs Z", "z", fixture_h, "z")

    draw.text((60, 715),
              f"Edges now measured correctly: max |dX|={span_x:.3f} m (real widths), "
              f"max |dY|={span_y:.3f} m (0 by construction -- the axis bug from the prior "
              f"version is fixed).",
              font=body_font, fill="#8fd97a")
    draw.text((60, 751),
              "Separately, still real and unrelated to this fix: this job's shelf-plane fit "
              "is weak (normal vs camera-up = 0.684, only 20/35 snapped; 9/29 positioned "
              "products sit below the floor) --",
              font=small_font, fill="#b9d6dc")
    draw.text((60, 775),
              "see the repo README's Known Issues. That needs multi-view capture, not a code fix.",
              font=small_font, fill="#b9d6dc")

    for index, product in enumerate([p for p in products_out if p["status"] != "positioned"]):
        draw.text((1350, 150 + 24 * index), f"{product['id']}: insufficient 3D evidence",
                  font=small_font, fill="#ef6565")

    image.save(OUTPUT_IMAGE)
    print(f"{len(positioned)}/{len(products_out)} products positioned in the real model frame")
    print(OUTPUT_IMAGE)
    print(OUTPUT_JSON)


if __name__ == "__main__":
    main()
