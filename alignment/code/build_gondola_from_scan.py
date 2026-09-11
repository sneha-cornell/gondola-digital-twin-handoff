"""
Fit a parametric gondola model directly to a COLMAP sparse reconstruction.

Bridges the two tools in this project that were previously connected only by
hand in Blender (see gondola_fixture_aligned*.blend, gondola_fixture_sparse.blend
-- a human repeatedly nudging scale/position to line a hand-built mesh up
against the raw scan):

    isolate_subject.py             filters a raw COLMAP point cloud down to
                                    just the photographed subject
    generate_parametric_gondola.py builds a clean parametric gondola mesh
                                    from explicit measurements

This script does the middle step with code instead of eyeballing it:
  1. isolate the subject (reuses isolate_subject.py's filters)
  2. fit an oriented top-down rectangle to it (RANSAC line fit + rectangle
     fit, adapted from gondola_scripts/src/quality/phase2_object_qc.py)
  3. detect shelf-height bands on the long faces and on the end-caps
     separately (reuses isolate_subject.py's detect_shelf_heights, run along
     each of the rectangle's two fitted axes)
  4. convert from COLMAP's arbitrary scale to meters using ONE known
     real-world measurement you provide
  5. call generate_parametric_gondola_fixture() with the fitted numbers

Auto-detected shelf counts and any dimension can still be overridden on the
command line if you know better than the geometry (e.g. you counted shelves
in a reference photo) -- this is meant to replace manual Blender alignment,
not remove your ability to specify how the gondola should look.

Usage:
    python3 build_gondola_from_scan.py \
        --colmap-txt sparse/0_txt \
        --anchor-dimension height --anchor-meters 1.8 \
        --output parametric_gondola_from_scan.json

Then build the .blend the same way as always:
    blender --background --python export_blender_scene.py -- \
        --results parametric_gondola_from_scan.json \
        --output gondola_fixture_from_scan.blend
"""

from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np

from isolate_subject import (
    build_basis,
    camera_centers_and_up,
    detect_shelf_heights,
    height_band_mask,
    parse_images_txt,
    parse_points3d_txt,
    proximity_mask,
    viewpoint_diversity_mask,
    write_ply,
)
from generate_parametric_gondola import generate_parametric_gondola_fixture
from frame_alignment import fit_model_frame


# ---------------------------------------------------------------------------
# Top-down oriented-rectangle fit. Ported from
# gondola_scripts/src/quality/phase2_object_qc.py's _ransac_lines /
# _fit_rectangle so this script has no dependency on that package (which
# imports sibling quality_report.py / trajectory_checks.py modules that are
# not present in this export).
# ---------------------------------------------------------------------------

def _density_mask(xyz: np.ndarray, k: int = 20, keep_percentile: float = 70.0) -> np.ndarray:
    if len(xyz) <= k + 1:
        return np.ones(len(xyz), dtype=bool)
    try:
        from sklearn.neighbors import KDTree
    except ImportError:
        return np.ones(len(xyz), dtype=bool)
    tree = KDTree(xyz)
    distances, _ = tree.query(xyz, k=k + 1)
    mean_distance = distances[:, 1:].mean(axis=1)
    density = 1.0 / (mean_distance + 1e-12)
    threshold = np.percentile(density, 100.0 - keep_percentile)
    return density >= threshold


def _main_cluster_mask(xy: np.ndarray, radius_factor: float = 0.65) -> np.ndarray:
    if len(xy) == 0:
        return np.zeros(0, dtype=bool)
    center = np.median(xy, axis=0)
    scale = max(float(np.ptp(xy[:, 0])), float(np.ptp(xy[:, 1])), 1e-12)
    return np.linalg.norm(xy - center, axis=1) <= radius_factor * scale


def _ransac_lines(
    xy: np.ndarray,
    *,
    n_lines: int = 4,
    iterations: int = 400,
    inlier_threshold: float | None = None,
    random_seed: int = 42,
) -> tuple[list[tuple[np.ndarray, np.ndarray]], list[float], list[int]]:
    if len(xy) < 10:
        return [], [], []
    scale = max(float(np.ptp(xy[:, 0])), float(np.ptp(xy[:, 1])), 1e-12)
    threshold = inlier_threshold or scale * 0.02
    remaining = xy.copy()
    lines: list[tuple[np.ndarray, np.ndarray]] = []
    residuals: list[float] = []
    supports: list[int] = []
    rng = np.random.default_rng(random_seed)

    for _ in range(n_lines):
        if len(remaining) < 10:
            break
        best_mask: np.ndarray | None = None
        best_count = 0
        for _ in range(iterations):
            selected = rng.choice(len(remaining), size=2, replace=False)
            p1, p2 = remaining[selected]
            direction = p2 - p1
            length = np.linalg.norm(direction)
            if length < 1e-9:
                continue
            normal = np.array([-direction[1], direction[0]]) / length
            distances = np.abs((remaining - p1) @ normal)
            mask = distances < threshold
            count = int(np.sum(mask))
            if count > best_count:
                best_count = count
                best_mask = mask
        if best_mask is None or best_count < max(8, int(0.02 * len(xy))):
            break

        inliers = remaining[best_mask]
        center = inliers.mean(axis=0)
        _, _, vt = np.linalg.svd(inliers - center, full_matrices=False)
        direction = vt[0]
        projection = (inliers - center) @ direction
        start = center + projection.min() * direction
        end = center + projection.max() * direction
        normal = np.array([-direction[1], direction[0]])
        residual = float(np.mean(np.abs((inliers - center) @ normal)))
        lines.append((start, end))
        residuals.append(residual)
        supports.append(best_count)
        remaining = remaining[~best_mask]

    return lines, residuals, supports


def _line_direction(line: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
    direction = line[1] - line[0]
    norm = np.linalg.norm(direction)
    if norm < 1e-12:
        return np.array([1.0, 0.0])
    direction = direction / norm
    return direction if direction[1] >= 0 else -direction


def _intersect(normal_a: np.ndarray, offset_a: float, normal_b: np.ndarray, offset_b: float) -> np.ndarray | None:
    matrix = np.stack([normal_a, normal_b])
    determinant = np.linalg.det(matrix)
    if abs(determinant) < 1e-9:
        return None
    return np.linalg.solve(matrix, np.array([offset_a, offset_b]))


def _counterclockwise(points: np.ndarray) -> np.ndarray:
    center = points.mean(axis=0)
    angles = np.arctan2(points[:, 1] - center[1], points[:, 0] - center[0])
    return points[np.argsort(angles)]


def _oriented_bbox_fallback(
    xy: np.ndarray, lines: list[tuple[np.ndarray, np.ndarray]], supports: list[int]
) -> dict[str, Any]:
    """Best-effort length/width/angle when a full 4-sided rectangle can't be
    fit (e.g. partial orbit coverage -- only 1-2 real faces have support).

    Picks an orientation from the best-supported RANSAC line if one exists
    (more reliable than blind PCA when most points come from one dominant
    face), otherwise falls back to PCA on the point cloud itself. Then just
    takes a robust (1st-99th percentile) span along that axis and its
    perpendicular -- always succeeds, but the perpendicular ("width") span
    is only as good as whatever partial data exists there, so callers should
    treat it as lower-confidence than a real 4-line rectangle fit.
    """
    if lines and supports:
        best = int(np.argmax(supports))
        direction = _line_direction(lines[best])
    else:
        centered = xy - xy.mean(axis=0)
        _, _, vt = np.linalg.svd(centered, full_matrices=False)
        direction = vt[0]
    angle = float(np.degrees(np.arctan2(direction[1], direction[0])) % 180)
    normal = np.array([-direction[1], direction[0]])

    center = xy.mean(axis=0)
    along = (xy - center) @ direction
    across = (xy - center) @ normal
    along_lo, along_hi = np.percentile(along, [1, 99])
    across_lo, across_hi = np.percentile(across, [1, 99])

    return {
        "length": float(along_hi - along_lo),
        "width": float(across_hi - across_lo),
        "angle_deg": angle,
        "fit_method": "oriented_bbox_fallback",
    }


def _fit_rectangle(lines: list[tuple[np.ndarray, np.ndarray]], xy: np.ndarray) -> dict[str, Any] | None:
    if len(lines) < 4:
        return None

    best: dict[str, Any] | None = None
    best_score = float("inf")
    indices = list(range(min(6, len(lines))))
    center = np.mean(xy, axis=0)
    cloud_scale = max(float(np.ptp(xy[:, 0])), float(np.ptp(xy[:, 1])), 1e-12)

    for chosen in combinations(indices, 4):
        selected = [lines[index] for index in chosen]
        directions = [_line_direction(line) for line in selected]
        for pair_a in combinations(range(4), 2):
            pair_b = tuple(index for index in range(4) if index not in pair_a)
            d_a0, d_a1 = directions[pair_a[0]], directions[pair_a[1]]
            d_b0, d_b1 = directions[pair_b[0]], directions[pair_b[1]]
            parallel_a = np.degrees(np.arccos(np.clip(abs(d_a0 @ d_a1), 0, 1)))
            parallel_b = np.degrees(np.arccos(np.clip(abs(d_b0 @ d_b1), 0, 1)))
            perpendicular = np.degrees(np.arccos(np.clip(abs(d_a0 @ d_b0), 0, 1)))
            perpendicular_error = abs(90.0 - perpendicular)
            if perpendicular_error > 35 or max(parallel_a, parallel_b) > 25:
                continue

            direction_a = d_a0 + np.sign(d_a0 @ d_a1) * d_a1
            direction_b = d_b0 + np.sign(d_b0 @ d_b1) * d_b1
            direction_a /= max(np.linalg.norm(direction_a), 1e-12)
            direction_b /= max(np.linalg.norm(direction_b), 1e-12)
            normal_a = np.array([-direction_a[1], direction_a[0]])
            normal_b = np.array([-direction_b[1], direction_b[0]])
            offsets_a = [float(normal_a @ ((selected[i][0] + selected[i][1]) / 2)) for i in pair_a]
            offsets_b = [float(normal_b @ ((selected[i][0] + selected[i][1]) / 2)) for i in pair_b]

            corners = []
            for offset_a in offsets_a:
                for offset_b in offsets_b:
                    corner = _intersect(normal_a, offset_a, normal_b, offset_b)
                    if corner is not None:
                        corners.append(corner)
            if len(corners) != 4:
                continue
            corners_array = _counterclockwise(np.asarray(corners))
            distances = np.linalg.norm(corners_array - center, axis=1)
            score = (
                parallel_a
                + parallel_b
                + 1.5 * perpendicular_error
                + 5.0 * float(np.mean(distances) / cloud_scale)
            )
            if score < best_score:
                best_score = score
                side_lengths = [
                    float(np.linalg.norm(corners_array[(i + 1) % 4] - corners_array[i]))
                    for i in range(4)
                ]
                sorted_sides = sorted(side_lengths)
                length = float(np.mean(sorted_sides[2:]))
                width = float(np.mean(sorted_sides[:2]))
                long_index = int(np.argmax(side_lengths))
                long_direction = corners_array[(long_index + 1) % 4] - corners_array[long_index]
                angle = float(np.degrees(np.arctan2(long_direction[1], long_direction[0])) % 180)
                best = {
                    "corners_2d": corners_array,
                    "center_2d": corners_array.mean(axis=0),
                    "side_lengths": side_lengths,
                    "length": length,
                    "width": width,
                    "angle_deg": angle,
                }
    return best


# ---------------------------------------------------------------------------
# Scan -> parametric-gondola pipeline
# ---------------------------------------------------------------------------

def isolate_and_fit(args: argparse.Namespace) -> tuple[dict[str, Any], Any]:
    """Fit the fixture's dimensions AND the frame they were measured in.

    Delegates to frame_alignment.fit_model_frame so the mesh's dimensions and
    the emitted COLMAP->model transform come from a single fit. Previously this
    function did the isolation/rectangle/shelf work itself and returned only the
    scalar dimensions -- the rotation and translation it had just computed were
    dropped on the floor, leaving nothing downstream able to place a product
    from the same scan against the resulting mesh.
    """
    frame = fit_model_frame(
        Path(args.colmap_txt),
        anchor_dimension=args.anchor_dimension,
        anchor_meters=args.anchor_meters,
        proximity_percentile=args.proximity_percentile,
        min_track_len=args.min_track_len,
        diversity_percentile=args.diversity_percentile,
        height_density_frac=args.height_density_frac,
        shelf_back_frac=args.shelf_back_frac,
        shelf_prominence=args.shelf_prominence,
        shelf_min_gap=args.shelf_min_gap,
        isolated_ply_output=args.isolated_ply_output,
    )
    bands = frame.raw["shelf_bands_m"]
    fit = {
        "raw_length": frame.raw["length"],
        "raw_width": frame.raw["width"],
        "raw_height": frame.raw["height"],
        "detected_side_ac_shelves": len(bands["side_ac"]),
        "detected_side_bd_shelves": len(bands["side_bd"]),
        "fit_confidence": frame.raw["fit_confidence"],
        "shelf_bands_m": bands,
    }
    return fit, frame


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fit a parametric gondola mesh to a COLMAP sparse reconstruction."
    )
    parser.add_argument("--colmap-txt", required=True, help="COLMAP TXT export dir (images.txt, points3D.txt)")
    parser.add_argument("--isolated-ply-output", default=None, help="Optional path to save the isolated point cloud")

    # Subject-isolation filters (same defaults as isolate_subject.py)
    parser.add_argument("--proximity-percentile", type=float, default=70.0)
    parser.add_argument("--min-track-len", type=int, default=3)
    parser.add_argument("--diversity-percentile", type=float, default=40.0)
    parser.add_argument("--height-density-frac", type=float, default=0.15)
    parser.add_argument("--density-percentile", type=float, default=70.0,
                         help="Top-down k-NN density keep-percentile before rectangle fitting")

    # Shelf-band detection tuning (same defaults as isolate_subject.py)
    parser.add_argument("--shelf-back-frac", type=float, default=0.5)
    parser.add_argument("--shelf-prominence", type=float, default=0.25)
    parser.add_argument("--shelf-min-gap", type=float, default=0.06)

    # Real-world scale anchor -- required, COLMAP scale is arbitrary
    parser.add_argument("--anchor-dimension", required=True, choices=["length", "width", "height"],
                         help="Which fitted dimension your --anchor-meters value corresponds to")
    parser.add_argument("--anchor-meters", type=float, required=True,
                         help="Known real-world size (in meters) of --anchor-dimension")
    parser.add_argument("--anchor-assumed", action="store_true",
                         help="Mark --anchor-meters as an assumed/typical value rather than an actual "
                              "measurement of this object -- stamps the output so it's not mistaken for one")

    # Manual overrides -- specify how you want the gondola to be, on top of
    # (or instead of) what the scan geometry detected.
    parser.add_argument("--bay-width", type=float, default=None,
                         help="Override bay width in meters (default: fitted length as one bay)")
    parser.add_argument("--num-bays", type=int, default=None,
                         help="Override bay count (default: 1, or fitted length / --bay-width)")
    parser.add_argument("--side-a-shelves", type=int, default=None, help="Override detected Side A shelf count")
    parser.add_argument("--side-c-shelves", type=int, default=None, help="Override detected Side C shelf count")
    parser.add_argument("--side-b-shelves", type=int, default=None, help="Override detected Side B shelf count")
    parser.add_argument("--side-d-shelves", type=int, default=None, help="Override detected Side D shelf count")
    parser.add_argument("--endcap-shelf-depth", type=float, default=0.35)
    parser.add_argument("--no-price-rails", action="store_true")
    parser.add_argument("--top-canopy", action="store_true")

    parser.add_argument("--even-shelf-spacing", action="store_true",
                         help="Space shelves evenly instead of placing them at the heights "
                              "detected in the scan. Even spacing is never where a real "
                              "fixture's boards are, so products positioned from the same "
                              "scan will not line up with them -- only use this when the "
                              "detected bands are clearly wrong.")
    parser.add_argument("--output", default="parametric_gondola_from_scan.json")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    fit, frame = isolate_and_fit(args)

    raw_value = {"length": fit["raw_length"], "width": fit["raw_width"], "height": fit["raw_height"]}[
        args.anchor_dimension
    ]
    if raw_value <= 1e-9:
        raise SystemExit(f"Fitted {args.anchor_dimension} is ~0 in raw units; cannot compute a scale factor.")
    # Same scale the frame transform uses -- the mesh and the transform have to
    # agree to the last digit or products land off by their ratio.
    scale = frame.scale

    length_m = fit["raw_length"] * scale
    width_m = fit["raw_width"] * scale
    height_m = fit["raw_height"] * scale
    print()
    anchor_note = " -- ASSUMED, not a measurement of this object" if args.anchor_assumed else ""
    print(f"Scale factor: {scale:.6f} meters/COLMAP-unit "
          f"(anchored on {args.anchor_dimension} = {args.anchor_meters} m{anchor_note})")
    print(f"Fitted real-world size: length={length_m:.3f} m  width={width_m:.3f} m  height={height_m:.3f} m")
    if fit["fit_confidence"] == "partial_coverage_fallback":
        print("CAUTION: no perpendicular edge family was found (partial orbit coverage) -- "
              "length is from the well-covered long faces, but WIDTH is a rough oriented-bbox "
              "estimate and end-cap shelf counts are unreliable. Re-run after recapturing the "
              "missing angles for a trustworthy width.")

    if args.bay_width is not None:
        bay_width = args.bay_width
        num_bays = args.num_bays if args.num_bays is not None else max(1, round(length_m / bay_width))
    else:
        num_bays = args.num_bays if args.num_bays is not None else 1
        bay_width = length_m / num_bays

    side_a = args.side_a_shelves if args.side_a_shelves is not None else (fit["detected_side_ac_shelves"] or 6)
    side_c = args.side_c_shelves if args.side_c_shelves is not None else (fit["detected_side_ac_shelves"] or 6)
    side_b = args.side_b_shelves if args.side_b_shelves is not None else (fit["detected_side_bd_shelves"] or 5)
    side_d = args.side_d_shelves if args.side_d_shelves is not None else (fit["detected_side_bd_shelves"] or 5)
    # Measured shelf planes beat evenly-spaced ones: the detector found these
    # bands in the scan, and products positioned from that same scan sit on
    # them. Overriding a side's count means the measured list no longer
    # describes that side, so fall back to even spacing for it.
    long_face_heights = fit["shelf_bands_m"]["side_ac"] or None
    end_cap_heights = fit["shelf_bands_m"]["side_bd"] or None
    if args.even_shelf_spacing:
        long_face_heights = end_cap_heights = None
    if long_face_heights and (side_a != len(long_face_heights) or side_c != len(long_face_heights)):
        print(f"Side A/C shelf count was overridden ({side_a}/{side_c} vs "
              f"{len(long_face_heights)} detected bands) -- using even spacing for the long faces.")
        long_face_heights = None
    if end_cap_heights and (side_b != len(end_cap_heights) or side_d != len(end_cap_heights)):
        print(f"Side B/D shelf count was overridden ({side_b}/{side_d} vs "
              f"{len(end_cap_heights)} detected bands) -- using even spacing for the end-caps.")
        end_cap_heights = None
    if long_face_heights:
        print("Long-face shelf heights (m, measured): "
              + ", ".join(f"{height:.3f}" for height in long_face_heights))
    print(f"Shelf counts -> Side A: {side_a}  Side C: {side_c}  Side B: {side_b}  Side D: {side_d}")
    print(f"Bays: {num_bays} x {bay_width:.3f} m = {num_bays * bay_width:.3f} m total run")

    model = generate_parametric_gondola_fixture(
        bay_width=bay_width,
        num_bays=num_bays,
        gondola_depth=width_m,
        gondola_height=height_m,
        side_a_shelves=side_a,
        side_c_shelves=side_c,
        side_b_shelves=side_b,
        side_d_shelves=side_d,
        endcap_shelf_depth=args.endcap_shelf_depth,
        side_a_shelf_heights=long_face_heights,
        side_c_shelf_heights=long_face_heights,
        side_b_shelf_heights=end_cap_heights,
        side_d_shelf_heights=end_cap_heights,
        has_price_tag_rails=not args.no_price_rails,
        has_top_canopy=args.top_canopy,
    )
    model["fit_from_scan"] = {
        "colmap_txt": str(args.colmap_txt),
        "anchor_dimension": args.anchor_dimension,
        "anchor_meters": args.anchor_meters,
        "anchor_assumed": args.anchor_assumed,
        "scale_meters_per_colmap_unit": scale,
        "raw_colmap_units": fit,
        "note": "Dimensions were fitted to the isolated sparse point cloud, not hand-entered.",
    }
    # The transform back to the source reconstruction. Anything else holding
    # coordinates from this COLMAP model (e.g. digital-twin-shelf-mapper's
    # product positions) can now be mapped into this mesh's frame directly,
    # instead of being eyeballed into place in Blender or guessed at from
    # bounding-box extents.
    model["colmap_alignment"] = frame.as_json()

    output_path = Path(args.output)
    output_path.write_text(json.dumps(model, indent=2), encoding="utf-8")
    print()
    print(f"PARAMETRIC_GONDOLA_FROM_SCAN_WRITTEN: {output_path}")


if __name__ == "__main__":
    main()
