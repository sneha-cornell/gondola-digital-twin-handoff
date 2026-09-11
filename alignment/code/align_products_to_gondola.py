"""
Put digital-twin-shelf-mapper product coordinates into the parametric gondola's frame.

THE PROBLEM

The two halves of this project work in different coordinate systems and neither
one records the relationship between them:

  * digital-twin-shelf-mapper emits products (results.json -> products[].p3d)
    and detected shelf planes in the *raw COLMAP world frame* of its own scan.
    apply_layout_model() only ever applies a uniform scale to that frame -- and
    only when layout_config.json supplies reference_width_m / reference_depth_m.
    With both null it returns scale 1.0 / "scene_units": untouched COLMAP units,
    arbitrary origin, arbitrary orientation.

  * build_gondola_from_scan.py emits the fixture in a canonical frame: +Z up
    with 0.0 at the floor, +X along the bay run centred on 0.0, +Y depth, in
    meters. It computes the COLMAP->canonical rotation internally, then discards
    it, keeping only scale_meters_per_colmap_unit.

So a product at p3d and a shelf board in the gondola JSON differ by an unknown
rotation, an unknown translation and an unknown scale. Recovering that from the
bounding-box min/max of the two clouds cannot work: the product cloud's extremes
include background structure the gondola fit deliberately filtered out, and
extents are invariant to the 180 degree flip about up, so the front face and the
back face score identically.

WHAT THIS DOES INSTEAD

Both sides observed the same physical fixture, so both can be expressed in a
frame defined by that fixture rather than by either reconstruction. The product
side already did the hard part -- it fitted shelf planes -- so the frame comes
out of those planes and their inlier points:

  1. up        the shelf planes' shared normal, sign-corrected against the
               camera-derived up vector (RANSAC leaves the sign arbitrary)
  2. bay run   the dominant horizontal direction of the shelf inlier points.
               NOT shelf["axes"]["u"]: detect_shelf_planes builds that basis
               with _plane_basis(), which picks an arbitrary rotation inside
               the plane, so the shelf "boards" are axis-aligned boxes in a
               frame unrelated to the fixture's own long axis.
  3. centre    robust (1st-99th percentile) midpoint of those points along the
               run and depth axes
  4. scale +   solved together by matching the *set* of detected shelf heights
     floor     against the gondola JSON's shelf heights (1-D ICP). This is the
               step min/max anchoring was standing in for, and why it kept
               drifting: the lowest detected shelf is not the floor, so
               aligning extremes bakes the unknown bottom-shelf clearance in
               as a constant offset.
  5. flip      of the two rotations satisfying 1-3 (they differ by 180 degrees
               about up), the one that seats more products on a real board.

Every step reports a residual, so a bad alignment announces itself instead of
quietly misplacing products.

USAGE

    python3 align_products_to_gondola.py \
        --results   /path/to/workspace/<job>/results.json \
        --gondola   output/gondola_from_aws_scan.json \
        --colmap-txt /path/to/workspace/<job>/text \
        --output    output/gondola_with_products.json

--colmap-txt is optional but recommended: it supplies the up vector, which
fixes the shelf normal's arbitrary sign.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from frame_alignment import ModelFrame


# ---------------------------------------------------------------------------
# Loading the product side
# ---------------------------------------------------------------------------

def _normalize(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    return vector / norm if norm > 1e-12 else vector


def load_product_points(payload: Any) -> tuple[np.ndarray, list[dict]]:
    """Pull 3D product positions out of any of this project's product files.

    Handles results.json (products[].p3d), the Blender placement exports
    (placements[].location) and product_coordinates.json (a flat list of
    {x, y, z}) -- all three describe the same COLMAP-frame points.
    """
    records: list[dict] = []
    if isinstance(payload, dict):
        for key in ("products", "placements"):
            if isinstance(payload.get(key), list) and payload[key]:
                records = payload[key]
                break
    elif isinstance(payload, list):
        records = payload

    points: list[list[float]] = []
    kept: list[dict] = []
    for record in records:
        if not isinstance(record, dict):
            continue
        position = None
        for key in ("p3d", "location", "position", "world_position", "xyz"):
            value = record.get(key)
            if isinstance(value, (list, tuple)) and len(value) == 3:
                position = [float(component) for component in value]
                break
        if position is None and all(axis in record for axis in ("x", "y", "z")):
            position = [float(record["x"]), float(record["y"]), float(record["z"])]
        if position is None:
            continue
        points.append(position)
        kept.append(record)

    if not points:
        raise SystemExit(
            "No 3D product positions found. Expected results.json (products[].p3d), "
            "a Blender placements JSON (placements[].location), or a list of {x, y, z}."
        )
    return np.array(points, dtype=float), kept


def load_shelf_planes(payload: dict) -> list[dict]:
    shelves = payload.get("shelves")
    if not isinstance(shelves, list) or not shelves:
        raise SystemExit(
            "The results JSON has no 'shelves' array. The alignment is driven by the "
            "detected shelf planes, so run the analyze step first (POST /api/analyze/<job>)."
        )
    missing = [key for key in ("normal", "axes", "bounds", "height") if key not in shelves[0]]
    if missing:
        raise SystemExit(f"Shelf entries are missing {missing}; expected detect_shelf_planes() output.")
    return shelves


def shelf_inlier_points(shelves: list[dict]) -> np.ndarray | None:
    """The RANSAC inlier points detect_shelf_planes stashes on each shelf.

    detect_shelf_planes writes them as `_cluster_points` and nothing strips the
    key before results.json is written, so they are normally available -- and
    they are the best evidence for the fixture's horizontal orientation.
    """
    clouds = [
        np.asarray(shelf["_cluster_points"], dtype=float)
        for shelf in shelves
        if isinstance(shelf.get("_cluster_points"), list) and shelf["_cluster_points"]
    ]
    if not clouds:
        return None
    cloud = np.concatenate(clouds, axis=0)
    return cloud if cloud.ndim == 2 and cloud.shape[1] == 3 else None


def up_from_colmap(colmap_dir: Path) -> np.ndarray:
    """World-up from camera orientations, matching how both pipelines derive it.

    COLMAP's camera +Y points down in the image, so R.T[:, 1] is world-down.
    Sign-consensus averaged before negating, so one upside-down frame in the
    set cannot cancel the others out.
    """
    from isolate_subject import parse_images_txt, quat_to_R

    images = parse_images_txt(colmap_dir / "images.txt")
    if not images:
        raise SystemExit(f"No camera poses in {colmap_dir / 'images.txt'}")
    down = np.array([quat_to_R(image["q"]).T[:, 1] for image in images.values()], dtype=float)
    mean = down.mean(axis=0)
    signs = np.where(down @ mean >= 0, 1.0, -1.0)
    return -_normalize((down * signs[:, None]).mean(axis=0))


# ---------------------------------------------------------------------------
# The gondola side
# ---------------------------------------------------------------------------

def long_face_heights(gondola: dict) -> list[float]:
    """Heights (m above floor) of the long-face shelves -- what a scan mostly sees."""
    by_side: dict[str, list[float]] = {}
    for shelf in gondola.get("shelves", []):
        by_side.setdefault(str(shelf.get("side_label", "?")), []).append(float(shelf["height"]))
    heights: list[float] = []
    for side in ("Side A", "Side C"):
        heights.extend(by_side.get(side, []))
    if not heights:
        heights = [height for side_heights in by_side.values() for height in side_heights]
    return sorted(set(heights))


# ---------------------------------------------------------------------------
# Fixture axes, from the shelf points
# ---------------------------------------------------------------------------

def plane_basis(up: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    helper = np.array([1.0, 0.0, 0.0]) if abs(up[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    first = _normalize(np.cross(up, helper))
    return first, _normalize(np.cross(up, first))


def estimate_fixture_axes(cloud: np.ndarray, up: np.ndarray) -> dict[str, Any]:
    """Find the fixture's own horizontal axes and centre from a point cloud.

    A gondola's shelf boards form a long, thin horizontal slab, so the dominant
    in-plane direction of the shelf points is the bay run. Points are trimmed to
    their central 98% per axis before the SVD, so a few stray inliers on a
    neighbouring fixture can't rotate the frame.
    """
    axis_a, axis_b = plane_basis(up)
    coords = np.column_stack([cloud @ axis_a, cloud @ axis_b])
    lo, hi = np.percentile(coords, [1, 99], axis=0)
    trimmed = coords[np.all((coords >= lo) & (coords <= hi), axis=1)]
    if len(trimmed) < 10:
        trimmed = coords

    centred = trimmed - np.median(trimmed, axis=0)
    _, singular, vt = np.linalg.svd(centred, full_matrices=False)
    direction = vt[0]
    run_axis = _normalize(direction[0] * axis_a + direction[1] * axis_b)
    depth_axis = _normalize(np.cross(up, run_axis))

    run = cloud @ run_axis
    depth = cloud @ depth_axis
    run_lo, run_hi = np.percentile(run, [1, 99])
    depth_lo, depth_hi = np.percentile(depth, [1, 99])
    # Anisotropy: how much longer the dominant direction is than the other one.
    # Near 1.0 the SVD is picking a direction out of noise and the yaw is not
    # actually determined -- worth telling the caller about.
    anisotropy = float(singular[0] / max(singular[1], 1e-12))

    return {
        "run_axis": run_axis,
        "depth_axis": depth_axis,
        "run_span": float(run_hi - run_lo),
        "depth_span": float(depth_hi - depth_lo),
        "run_centre": float(0.5 * (run_lo + run_hi)),
        "depth_centre": float(0.5 * (depth_lo + depth_hi)),
        "anisotropy": anisotropy,
        "point_count": int(len(cloud)),
    }


# ---------------------------------------------------------------------------
# Scale + floor offset, from shelf-height sets
# ---------------------------------------------------------------------------

def fit_scale_and_floor(
    detected_heights: np.ndarray,
    model_heights: list[float],
    *,
    scale_hint: float,
    iterations: int = 40,
    max_scale_drift: float = 1.6,
) -> dict[str, Any]:
    """Solve z_model = scale * h_colmap + offset by matching shelf-height sets.

    1-D iterative closest point: assign each detected shelf to the nearest model
    shelf height, least-squares refit (scale, offset) on the pairs, repeat.
    Seeded from `scale_hint` at every possible vertical shift, because matching
    the wrong shelf to the wrong shelf is a local minimum a single start falls
    straight into.

    Two guards keep it off the degenerate optimum, which is otherwise very
    attractive: driving the scale toward zero collapses every detected shelf
    onto ONE model shelf, scoring a perfect residual and a maximal match count.
    So a candidate must (a) spread its matches over at least two distinct model
    shelves, in the same vertical order as the detections, and (b) stay within
    `max_scale_drift` of the seed scale, which came from a dimension both sides
    measured independently.
    """
    model = np.array(sorted(set(float(h) for h in model_heights)), dtype=float)
    detected = np.sort(np.asarray(detected_heights, dtype=float))
    if len(detected) < 2 or len(model) < 2:
        raise SystemExit("Need at least two detected shelves and two model shelves to fit scale.")

    pitch = float(np.median(np.diff(model)))
    scale_lo, scale_hi = scale_hint / max_scale_drift, scale_hint * max_scale_drift
    best: dict[str, Any] | None = None
    rejected_degenerate = 0

    for seed_index in range(len(model)):
        scale = float(scale_hint)
        offset = float(model[seed_index] - scale * detected[0])
        for _ in range(iterations):
            projected = scale * detected + offset
            nearest = model[np.abs(projected[:, None] - model[None, :]).argmin(axis=1)]
            # Only pairs within half a shelf pitch inform the refit; the rest
            # are detections of something the model has no shelf for.
            inliers = np.abs(projected - nearest) <= pitch * 0.5
            if int(inliers.sum()) < 2 or len(np.unique(nearest[inliers])) < 2:
                break
            design = np.column_stack([detected[inliers], np.ones(int(inliers.sum()))])
            solution, *_ = np.linalg.lstsq(design, nearest[inliers], rcond=None)
            new_scale = float(np.clip(solution[0], scale_lo, scale_hi))
            new_offset = float(solution[1])
            converged = abs(new_scale - scale) < 1e-9 and abs(new_offset - offset) < 1e-9
            scale, offset = new_scale, new_offset
            if converged:
                break

        projected = scale * detected + offset
        nearest = model[np.abs(projected[:, None] - model[None, :]).argmin(axis=1)]
        inliers = np.abs(projected - nearest) <= pitch * 0.5
        matched_model = nearest[inliers]
        distinct = int(len(np.unique(matched_model)))
        if not (scale_lo <= scale <= scale_hi) or int(inliers.sum()) < 2:
            continue
        # Order-preserving: shelves stack, so sorted detections must map to
        # non-decreasing model heights. Collapsed and shuffled fits fail here.
        if distinct < 2 or np.any(np.diff(matched_model) < 0):
            rejected_degenerate += 1
            continue
        candidate = {
            "scale": scale,
            "offset": offset,
            "matched": int(inliers.sum()),
            "distinct_model_shelves_matched": distinct,
            "detected_count": int(len(detected)),
            "model_count": int(len(model)),
            "rms_m": float(np.sqrt(np.mean((projected[inliers] - nearest[inliers]) ** 2))),
            "max_error_m": float(np.max(np.abs(projected[inliers] - nearest[inliers]))),
            "seed_model_index": seed_index,
            "scale_drift_vs_hint": float(scale / scale_hint),
        }
        # Covering more distinct model shelves is the strongest evidence, then
        # more matched detections, then a tighter residual.
        key = (-distinct, -candidate["matched"], candidate["rms_m"])
        if best is None or key < (
            -best["distinct_model_shelves_matched"], -best["matched"], best["rms_m"]
        ):
            best = candidate

    if best is None:
        raise SystemExit(
            "Could not match the detected shelf heights to the gondola's shelf heights "
            f"({rejected_degenerate} candidate fits were rejected as collapsed or "
            "out-of-order). Check that both describe the same fixture, and note that the "
            "generator's default even shelf spacing will not match a real fixture -- "
            "rebuild the gondola JSON with measured shelf heights first."
        )
    return best


# ---------------------------------------------------------------------------
# Frame construction
# ---------------------------------------------------------------------------

def build_frame(
    axes: dict[str, Any],
    up: np.ndarray,
    *,
    scale: float,
    floor_height: float,
    flip: bool,
) -> ModelFrame:
    """Assemble the COLMAP -> gondola-model similarity transform.

    `floor_height` is in COLMAP up-coordinates (dot(point, up)): the height that
    becomes model Z = 0. `flip` applies the 180 degree rotation about up, the
    one degree of freedom the geometry cannot resolve by itself.
    """
    x_axis = -axes["run_axis"] if flip else axes["run_axis"]
    y_axis = _normalize(np.cross(up, x_axis))   # right-handed; flipping X alone would mirror
    rotation = np.stack([x_axis, y_axis, up])

    # {run, depth, up} is orthonormal, so the fixture centre reconstructs
    # directly from the projections onto it.
    origin = (
        axes["run_centre"] * axes["run_axis"]
        + axes["depth_centre"] * axes["depth_axis"]
        + floor_height * up
    )
    return ModelFrame(scale=scale, rotation=rotation, origin=origin)


# ---------------------------------------------------------------------------
# Snapping products onto model shelves
# ---------------------------------------------------------------------------

def snap_products(
    points_model: np.ndarray,
    gondola: dict,
    *,
    tolerance: float,
    max_clearance: float = 0.25,
) -> list[dict | None]:
    """Assign each product to the model shelf board it is standing on.

    A product's 3D position is its *centre*, which sits 5-15 cm above the board
    it rests on -- so this measures clearance above the board's top surface, not
    distance to the board's mid-plane, and prefers the nearest board *below* the
    product. Scoring by |distance to centroid| instead would push every product
    on a tall item toward the shelf above it.

    Note the bounds convention differs between the two projects: the gondola
    generator writes `bounds` relative to the shelf centroid, while
    detect_shelf_planes writes absolute projections onto its plane basis. These
    are the generator's shelves, so bounds are centroid-relative here.
    """
    shelves = gondola.get("shelves", [])
    assignments: list[dict | None] = []
    for point in points_model:
        best: dict | None = None
        for shelf in shelves:
            axis_u = np.array(shelf["axes"]["u"], dtype=float)
            axis_v = np.array(shelf["axes"]["v"], dtype=float)
            normal = np.array(shelf["normal"], dtype=float)
            offset = point - np.array(shelf["centroid"], dtype=float)
            local_u = float(offset @ axis_u)
            local_v = float(offset @ axis_v)
            bounds_u, bounds_v = shelf["bounds"]["u"], shelf["bounds"]["v"]
            overflow = max(
                0.0,
                bounds_u[0] - local_u, local_u - bounds_u[1],
                bounds_v[0] - local_v, local_v - bounds_v[1],
            )
            if overflow > tolerance:
                continue
            # volume["n"] is absolute along the normal; [1] is the board's top.
            board_top = float(shelf["mesh"]["volume"]["n"][1])
            clearance = float(point @ normal) - board_top
            if clearance < -tolerance or clearance > max_clearance:
                continue
            # Resting on the board sorts ahead of sunk into it, then nearest wins.
            score = (0 if clearance >= 0 else 1, abs(clearance), overflow)
            if best is None or score < best["_score"]:
                best = {
                    "_score": score,
                    "shelf_id": shelf["id"],
                    "side_label": shelf.get("side_label"),
                    "unit": shelf.get("unit"),
                    "shelf_local": {"u": local_u, "v": local_v, "n": clearance},
                    "clearance_m": clearance,
                    "footprint_overflow_m": overflow,
                }
        if best is not None:
            best.pop("_score")
        assignments.append(best)
    return assignments


def score_alignment(points_model: np.ndarray, gondola: dict, *, tolerance: float,
                    max_clearance: float = 0.25) -> dict[str, Any]:
    assignments = snap_products(points_model, gondola, tolerance=tolerance,
                                max_clearance=max_clearance)
    placed = [assignment for assignment in assignments if assignment is not None]
    clearances = [assignment["clearance_m"] for assignment in placed]
    layout = gondola.get("layout", {})
    half_run = 0.5 * float(layout.get("total_bay_run_m", 0.0))
    half_depth = 0.5 * float(layout.get("gondola_depth_m", 0.0))
    height = float(layout.get("gondola_height_m", 0.0))
    endcap = float(layout.get("endcap_shelf_depth_m", 0.0))
    inside = int(np.sum(
        (np.abs(points_model[:, 0]) <= half_run + endcap + tolerance)
        & (np.abs(points_model[:, 1]) <= half_depth + tolerance)
        & (points_model[:, 2] >= -tolerance)
        & (points_model[:, 2] <= height + tolerance)
    ))
    return {
        "product_count": int(len(points_model)),
        "snapped_to_a_shelf": len(placed),
        "snapped_fraction": float(len(placed) / max(len(points_model), 1)),
        "inside_fixture_bbox": inside,
        "inside_fraction": float(inside / max(len(points_model), 1)),
        "median_clearance_m": float(np.median(clearances)) if clearances else None,
        "assignments": assignments,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Transform digital-twin-shelf-mapper product coordinates into the gondola model frame."
    )
    parser.add_argument("--results", required=True,
                        help="Product-side results.json (needs shelves[]; products[] unless --products is given)")
    parser.add_argument("--gondola", required=True, help="Gondola JSON from build_gondola_from_scan.py")
    parser.add_argument("--products", default=None,
                        help="Optional separate product file (Blender placements / product_coordinates JSON)")
    parser.add_argument("--colmap-txt", default=None,
                        help="Product scan's COLMAP TXT dir; supplies the up vector (recommended)")
    parser.add_argument("--anchor", choices=["run", "height", "depth"], default="run",
                        help="Which fixture dimension seeds the scale before shelf matching (default: run)")
    parser.add_argument("--flip", choices=["auto", "on", "off"], default="auto",
                        help="180 degree rotation about up. 'auto' picks whichever seats more products.")
    parser.add_argument("--tolerance", type=float, default=0.12,
                        help="Shelf-snap footprint tolerance in meters (default: 0.12)")
    parser.add_argument("--max-clearance", type=float, default=0.25,
                        help="Most a product centre may sit above a board and still count as "
                             "resting on it, in meters (default: 0.25). A product centre is "
                             "realistically 5-15 cm up; a loose value lets a mis-triangulated "
                             "detection get adopted by the shelf below it.")
    parser.add_argument("--output", default=None,
                        help="Write the gondola JSON with products[] filled in, in model coords")
    parser.add_argument("--report", default=None, help="Write the alignment report JSON here")
    return parser


def main() -> None:
    args = build_parser().parse_args()

    results = json.loads(Path(args.results).read_text(encoding="utf-8"))
    gondola = json.loads(Path(args.gondola).read_text(encoding="utf-8"))
    product_payload = json.loads(Path(args.products).read_text(encoding="utf-8")) if args.products else results

    shelves = load_shelf_planes(results)
    points_colmap, product_records = load_product_points(product_payload)
    print(f"Loaded {len(shelves)} detected shelf planes and {len(points_colmap)} product positions")

    # 1. Up. The shelf planes' own normal is what the products rest on and what
    # every detected height was measured against, so it is the up vector -- the
    # cameras only settle its sign, which RANSAC leaves arbitrary.
    up = _normalize(np.array(shelves[0]["normal"], dtype=float))
    if args.colmap_txt:
        camera_up = up_from_colmap(Path(args.colmap_txt))
        if float(up @ camera_up) < 0:
            up = -up
        print(f"Shelf normal vs camera up: {float(up @ camera_up):.3f} (1.0 = perfect)")
        if float(up @ camera_up) < 0.85:
            print("WARNING: the shelf planes are more than ~30 degrees off the camera-derived "
                  "up vector. One of the two is wrong; the shelf detection may have locked onto "
                  "a wall or the floor rather than the shelf boards.")
    else:
        print("WARNING: no --colmap-txt, so the shelf normal's arbitrary RANSAC sign is taken "
              "as up. If the result comes out upside down, pass --colmap-txt.")

    # 2. Fixture axes and centre, from the shelf inlier points where available.
    cloud = shelf_inlier_points(shelves)
    if cloud is None:
        cloud = points_colmap
        print("WARNING: shelves carry no _cluster_points, so the bay-run direction and fixture "
              "centre come from the product positions alone. Products only cover the shelves "
              "they were detected on, so the centre may be biased toward the scanned face.")
    axes = estimate_fixture_axes(cloud, up)
    print(f"Bay-run direction from {axes['point_count']} points, anisotropy {axes['anisotropy']:.2f} "
          f"(run {axes['run_span']:.3f} x depth {axes['depth_span']:.3f} COLMAP units)")
    if axes["anisotropy"] < 1.5:
        print("WARNING: the point cloud is nearly square in plan, so the dominant direction is "
              "weak evidence for the bay run. Check the result against a reference photo.")

    # 3. Seed the scale from a fixture dimension both sides measured.
    layout = gondola.get("layout", {})
    detected_heights = np.array(
        [float(np.array(shelf["centroid"], dtype=float) @ up) for shelf in shelves], dtype=float
    )
    if args.anchor == "run":
        anchor_model, anchor_detected = float(layout.get("total_bay_run_m", 0.0)), axes["run_span"]
    elif args.anchor == "depth":
        anchor_model, anchor_detected = float(layout.get("gondola_depth_m", 0.0)), axes["depth_span"]
    else:
        anchor_model = float(layout.get("gondola_height_m", 0.0))
        anchor_detected = float(detected_heights.max() - detected_heights.min())
    if anchor_model <= 0 or anchor_detected <= 0:
        raise SystemExit(
            f"Cannot seed the scale from --anchor {args.anchor}: "
            f"model={anchor_model}, detected={anchor_detected}"
        )
    scale_hint = anchor_model / anchor_detected
    print(f"Scale seed from {args.anchor}: {anchor_model:.3f} m / {anchor_detected:.3f} units "
          f"= {scale_hint:.6f} m per COLMAP unit")

    # 4. Refine scale and recover the floor by matching shelf-height sets.
    model_heights = long_face_heights(gondola)
    fit = fit_scale_and_floor(detected_heights, model_heights, scale_hint=scale_hint)
    print(f"Shelf-height match: {fit['matched']}/{fit['detected_count']} detected shelves matched "
          f"{fit['model_count']} model shelves, RMS {fit['rms_m'] * 100:.1f} cm, "
          f"worst {fit['max_error_m'] * 100:.1f} cm")
    print(f"  refined scale {fit['scale']:.6f} m/unit "
          f"({(fit['scale_drift_vs_hint'] - 1) * 100:+.1f}% vs the {args.anchor} seed)")
    floor_height = -fit["offset"] / fit["scale"]

    # 5. Resolve the 180 degree flip by which option seats more products.
    scored = []
    for flip in ([False, True] if args.flip == "auto" else [args.flip == "on"]):
        frame = build_frame(axes, up, scale=fit["scale"], floor_height=floor_height, flip=flip)
        report = score_alignment(frame.to_model(points_colmap), gondola,
                                 tolerance=args.tolerance, max_clearance=args.max_clearance)
        scored.append((frame, report, flip))
        print(f"  flip={'on ' if flip else 'off'}: {report['snapped_to_a_shelf']}/{report['product_count']} "
              f"products seated, {report['inside_fraction'] * 100:.0f}% inside the fixture")
    frame, report, flip = max(
        scored, key=lambda item: (item[1]["snapped_to_a_shelf"], item[1]["inside_fraction"])
    )
    if len(scored) > 1:
        margin = abs(scored[0][1]["snapped_to_a_shelf"] - scored[1][1]["snapped_to_a_shelf"])
        if margin <= max(2, int(0.05 * len(points_colmap))):
            print("WARNING: both flips score about the same, so which long face is Side A is not "
                  "determined by the geometry. Pass --flip on/off after checking one product "
                  "against a reference photo.")

    points_model = frame.to_model(points_colmap)
    print()
    print(f"Chose flip={'on' if flip else 'off'}")
    print(f"Products in model frame: X {points_model[:, 0].min():+.3f}..{points_model[:, 0].max():+.3f}  "
          f"Y {points_model[:, 1].min():+.3f}..{points_model[:, 1].max():+.3f}  "
          f"Z {points_model[:, 2].min():+.3f}..{points_model[:, 2].max():+.3f}")
    print(f"Fixture is X +/-{0.5 * float(layout.get('total_bay_run_m', 0)):.3f}  "
          f"Y +/-{0.5 * float(layout.get('gondola_depth_m', 0)):.3f}  "
          f"Z 0..{float(layout.get('gondola_height_m', 0)):.3f}")
    print(f"Seated on a shelf: {report['snapped_to_a_shelf']}/{report['product_count']} "
          f"({report['snapped_fraction'] * 100:.0f}%)")
    if report["median_clearance_m"] is not None:
        print(f"Median clearance above the assigned board: "
              f"{report['median_clearance_m'] * 100:.1f} cm "
              f"(a product centre should sit ~5-15 cm above the board it rests on)")

    assignments = report.pop("assignments")
    placed_products = []
    for record, point, assignment in zip(product_records, points_model, assignments):
        raw = record.get("p3d") or record.get("location")
        placed_products.append({
            "id": record.get("id") or record.get("detection_id") or record.get("object"),
            "product_name": (
                record.get("product_name") or record.get("display_label") or record.get("label")
            ),
            "position": [float(value) for value in point],
            "position_colmap": [float(value) for value in raw] if raw else None,
            "confidence": record.get("confidence"),
            "shelf_id": assignment["shelf_id"] if assignment else None,
            "side_label": assignment["side_label"] if assignment else None,
            "shelf_local": assignment["shelf_local"] if assignment else None,
            "clearance_m": assignment["clearance_m"] if assignment else None,
            "seated": assignment is not None,
        })

    alignment_block = {
        "source_results": str(args.results),
        "method": "shelf-plane frame + 1-D shelf-height ICP",
        "transform": frame.as_json(),
        "scale_seed": {
            "anchor": args.anchor,
            "model_m": anchor_model,
            "detected_colmap": float(anchor_detected),
            "scale": float(scale_hint),
        },
        "shelf_height_fit": fit,
        "fixture_axes": {
            "run_span_colmap": axes["run_span"],
            "depth_span_colmap": axes["depth_span"],
            "run_span_m": axes["run_span"] * fit["scale"],
            "depth_span_m": axes["depth_span"] * fit["scale"],
            "anisotropy": axes["anisotropy"],
            "source": "shelf_cluster_points" if shelf_inlier_points(shelves) is not None else "product_positions",
            "flip_applied": bool(flip),
        },
        "quality": report,
    }

    if args.output:
        output = dict(gondola)
        output["products"] = placed_products
        output["product_alignment"] = alignment_block
        for shelf in output.get("shelves", []):
            shelf["product_count"] = sum(
                1 for product in placed_products if product["shelf_id"] == shelf["id"]
            )
        Path(args.output).write_text(json.dumps(output, indent=2), encoding="utf-8")
        print(f"\nALIGNED_GONDOLA_WRITTEN: {args.output}")
    if args.report:
        Path(args.report).write_text(json.dumps(alignment_block, indent=2), encoding="utf-8")
        print(f"ALIGNMENT_REPORT_WRITTEN: {args.report}")


if __name__ == "__main__":
    main()
