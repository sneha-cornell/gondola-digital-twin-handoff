"""
Recover the COLMAP -> gondola-model frame transform that
build_gondola_from_scan.py currently computes internally and throws away.

The parametric gondola is emitted in its own canonical frame:

    +Z  up, 0.0 = floor
    +X  along the bay run (length), 0.0 = centre of the run
    +Y  depth, 0.0 = the shared back panel between Side A and Side C
    units: meters

A COLMAP sparse model is in an arbitrary frame: unknown scale, unknown
orientation, origin wherever the first registered camera happened to put it.
Any 3D product position that came out of the same reconstruction therefore
needs a 7-DoF similarity transform (scale, rotation, translation) before it
can be compared against the gondola model:

    p_model = scale * R @ (p_colmap - origin_colmap)

Every term is already available from the same fit the mesh builder does:

    R rows      the fitted object axes -- long axis, depth axis, up vector
    origin      centre of the fitted top-down rectangle, dropped to floor level
    scale       meters per COLMAP unit, from the caller's one real measurement

This module computes that transform, so nothing downstream has to guess it
from bounding-box min/max heuristics.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
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
)


# ---------------------------------------------------------------------------
# The transform
# ---------------------------------------------------------------------------

@dataclass
class ModelFrame:
    """Similarity transform from a COLMAP world frame to the gondola model frame."""

    scale: float                 # meters per COLMAP unit
    rotation: np.ndarray         # 3x3, rows = model X/Y/Z axes expressed in COLMAP frame
    origin: np.ndarray           # COLMAP-frame point that maps to model (0, 0, 0)
    raw: dict[str, Any] = field(default_factory=dict)  # diagnostics, raw COLMAP units

    def to_model(self, points: np.ndarray) -> np.ndarray:
        """COLMAP coords -> gondola model coords (meters)."""
        points = np.atleast_2d(np.asarray(points, dtype=float))
        return self.scale * ((points - self.origin) @ self.rotation.T)

    def to_colmap(self, points: np.ndarray) -> np.ndarray:
        """Gondola model coords (meters) -> COLMAP coords."""
        points = np.atleast_2d(np.asarray(points, dtype=float))
        return (points / self.scale) @ self.rotation + self.origin

    def matrix(self) -> np.ndarray:
        """The 4x4 homogeneous form of to_model (row-vector convention: p @ M.T)."""
        matrix = np.eye(4)
        matrix[:3, :3] = self.scale * self.rotation
        matrix[:3, 3] = -self.scale * (self.rotation @ self.origin)
        return matrix

    def as_json(self) -> dict[str, Any]:
        return {
            "convention": "p_model = scale * R @ (p_colmap - origin_colmap)",
            "frame": {
                "up": "+Z", "floor_z": 0.0,
                "bay_run": "+X", "depth": "+Y",
                "origin": "floor level, centre of the bay run, on the back-panel plane",
                "units": "meters",
            },
            "scale_meters_per_colmap_unit": float(self.scale),
            "rotation_model_from_colmap": [[float(v) for v in row] for row in self.rotation],
            "origin_colmap": [float(v) for v in self.origin],
            "matrix_4x4_model_from_colmap": [[float(v) for v in row] for row in self.matrix()],
            "raw_colmap": self.raw,
        }


def fit_model_frame(
    colmap_dir: Path,
    *,
    anchor_dimension: str,
    anchor_meters: float,
    proximity_percentile: float = 70.0,
    min_track_len: int = 3,
    diversity_percentile: float = 40.0,
    height_density_frac: float = 0.15,
    shelf_back_frac: float = 0.5,
    shelf_prominence: float = 0.25,
    shelf_min_gap: float = 0.06,
    flip_long_axis: bool = False,
    isolated_ply_output: str | None = None,
    verbose: bool = True,
) -> ModelFrame:
    """Fit the gondola model frame to a COLMAP sparse reconstruction.

    Uses the same subject-isolation filters as build_gondola_from_scan.py, so
    the frame it returns is the frame that script's mesh is implicitly built in.
    """
    from build_gondola_from_scan import (
        _density_mask,
        _fit_rectangle,
        _main_cluster_mask,
        _oriented_bbox_fallback,
        _ransac_lines,
    )

    def log(message: str) -> None:
        if verbose:
            print(message)

    images = parse_images_txt(colmap_dir / "images.txt")
    pts, _cols, tracks = parse_points3d_txt(colmap_dir / "points3D.txt")
    log(f"Loaded {len(images)} camera poses, {len(pts)} 3D points")

    cam_centers, up, alignment = camera_centers_and_up(images)
    log(f"Up-vector alignment across cameras: {alignment:.3f} (1.0 = perfect)")
    u0, v0 = build_basis(up)
    cam_array = np.array(list(cam_centers.values()))
    centroid = cam_array.mean(axis=0)

    prox_mask, _, _ = proximity_mask(pts, tracks, cam_centers, proximity_percentile)
    div_mask, _, _ = viewpoint_diversity_mask(pts, tracks, cam_centers, min_track_len, diversity_percentile)
    pre_height = prox_mask & div_mask
    if not pre_height.any():
        raise SystemExit("No points survived the proximity + viewpoint-diversity filters.")
    height_mask, band_lo, band_hi = height_band_mask(pts[pre_height], up, height_density_frac)
    isolated = pts[pre_height][height_mask]
    isolated_cols = _cols[pre_height][height_mask]
    log(f"Isolated {len(isolated)}/{len(pts)} points; height band [{band_lo:.3f}, {band_hi:.3f}]")
    if len(isolated) < 30:
        raise SystemExit("Too few isolated points to fit a frame. Loosen the filter percentiles.")

    if isolated_ply_output:
        from isolate_subject import write_ply

        write_ply(Path(isolated_ply_output), isolated, isolated_cols)
        log(f"Wrote isolated point cloud: {isolated_ply_output}")

    rel = isolated - centroid
    xy = np.column_stack([rel @ u0, rel @ v0])
    height = rel @ up

    density = _density_mask(np.column_stack([xy, height]), keep_percentile=70.0)
    keep = np.zeros(len(xy), dtype=bool)
    keep[np.nonzero(density)[0]] = True
    xy_f = xy[keep]
    cluster = _main_cluster_mask(xy_f)
    # Index back into `isolated` so the height of every surviving point is known too.
    surviving = np.nonzero(keep)[0][cluster]
    xy_f = xy[surviving]
    height_f = height[surviving]
    if len(xy_f) < 20:
        raise SystemExit("Too few points survived density/cluster filtering to fit a frame.")

    lines, _residuals, supports = _ransac_lines(xy_f, n_lines=4)
    rectangle = _fit_rectangle(lines, xy_f)
    if rectangle is not None:
        fit_confidence = "full_rectangle"
        center_2d = np.asarray(rectangle["center_2d"], dtype=float)
    else:
        fit_confidence = "partial_coverage_fallback"
        rectangle = _oriented_bbox_fallback(xy_f, lines, supports)
        center_2d = None  # derived below from robust spans along the fitted axes
    log(f"Rectangle fit: {fit_confidence}  length={rectangle['length']:.3f}  "
        f"width={rectangle['width']:.3f}  angle={rectangle['angle_deg']:.1f} deg (raw units)")

    # Model axes, in the COLMAP frame.
    angle = np.radians(rectangle["angle_deg"])
    x_axis = np.cos(angle) * u0 + np.sin(angle) * v0     # along the bay run
    if flip_long_axis:
        x_axis = -x_axis
    z_axis = up / np.linalg.norm(up)
    x_axis = x_axis - (x_axis @ z_axis) * z_axis          # re-orthogonalise against up
    x_axis /= np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)                     # right-handed depth axis
    rotation = np.stack([x_axis, y_axis, z_axis])

    # Origin: centre of the object in X/Y, floor level in Z.
    along = rel @ x_axis
    across = rel @ y_axis
    if center_2d is not None:
        # Rectangle corners live in the (u0, v0) basis -- project its centre onto
        # the model axes, both of which are also in that plane.
        center_3d = centroid + center_2d[0] * u0 + center_2d[1] * v0
        center_along = (center_3d - centroid) @ x_axis
        center_across = (center_3d - centroid) @ y_axis
    else:
        center_along = float(np.mean(np.percentile(along, [1, 99])))
        center_across = float(np.mean(np.percentile(across, [1, 99])))
    # The height span and the floor both come from every isolated point, which
    # is what build_gondola_from_scan.py has always reported -- so routing that
    # script through this module doesn't silently move the mesh's dimensions.
    # The main-cluster-only span is kept as a diagnostic: a big disagreement
    # means background points are still dragging the floor down, and the scale
    # anchor is partly measuring them rather than the fixture.
    floor_lo, ceil_hi = np.percentile(height, [1, 99])
    raw_height = float(ceil_hi - floor_lo)
    tight_lo, tight_hi = np.percentile(height_f, [1, 99])
    origin = centroid + center_along * x_axis + center_across * y_axis + floor_lo * z_axis

    raw_value = {
        "length": rectangle["length"],
        "width": rectangle["width"],
        "height": raw_height,
    }[anchor_dimension]
    if raw_value <= 1e-9:
        raise SystemExit(f"Fitted {anchor_dimension} is ~0 in raw units; cannot compute a scale.")
    scale = anchor_meters / raw_value

    long_face_peaks, _ = detect_shelf_heights(
        rel, up, y_axis, back_frac=shelf_back_frac,
        prominence_frac=shelf_prominence, min_gap_frac=shelf_min_gap,
    )
    end_cap_peaks, _ = detect_shelf_heights(
        rel, up, x_axis, back_frac=shelf_back_frac,
        prominence_frac=shelf_prominence, min_gap_frac=shelf_min_gap,
    )
    # detect_shelf_heights measures height relative to `centroid`; the model
    # frame measures it from the floor. Convert, and scale to meters.
    def to_model_height(peaks: list[float]) -> list[float]:
        return sorted(float((peak - floor_lo) * scale) for peak in peaks)

    frame = ModelFrame(
        scale=scale,
        rotation=rotation,
        origin=origin,
        raw={
            "length": float(rectangle["length"]),
            "width": float(rectangle["width"]),
            "height": raw_height,
            "height_main_cluster_only": float(tight_hi - tight_lo),
            "floor_disagreement": float(tight_lo - floor_lo),
            "floor_height_along_up": float(floor_lo),
            "fit_confidence": fit_confidence,
            "up_vector_alignment": float(alignment),
            "camera_centroid": [float(v) for v in centroid],
            "isolated_point_count": int(len(isolated)),
            "shelf_bands_m": {
                "side_ac": to_model_height(long_face_peaks),
                "side_bd": to_model_height(end_cap_peaks),
            },
            "anchor": {
                "dimension": anchor_dimension,
                "meters": float(anchor_meters),
            },
        },
    )

    if verbose:
        extents = frame.to_model(isolated)
        print("Isolated cloud in model frame (meters):")
        print(f"  X (bay run) {extents[:, 0].min():+.3f} .. {extents[:, 0].max():+.3f}")
        print(f"  Y (depth)   {extents[:, 1].min():+.3f} .. {extents[:, 1].max():+.3f}")
        print(f"  Z (height)  {extents[:, 2].min():+.3f} .. {extents[:, 2].max():+.3f}")
        cams = frame.to_model(cam_array)
        front = int(np.sum(cams[:, 1] < 0))
        print(f"Camera coverage: {front} on -Y (Side A), {len(cams) - front} on +Y (Side C)")
        for label, peaks in frame.raw["shelf_bands_m"].items():
            print(f"  detected shelf bands {label}: " + ", ".join(f"{p:.3f}" for p in peaks))

    return frame


# ---------------------------------------------------------------------------
# Cross-model alignment (two different reconstructions of the same fixture)
# ---------------------------------------------------------------------------

def umeyama(source: np.ndarray, target: np.ndarray, *, with_scale: bool = True):
    """Least-squares similarity transform mapping source -> target.

    Returns (scale, R, t) such that  target ~= scale * R @ source + t.
    Use when the products and the gondola came out of *different* COLMAP
    reconstructions and you have >= 3 corresponding points (shelf-plane
    intersections, fixture corners, a measured tape target).
    """
    source = np.asarray(source, dtype=float)
    target = np.asarray(target, dtype=float)
    if source.shape != target.shape or len(source) < 3:
        raise ValueError("need matching source/target arrays with at least 3 points")

    mu_source, mu_target = source.mean(axis=0), target.mean(axis=0)
    a, b = source - mu_source, target - mu_target
    covariance = (b.T @ a) / len(source)
    u, singular, vt = np.linalg.svd(covariance)
    correction = np.eye(3)
    if np.linalg.det(u) * np.linalg.det(vt) < 0:
        correction[2, 2] = -1.0
    rotation = u @ correction @ vt
    scale = float(np.trace(np.diag(singular) @ correction) / (a ** 2).sum() * len(source)) if with_scale else 1.0
    translation = mu_target - scale * rotation @ mu_source
    return scale, rotation, translation


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compute (and save) the COLMAP -> gondola-model frame transform."
    )
    parser.add_argument("--colmap-txt", required=True, help="COLMAP TXT export dir")
    parser.add_argument("--anchor-dimension", required=True, choices=["length", "width", "height"])
    parser.add_argument("--anchor-meters", type=float, required=True)
    parser.add_argument("--flip-long-axis", action="store_true",
                        help="Rotate the model frame 180 deg about up (swaps Side A/C and Side B/D). "
                             "Geometry alone cannot tell which long face is the front.")
    parser.add_argument("--proximity-percentile", type=float, default=70.0)
    parser.add_argument("--min-track-len", type=int, default=3)
    parser.add_argument("--diversity-percentile", type=float, default=40.0)
    parser.add_argument("--height-density-frac", type=float, default=0.15)
    parser.add_argument("--shelf-back-frac", type=float, default=0.5)
    parser.add_argument("--shelf-prominence", type=float, default=0.25)
    parser.add_argument("--shelf-min-gap", type=float, default=0.06)
    parser.add_argument("--output", default=None, help="Write the transform as JSON to this path")
    return parser


def main() -> None:
    args = build_parser().parse_args()
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
        flip_long_axis=args.flip_long_axis,
    )
    payload = frame.as_json()
    print()
    print(json.dumps(payload, indent=2))
    if args.output:
        Path(args.output).write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"\nFRAME_TRANSFORM_WRITTEN: {args.output}")


if __name__ == "__main__":
    main()
