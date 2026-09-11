"""
Round-trip check for align_products_to_gondola.py.

Takes a gondola JSON, invents a COLMAP frame for it (random rotation, random
origin, random scale), writes a synthetic results.json in that frame that mimics
what digital-twin-shelf-mapper's detect_shelf_planes / project_detections_to_shelves
actually emit -- including the two conventions that make this hard:

  * shelf["axes"] comes from _plane_basis(), an arbitrary in-plane rotation that
    has nothing to do with the fixture's long axis
  * shelf["bounds"] are absolute projections onto that basis, not relative to
    the shelf centroid

then checks the aligner puts the products back where they started.

    python3 test_alignment_roundtrip.py --gondola ../output/gondola_from_aws_scan.json
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np


def random_rotation(rng: np.random.Generator) -> np.ndarray:
    matrix, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    if np.linalg.det(matrix) < 0:
        matrix[:, 0] *= -1
    return matrix


def plane_basis(up: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Exactly geometry._plane_basis: an arbitrary rotation within the plane."""
    helper = np.array([1.0, 0.0, 0.0])
    if abs(float(helper @ up)) > 0.9:
        helper = np.array([0.0, 1.0, 0.0])
    first = np.cross(up, helper)
    first /= np.linalg.norm(first)
    second = np.cross(up, first)
    return first, second / np.linalg.norm(second)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gondola", required=True)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--products-per-shelf", type=int, default=12)
    parser.add_argument("--noise-m", type=float, default=0.01,
                        help="Gaussian noise added to every synthetic 3D point (meters)")
    parser.add_argument("--tolerance-cm", type=float, default=5.0,
                        help="Pass/fail threshold on the median recovery error")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    gondola = json.loads(Path(args.gondola).read_text(encoding="utf-8"))
    shelves_model = [s for s in gondola["shelves"] if s.get("side_label") in ("Side A", "Side C")]
    if not shelves_model:
        raise SystemExit("The gondola JSON has no Side A / Side C shelves to sample.")

    # The ground-truth COLMAP frame: model -> colmap is q = R.T @ (p / s) + t.
    rotation = random_rotation(rng)              # rows = model axes in colmap frame
    scale = float(rng.uniform(0.5, 2.0))         # meters per colmap unit
    origin = rng.normal(scale=5.0, size=3)       # colmap-frame position of model (0,0,0)

    def to_colmap(points_model: np.ndarray) -> np.ndarray:
        points_model = np.atleast_2d(points_model)
        return (points_model / scale) @ rotation + origin

    up_colmap = rotation[2]                      # model +Z expressed in the colmap frame
    basis_u, basis_v = plane_basis(up_colmap)    # arbitrary in-plane basis, as the real code does

    truth_points: list[np.ndarray] = []
    products: list[dict] = []
    shelves_json: list[dict] = []

    for index, shelf in enumerate(shelves_model, start=1):
        centroid = np.array(shelf["centroid"], dtype=float)
        bounds_u = shelf["bounds"]["u"]
        bounds_v = shelf["bounds"]["v"]
        top_z = float(shelf["mesh"]["volume"]["n"][1])

        # Products sitting on the board, in model coords.
        local_u = rng.uniform(bounds_u[0] * 0.9, bounds_u[1] * 0.9, args.products_per_shelf)
        local_v = rng.uniform(bounds_v[0] * 0.6, bounds_v[1] * 0.6, args.products_per_shelf)
        on_shelf = np.column_stack([
            centroid[0] + local_u,
            centroid[1] + local_v,
            np.full(args.products_per_shelf, top_z + 0.11),   # product centre above the board
        ])
        truth_points.append(on_shelf)
        for point in to_colmap(on_shelf) + rng.normal(scale=args.noise_m / scale, size=(len(on_shelf), 3)):
            products.append({
                "id": f"det-{len(products):04d}",
                "p3d": [float(v) for v in point],
                "product_name": "Synthetic Product",
                "confidence": 0.9,
            })

        # The shelf plane as detect_shelf_planes would report it: a slab of
        # inlier points on the board, then bounds/centroid/height in the
        # arbitrary plane basis.
        board_u = rng.uniform(bounds_u[0], bounds_u[1], 400)
        board_v = rng.uniform(bounds_v[0], bounds_v[1], 400)
        board = np.column_stack([
            centroid[0] + board_u,
            centroid[1] + board_v,
            np.full(400, top_z),
        ])
        board_colmap = to_colmap(board) + rng.normal(scale=args.noise_m / scale, size=(400, 3))
        coords_u = board_colmap @ basis_u
        coords_v = board_colmap @ basis_v
        height = float(np.mean(board_colmap @ up_colmap))
        centre_u = float(0.5 * (coords_u.min() + coords_u.max()))
        centre_v = float(0.5 * (coords_v.min() + coords_v.max()))
        shelves_json.append({
            "id": f"shelf-{index}",
            "height": height,
            "centroid": (basis_u * centre_u + basis_v * centre_v + up_colmap * height).tolist(),
            "normal": (up_colmap * (1 if index % 2 else -1)).tolist(),  # RANSAC sign is arbitrary
            "axes": {"u": basis_u.tolist(), "v": basis_v.tolist()},
            "bounds": {
                "u": [float(coords_u.min()), float(coords_u.max())],
                "v": [float(coords_v.min()), float(coords_v.max())],
            },
            "extents": {"width": 1.0, "depth": 0.5, "thickness": 0.045},
            "mesh": {"volume": {"u": [float(coords_u.min()), float(coords_u.max())],
                                "v": [float(coords_v.min()), float(coords_v.max())],
                                "n": [height - 0.045, height]}},
            "point_count": 400,
            "_cluster_points": board_colmap.tolist(),
        })

    truth = np.concatenate(truth_points, axis=0)
    results = {"job_id": "synthetic", "shelves": shelves_json, "products": products,
               "layout": {"scale_factor": 1.0, "scale_source": "scene_units"}}

    with tempfile.TemporaryDirectory() as tmp:
        results_path = Path(tmp) / "results.json"
        output_path = Path(tmp) / "aligned.json"
        results_path.write_text(json.dumps(results), encoding="utf-8")

        print(f"Ground truth: scale={scale:.6f} m/unit, origin={np.round(origin, 3)}")
        print(f"{len(products)} products on {len(shelves_json)} shelves, "
              f"{args.noise_m * 100:.1f} cm noise\n")

        # A double-sided gondola is symmetric under a 180 degree turn about up,
        # so on synthetic data the flip genuinely cannot be recovered from
        # geometry -- only a photo or a per-side product difference settles it.
        # Score both and report the better, rather than pretending the aligner
        # got it wrong for guessing one of two indistinguishable answers.
        outcomes = {}
        for flip in ("off", "on"):
            completed = subprocess.run(
                [sys.executable, str(Path(__file__).with_name("align_products_to_gondola.py")),
                 "--results", str(results_path), "--gondola", args.gondola,
                 "--output", str(output_path), "--flip", flip],
                capture_output=True, text=True,
            )
            if completed.returncode != 0:
                print(completed.stdout)
                print(completed.stderr)
                raise SystemExit(f"aligner failed with --flip {flip}")
            if flip == "off":
                print(completed.stdout)
            aligned = json.loads(output_path.read_text(encoding="utf-8"))
            recovered = np.array([p["position"] for p in aligned["products"]], dtype=float)
            outcomes[flip] = (np.linalg.norm(recovered - truth, axis=1), aligned)

    flip = min(outcomes, key=lambda key: float(np.median(outcomes[key][0])))
    errors, aligned = outcomes[flip]
    other = float(np.median(outcomes["on" if flip == "off" else "off"][0]))
    recovered_scale = aligned["product_alignment"]["transform"]["scale_meters_per_colmap_unit"]
    seated = sum(1 for product in aligned["products"] if product["seated"])

    print("--- round trip ---")
    print(f"recovered scale {recovered_scale:.6f} vs truth {scale:.6f} "
          f"({100 * (recovered_scale / scale - 1):+.2f}%)")
    print(f"matching flip: --flip {flip}  (the other one lands {other * 100:.0f} cm out, "
          f"as a 180 degree yaw should)")
    print(f"position error: median {np.median(errors) * 100:.2f} cm  "
          f"p95 {np.percentile(errors, 95) * 100:.2f} cm  max {errors.max() * 100:.2f} cm")
    print(f"seated on a board: {seated}/{len(errors)}")

    if np.median(errors) * 100 > args.tolerance_cm:
        raise SystemExit(f"FAIL: median error {np.median(errors) * 100:.2f} cm "
                         f"exceeds {args.tolerance_cm} cm")
    print("PASS")


if __name__ == "__main__":
    main()
