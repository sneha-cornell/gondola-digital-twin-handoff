"""
Build a synthetic digital-twin-shelf-mapper results.json from a gondola JSON.

Used by test_alignment_roundtrip.py, and standalone to produce demo input when
no real job output is at hand. The point is to reproduce what the product
pipeline actually emits, including the two conventions that make alignment
awkward:

  * shelf["axes"] comes from geometry._plane_basis() -- an arbitrary rotation
    inside the shelf plane, unrelated to the fixture's long axis
  * shelf["bounds"] are absolute projections onto that basis, whereas the
    gondola generator writes bounds relative to the shelf centroid
  * shelf["normal"] carries RANSAC's arbitrary sign, which flips per plane

Products are placed on the boards in model coordinates, then pushed out into a
randomly chosen COLMAP frame, so the true answer is known exactly.

    python3 synthetic_scan.py --gondola ../output/gondola_from_aws_scan.json \\
        --output /tmp/results.json --truth /tmp/truth.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

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


# A few plausible names so demo output reads like a planogram rather than
# "Product 17". Nothing here depends on the specific strings.
DEMO_NAMES = [
    "Dot's Homestyle Pretzels Honey Mustard", "Kettle Brand Pepperoncini",
    "Lay's Classic", "Doritos Nacho Cheese", "Cheez-It Original",
    "Pringles Sour Cream & Onion", "Ritz Crackers", "Oreo Original",
    "Goldfish Cheddar", "Chex Mix Traditional", "Takis Fuego",
    "Sun Chips Harvest Cheddar",
]


def build_synthetic_results(
    gondola: dict,
    *,
    seed: int = 7,
    products_per_shelf: int = 12,
    noise_m: float = 0.01,
    sides: tuple[str, ...] = ("Side A", "Side C"),
    vary_counts: bool = False,
    stray_fraction: float = 0.0,
) -> dict[str, Any]:
    """Synthesise a results.json plus the ground truth that produced it.

    `vary_counts` gives each shelf a different facing count (a real shelf is
    never uniformly full), and `stray_fraction` puts some products nowhere near
    a board -- both make the quality numbers mean something, since a demo where
    every product is perfectly seated cannot show a failure being caught.
    """
    rng = np.random.default_rng(seed)
    shelves_model = [s for s in gondola["shelves"] if s.get("side_label") in sides]
    if not shelves_model:
        raise SystemExit(f"The gondola JSON has no {'/'.join(sides)} shelves to sample.")

    rotation = random_rotation(rng)          # rows = model axes in the colmap frame
    scale = float(rng.uniform(0.5, 2.0))     # meters per colmap unit
    origin = rng.normal(scale=5.0, size=3)   # colmap-frame position of model (0,0,0)

    def to_colmap(points_model: np.ndarray) -> np.ndarray:
        points_model = np.atleast_2d(points_model)
        return (points_model / scale) @ rotation + origin

    up_colmap = rotation[2]
    basis_u, basis_v = plane_basis(up_colmap)

    truth: list[np.ndarray] = []
    products: list[dict] = []
    shelves_json: list[dict] = []

    for index, shelf in enumerate(shelves_model, start=1):
        centroid = np.array(shelf["centroid"], dtype=float)
        bounds_u = shelf["bounds"]["u"]
        bounds_v = shelf["bounds"]["v"]
        top = float(shelf["mesh"]["volume"]["n"][1])

        count = products_per_shelf
        if vary_counts:
            count = int(rng.integers(max(2, products_per_shelf // 2), products_per_shelf + 6))

        # Facings spread along the board, product centres a realistic 6-14 cm up.
        local_u = np.sort(rng.uniform(bounds_u[0] * 0.92, bounds_u[1] * 0.92, count))
        local_v = rng.uniform(bounds_v[0] * 0.5, bounds_v[1] * 0.5, count)
        heights = rng.uniform(0.06, 0.14, count)
        on_shelf = np.column_stack([
            centroid[0] + local_u,
            centroid[1] + local_v,
            top + heights,
        ])
        truth.append(on_shelf)
        noisy = to_colmap(on_shelf) + rng.normal(scale=noise_m / scale, size=(count, 3))
        for offset, point in enumerate(noisy):
            products.append({
                "id": f"det-{len(products):04d}",
                "p3d": [float(value) for value in point],
                "product_name": DEMO_NAMES[(index * 7 + offset) % len(DEMO_NAMES)],
                "confidence": float(np.round(rng.uniform(0.45, 0.95), 3)),
                "shelf_id": f"shelf-{index}",
            })

        # The shelf plane as detect_shelf_planes reports it.
        board = np.column_stack([
            centroid[0] + rng.uniform(bounds_u[0], bounds_u[1], 400),
            centroid[1] + rng.uniform(bounds_v[0], bounds_v[1], 400),
            np.full(400, top),
        ])
        board_colmap = to_colmap(board) + rng.normal(scale=noise_m / scale, size=(400, 3))
        coords_u = board_colmap @ basis_u
        coords_v = board_colmap @ basis_v
        height = float(np.mean(board_colmap @ up_colmap))
        centre_u = float(0.5 * (coords_u.min() + coords_u.max()))
        centre_v = float(0.5 * (coords_v.min() + coords_v.max()))
        shelves_json.append({
            "id": f"shelf-{index}",
            "height": height,
            "centroid": (basis_u * centre_u + basis_v * centre_v + up_colmap * height).tolist(),
            "normal": (up_colmap * (1 if index % 2 else -1)).tolist(),   # RANSAC sign is arbitrary
            "axes": {"u": basis_u.tolist(), "v": basis_v.tolist()},
            "bounds": {"u": [float(coords_u.min()), float(coords_u.max())],
                       "v": [float(coords_v.min()), float(coords_v.max())]},
            "extents": {"width": 1.0, "depth": 0.5, "thickness": 0.045},
            "mesh": {"volume": {"u": [float(coords_u.min()), float(coords_u.max())],
                                "v": [float(coords_v.min()), float(coords_v.max())],
                                "n": [height - 0.045, height]}},
            "point_count": 400,
            "_cluster_points": board_colmap.tolist(),
        })

    truth_points = np.concatenate(truth, axis=0)

    # Strays: detections whose 3D position landed off the fixture entirely
    # (a mis-triangulated ray, a product on a neighbouring unit). They should
    # come out of the aligner as unseated, not silently snapped to a board.
    stray_count = int(round(stray_fraction * len(products)))
    if stray_count:
        strays_model = np.column_stack([
            rng.uniform(-6.0, 6.0, stray_count),
            rng.uniform(-3.0, 3.0, stray_count),
            rng.uniform(0.0, 2.4, stray_count),
        ])
        for offset, point in enumerate(to_colmap(strays_model)):
            products.append({
                "id": f"stray-{offset:04d}",
                "p3d": [float(value) for value in point],
                "product_name": "Unknown",
                "confidence": 0.2,
            })
        truth_points = np.concatenate([truth_points, strays_model], axis=0)

    results = {
        "job_id": "synthetic-demo",
        "summary": {"shelf_count": len(shelves_json), "product_count": len(products)},
        "layout": {"scale_factor": 1.0, "scale_source": "scene_units"},
        "shelves": shelves_json,
        "products": products,
    }
    ground_truth = {
        "scale_meters_per_colmap_unit": scale,
        "rotation_model_from_colmap": rotation.tolist(),
        "origin_colmap": origin.tolist(),
        "positions_model": truth_points.tolist(),
        "stray_count": stray_count,
    }
    return {"results": results, "truth": ground_truth}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--gondola", required=True)
    parser.add_argument("--output", required=True, help="Where to write the synthetic results.json")
    parser.add_argument("--truth", default=None, help="Where to write the ground-truth JSON")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--products-per-shelf", type=int, default=12)
    parser.add_argument("--noise-m", type=float, default=0.01)
    parser.add_argument("--vary-counts", action="store_true")
    parser.add_argument("--stray-fraction", type=float, default=0.0)
    args = parser.parse_args()

    gondola = json.loads(Path(args.gondola).read_text(encoding="utf-8"))
    built = build_synthetic_results(
        gondola,
        seed=args.seed,
        products_per_shelf=args.products_per_shelf,
        noise_m=args.noise_m,
        vary_counts=args.vary_counts,
        stray_fraction=args.stray_fraction,
    )
    Path(args.output).write_text(json.dumps(built["results"]), encoding="utf-8")
    print(f"SYNTHETIC_RESULTS_WRITTEN: {args.output} "
          f"({len(built['results']['products'])} products, "
          f"{len(built['results']['shelves'])} shelf planes)")
    print(f"  ground-truth scale {built['truth']['scale_meters_per_colmap_unit']:.6f} m/unit")
    if args.truth:
        Path(args.truth).write_text(json.dumps(built["truth"]), encoding="utf-8")
        print(f"GROUND_TRUTH_WRITTEN: {args.truth}")


if __name__ == "__main__":
    main()
