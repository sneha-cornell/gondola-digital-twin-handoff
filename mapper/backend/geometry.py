from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from math import cos, radians
from typing import Iterable

import numpy as np
from scipy.spatial import Delaunay, QhullError, cKDTree
from sklearn.cluster import DBSCAN

from colmap_text import Camera, ImagePose, Reconstruction

SHELF_NORMAL_MIN_ALIGNMENT: float = 0.64  # cos(50°) — shelf normal must align ≥ 50° with world-up
MIN_SHELF_AREA_M2: float = 0.20           # drop clusters smaller than 20 cm × 10 cm
OBSERVED_SHELF_MATCH_MIN_POINTS: int = 8
OBSERVED_SHELF_MATCH_MAX_HEIGHT_DELTA_M: float = 0.22
OBSERVED_SHELF_MATCH_MAX_OVERFLOW_M: float = 0.28


@dataclass(slots=True)
class PlaneCandidate:
    normal: np.ndarray
    center: np.ndarray
    inliers: np.ndarray


def _normalize(vector: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vector)
    if norm == 0:
        return vector
    return vector / norm


def _fit_plane(points: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    if len(points) < 3:
        return None
    p0, p1, p2 = points[:3]
    normal = np.cross(p1 - p0, p2 - p0)
    normal = _normalize(normal)
    if np.linalg.norm(normal) == 0:
        return None
    return normal, p0


def _extract_plane_candidates(
    points: np.ndarray,
    max_planes: int = 12,
    iterations: int = 600,
    distance_threshold: float = 0.04,
    min_inliers: int | None = None,
    hint_up: np.ndarray | None = None,
    up_alignment_min: float = SHELF_NORMAL_MIN_ALIGNMENT * 0.72,
) -> list[PlaneCandidate]:
    # Scale minimum inlier count to point-cloud density: 3% of total points,
    # clamped so sparse scans (< 1 000 pts) still find shelves and huge clouds
    # don't demand thousands of perfectly coplanar points per shelf.
    effective_min_inliers = min_inliers if min_inliers is not None else int(
        np.clip(len(points) * 0.03, 30, 250)
    )
    candidates: list[PlaneCandidate] = []
    remaining = points.copy()
    rng = np.random.default_rng(42)
    # Normalise hint_up once for cheap per-iteration dot products.
    up_unit: np.ndarray | None = _normalize(hint_up) if hint_up is not None else None

    while len(remaining) >= effective_min_inliers and len(candidates) < max_planes:
        best_inliers: np.ndarray | None = None
        best_normal: np.ndarray | None = None
        best_point: np.ndarray | None = None

        for _ in range(iterations):
            sample_indices = rng.choice(len(remaining), size=3, replace=False)
            fitted = _fit_plane(remaining[sample_indices])
            if fitted is None:
                continue
            normal, plane_point = fitted
            # When we know the world-up direction, reject planes whose normal
            # is too far from horizontal (i.e. walls and floors).  Using a
            # softer threshold here than the post-selection filter so that
            # slightly-tilted shelves still make it through RANSAC.
            if up_unit is not None and abs(float(np.dot(normal, up_unit))) < up_alignment_min:
                continue
            distances = np.abs((remaining - plane_point) @ normal)
            inlier_indices = np.where(distances <= distance_threshold)[0]
            if best_inliers is None or len(inlier_indices) > len(best_inliers):
                best_inliers = inlier_indices
                best_normal = normal
                best_point = plane_point

        if best_inliers is None or len(best_inliers) < effective_min_inliers or best_normal is None or best_point is None:
            break

        inlier_points = remaining[best_inliers]
        center = inlier_points.mean(axis=0)
        normal = _refine_plane_normal(inlier_points, best_normal)
        candidates.append(PlaneCandidate(normal=normal, center=center, inliers=inlier_points))

        mask = np.ones(len(remaining), dtype=bool)
        mask[best_inliers] = False
        remaining = remaining[mask]

    return candidates


def _refine_plane_normal(points: np.ndarray, seed_normal: np.ndarray) -> np.ndarray:
    centered = points - points.mean(axis=0)
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    normal = _normalize(vh[-1])
    if normal @ seed_normal < 0:
        normal *= -1
    return normal


def _plane_basis(normal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    helper = np.array([1.0, 0.0, 0.0], dtype=float)
    if abs(np.dot(helper, normal)) > 0.9:
        helper = np.array([0.0, 1.0, 0.0], dtype=float)
    u = _normalize(np.cross(normal, helper))
    v = _normalize(np.cross(normal, u))
    return u, v


def _compose_world_point(
    axis_u: np.ndarray,
    axis_v: np.ndarray,
    axis_n: np.ndarray,
    coord_u: float,
    coord_v: float,
    coord_n: float,
) -> np.ndarray:
    return axis_u * coord_u + axis_v * coord_v + axis_n * coord_n


def _distance_to_segment(point: np.ndarray, start: np.ndarray, end: np.ndarray) -> float:
    segment = end - start
    length_squared = float(np.dot(segment, segment))
    if length_squared == 0:
        return float(np.linalg.norm(point - start))
    t = float(np.clip(np.dot(point - start, segment) / length_squared, 0.0, 1.0))
    projection = start + segment * t
    return float(np.linalg.norm(point - projection))


def _downsample_surface_points(coords_u: np.ndarray, coords_v: np.ndarray) -> tuple[np.ndarray, float]:
    min_u, max_u = float(coords_u.min()), float(coords_u.max())
    min_v, max_v = float(coords_v.min()), float(coords_v.max())
    area = max((max_u - min_u) * (max_v - min_v), 1e-4)
    target_vertices = 1200.0
    cell_size = float(np.clip(np.sqrt(area / target_vertices), 0.03, 0.12))

    buckets: dict[tuple[int, int], list[np.ndarray]] = {}
    for coord_u, coord_v in zip(coords_u, coords_v):
        key = (int(np.floor(coord_u / cell_size)), int(np.floor(coord_v / cell_size)))
        buckets.setdefault(key, []).append(np.array([coord_u, coord_v], dtype=float))

    sampled_points = [
        np.mean(bucket_points, axis=0)
        for bucket_points in buckets.values()
    ]
    return np.array(sampled_points, dtype=float), cell_size


def _build_surface_topology(sampled_uv: np.ndarray, cell_size: float) -> tuple[list[list[int]], list[list[int]]]:
    if len(sampled_uv) < 3:
        return [], []

    try:
        triangulation = Delaunay(sampled_uv)
    except QhullError:
        return [], []
    tree = cKDTree(sampled_uv)
    nearest_distances, _ = tree.query(sampled_uv, k=min(2, len(sampled_uv)))
    if len(sampled_uv) > 1:
        spacing = float(np.median(nearest_distances[:, 1]))
    else:
        spacing = cell_size

    max_edge = max(spacing * 4.5, cell_size * 3.0, 0.12)
    max_area = max(0.5 * max_edge * max_edge, 0.02)
    kept_triangles: list[list[int]] = []
    edge_counts: Counter[tuple[int, int]] = Counter()

    for simplex in triangulation.simplices:
        triangle = sampled_uv[simplex]
        edge_lengths = [
            float(np.linalg.norm(triangle[1] - triangle[0])),
            float(np.linalg.norm(triangle[2] - triangle[1])),
            float(np.linalg.norm(triangle[0] - triangle[2])),
        ]
        if max(edge_lengths) > max_edge:
            continue

        area = 0.5 * abs(np.cross(triangle[1] - triangle[0], triangle[2] - triangle[0]))
        if area > max_area or area < 1e-5:
            continue

        kept = [int(simplex[0]), int(simplex[1]), int(simplex[2])]
        kept_triangles.append(kept)
        edge_counts.update(
            [
                tuple(sorted((kept[0], kept[1]))),
                tuple(sorted((kept[1], kept[2]))),
                tuple(sorted((kept[2], kept[0]))),
            ]
        )

    boundary_edges = [
        [edge[0], edge[1]]
        for edge, count in edge_counts.items()
        if count == 1
    ]
    return kept_triangles, boundary_edges


def _point_in_triangle(point: np.ndarray, a: np.ndarray, b: np.ndarray, c: np.ndarray, tolerance: float) -> bool:
    v0 = c - a
    v1 = b - a
    v2 = point - a
    denominator = float(v0[0] * v1[1] - v1[0] * v0[1])
    if abs(denominator) < 1e-9:
        return False

    inv_denominator = 1.0 / denominator
    alpha = float((v2[0] * v1[1] - v1[0] * v2[1]) * inv_denominator)
    beta = float((v0[0] * v2[1] - v2[0] * v0[1]) * inv_denominator)
    gamma = 1.0 - alpha - beta
    return alpha >= -tolerance and beta >= -tolerance and gamma >= -tolerance


def _point_in_surface_mesh(point_uv: np.ndarray, shelf: dict) -> bool:
    top_vertices_uv = np.array(shelf["mesh"]["top_vertices_uv"], dtype=float)
    top_triangles = shelf["mesh"]["top_triangles"]
    tolerance = max(0.02, 0.5 * shelf["mesh"]["sample_step"])
    for triangle_indices in top_triangles:
        a, b, c = top_vertices_uv[triangle_indices]
        if _point_in_triangle(point_uv, a, b, c, tolerance=tolerance):
            return True
    return False


def _build_shelf_mesh(
    shelf_id: str,
    basis_u: np.ndarray,
    basis_v: np.ndarray,
    dominant_normal: np.ndarray,
    top_vertices_uv: np.ndarray,
    top_triangles: list[list[int]],
    boundary_edges: list[list[int]],
    mean_height: float,
    thickness: float,
    sample_step: float,
) -> dict:
    top_height = mean_height + thickness * 0.5
    bottom_height = mean_height - thickness * 0.5
    top_face = [
        _compose_world_point(basis_u, basis_v, dominant_normal, float(coord_u), float(coord_v), top_height).tolist()
        for coord_u, coord_v in top_vertices_uv
    ]
    bottom_face = [
        _compose_world_point(basis_u, basis_v, dominant_normal, float(coord_u), float(coord_v), bottom_height).tolist()
        for coord_u, coord_v in top_vertices_uv
    ]
    vertices = top_face + bottom_face
    top_vertex_count = len(top_vertices_uv)
    bottom_offset = top_vertex_count
    triangles = [triangle[:] for triangle in top_triangles]
    triangles.extend(
        [
            [bottom_offset + triangle[0], bottom_offset + triangle[2], bottom_offset + triangle[1]]
            for triangle in top_triangles
        ]
    )
    for start_index, end_index in boundary_edges:
        triangles.append([start_index, end_index, bottom_offset + end_index])
        triangles.append([start_index, bottom_offset + end_index, bottom_offset + start_index])

    min_u = float(top_vertices_uv[:, 0].min())
    max_u = float(top_vertices_uv[:, 0].max())
    min_v = float(top_vertices_uv[:, 1].min())
    max_v = float(top_vertices_uv[:, 1].max())

    return {
        "id": f"{shelf_id}-mesh",
        "type": "triangulated_prism",
        "vertices": vertices,
        "triangles": triangles,
        "top_face": top_face,
        "bottom_face": bottom_face,
        "top_vertices_uv": top_vertices_uv.tolist(),
        "top_triangles": top_triangles,
        "boundary_edges": boundary_edges,
        "sample_step": sample_step,
        "volume": {
            "u": [min_u, max_u],
            "v": [min_v, max_v],
            "n": [bottom_height, top_height],
        },
    }


def _build_board_mesh(
    shelf_id: str,
    basis_u: np.ndarray,
    basis_v: np.ndarray,
    dominant_normal: np.ndarray,
    min_u: float,
    max_u: float,
    min_v: float,
    max_v: float,
    mean_height: float,
    thickness: float,
) -> dict:
    top_height = mean_height + thickness * 0.5
    bottom_height = mean_height - thickness * 0.5
    top_vertices_uv = np.array(
        [
            [min_u, min_v],
            [max_u, min_v],
            [max_u, max_v],
            [min_u, max_v],
        ],
        dtype=float,
    )
    top_triangles = [[0, 1, 2], [0, 2, 3]]
    boundary_edges = [[0, 1], [1, 2], [2, 3], [3, 0]]

    top_face = [
        _compose_world_point(basis_u, basis_v, dominant_normal, float(coord_u), float(coord_v), top_height).tolist()
        for coord_u, coord_v in top_vertices_uv
    ]
    bottom_face = [
        _compose_world_point(basis_u, basis_v, dominant_normal, float(coord_u), float(coord_v), bottom_height).tolist()
        for coord_u, coord_v in top_vertices_uv
    ]
    vertices = top_face + bottom_face
    triangles = top_triangles + [[4, 6, 5], [4, 7, 6]]
    for start_index, end_index in boundary_edges:
        bottom_offset = 4
        triangles.append([start_index, end_index, bottom_offset + end_index])
        triangles.append([start_index, bottom_offset + end_index, bottom_offset + start_index])

    return {
        "id": f"{shelf_id}-board",
        "type": "shelf_board",
        "vertices": vertices,
        "triangles": triangles,
        "top_face": top_face,
        "bottom_face": bottom_face,
        "top_vertices_uv": top_vertices_uv.tolist(),
        "top_triangles": top_triangles,
        "boundary_edges": boundary_edges,
        "sample_step": max((max_u - min_u), (max_v - min_v)) / 4.0,
        "volume": {
            "u": [min_u, max_u],
            "v": [min_v, max_v],
            "n": [bottom_height, top_height],
        },
    }


def _fit_axis_bounds(
    values: np.ndarray,
    lower_quantile: float,
    upper_quantile: float,
    padding_ratio: float,
    minimum_span: float,
) -> tuple[float, float]:
    raw_min = float(values.min())
    raw_max = float(values.max())
    low, high = np.quantile(values, [lower_quantile, upper_quantile])
    low = float(low)
    high = float(high)
    core_span = max(high - low, minimum_span)
    padding = max(core_span * padding_ratio, 0.03)
    low = max(raw_min, low - padding)
    high = min(raw_max, high + padding)

    if high - low < minimum_span:
        center = 0.5 * (low + high)
        half_span = minimum_span * 0.5
        low = center - half_span
        high = center + half_span

    return float(low), float(high)


def _fit_board_bounds(coords_u: np.ndarray, coords_v: np.ndarray) -> tuple[float, float, float, float]:
    min_u, max_u = _fit_axis_bounds(
        coords_u,
        lower_quantile=0.04,
        upper_quantile=0.96,
        padding_ratio=0.04,
        minimum_span=0.35,
    )
    min_v, max_v = _fit_axis_bounds(
        coords_v,
        lower_quantile=0.08,
        upper_quantile=0.92,
        padding_ratio=0.10,
        minimum_span=0.18,
    )
    return min_u, max_u, min_v, max_v


def _board_margin(bounds: list[float]) -> float:
    span = float(bounds[1] - bounds[0])
    return max(0.04, span * 0.05)


def _clamp(value: float, minimum: float, maximum: float) -> float:
    return float(np.clip(value, minimum, maximum))


def _board_contains_local(point: dict, shelf: dict) -> bool:
    volume = shelf["mesh"]["volume"]
    margin_u = _board_margin(volume["u"])
    margin_v = _board_margin(volume["v"])
    return (
        volume["u"][0] - margin_u <= point["u"] <= volume["u"][1] + margin_u
        and volume["v"][0] - margin_v <= point["v"] <= volume["v"][1] + margin_v
    )


def _board_anchor_point(local_point: dict, shelf: dict) -> tuple[np.ndarray, dict]:
    axis_u = np.array(shelf["axes"]["u"], dtype=float)
    axis_v = np.array(shelf["axes"]["v"], dtype=float)
    normal = np.array(shelf["normal"], dtype=float)
    volume = shelf["mesh"]["volume"]
    thickness = float(shelf["extents"]["thickness"])

    anchor_u = _clamp(local_point["u"], volume["u"][0], volume["u"][1])
    anchor_v = _clamp(local_point["v"], volume["v"][0], volume["v"][1])
    top_n = float(volume["n"][1])
    anchor_n = top_n + max(0.03, thickness * 0.75)
    world_point = _compose_world_point(axis_u, axis_v, normal, anchor_u, anchor_v, anchor_n)
    return world_point, {"u": anchor_u, "v": anchor_v, "n": anchor_n}


def _select_rack_axes(shelves: list[dict]) -> tuple[str, str]:
    if not shelves:
        return "u", "v"

    mean_u = float(np.mean([shelf["mesh"]["volume"]["u"][1] - shelf["mesh"]["volume"]["u"][0] for shelf in shelves]))
    mean_v = float(np.mean([shelf["mesh"]["volume"]["v"][1] - shelf["mesh"]["volume"]["v"][0] for shelf in shelves]))
    if mean_u >= mean_v:
        return "u", "v"
    return "v", "u"


def _rebuild_board_mesh(
    shelf: dict,
    width_axis_name: str,
    depth_axis_name: str,
    width_bounds: list[float],
    depth_bounds: list[float],
) -> dict:
    axis_bounds = {
        width_axis_name: [float(width_bounds[0]), float(width_bounds[1])],
        depth_axis_name: [float(depth_bounds[0]), float(depth_bounds[1])],
    }
    return _build_board_mesh(
        shelf_id=shelf["id"],
        basis_u=np.array(shelf["axes"]["u"], dtype=float),
        basis_v=np.array(shelf["axes"]["v"], dtype=float),
        dominant_normal=np.array(shelf["normal"], dtype=float),
        min_u=axis_bounds["u"][0],
        max_u=axis_bounds["u"][1],
        min_v=axis_bounds["v"][0],
        max_v=axis_bounds["v"][1],
        mean_height=float(shelf["height"]),
        thickness=float(shelf["extents"]["thickness"]),
    )


def _normalize_shelf_layout(shelves: list[dict]) -> list[dict]:
    if len(shelves) <= 1:
        return shelves

    width_axis_name, depth_axis_name = _select_rack_axes(shelves)
    width_mins = np.array([shelf["mesh"]["volume"][width_axis_name][0] for shelf in shelves], dtype=float)
    width_maxs = np.array([shelf["mesh"]["volume"][width_axis_name][1] for shelf in shelves], dtype=float)
    depth_mins = np.array([shelf["mesh"]["volume"][depth_axis_name][0] for shelf in shelves], dtype=float)
    depth_maxs = np.array([shelf["mesh"]["volume"][depth_axis_name][1] for shelf in shelves], dtype=float)

    # Width: use consensus across shelves — a gondola bay has one fixed width.
    width_min = float(np.median(width_mins))
    width_max = float(np.median(width_maxs))
    width_span = max(width_max - width_min, 0.7)
    width_padding = max(0.04, width_span * 0.03)
    normalized_width_bounds = [width_min - width_padding, width_max + width_padding]

    # Depth: each shelf keeps its own measured bounds so boards that are
    # shallower/deeper than average aren't forced into an identical footprint.
    # Shelves whose depth span is an extreme outlier (< 40 % or > 250 % of the
    # median) fall back to the consensus estimate to avoid degenerate boards.
    depth_spans = depth_maxs - depth_mins
    consensus_depth_span = float(np.median(depth_spans))
    consensus_depth_min = float(np.quantile(depth_mins, 0.35))
    consensus_depth_max = float(np.quantile(depth_maxs, 0.65))

    normalized_shelves: list[dict] = []
    for shelf, d_min, d_max in zip(shelves, depth_mins, depth_maxs):
        raw_span = d_max - d_min
        if consensus_depth_span > 0 and (
            raw_span < consensus_depth_span * 0.40
            or raw_span > consensus_depth_span * 2.50
        ):
            d_min, d_max = consensus_depth_min, consensus_depth_max

        depth_span = max(d_max - d_min, 0.28)
        depth_padding_back = max(0.02, depth_span * 0.04)
        depth_padding_front = max(0.03, depth_span * 0.06)
        shelf_depth_bounds = [d_min - depth_padding_back, d_max + depth_padding_front]

        mesh = _rebuild_board_mesh(
            shelf,
            width_axis_name=width_axis_name,
            depth_axis_name=depth_axis_name,
            width_bounds=normalized_width_bounds,
            depth_bounds=shelf_depth_bounds,
        )
        enriched_shelf = dict(shelf)
        enriched_shelf["mesh"] = mesh
        enriched_shelf["bounds"] = {
            "u": mesh["volume"]["u"],
            "v": mesh["volume"]["v"],
        }
        enriched_shelf["extents"] = {
            "width": float(mesh["volume"][width_axis_name][1] - mesh["volume"][width_axis_name][0]),
            "depth": float(mesh["volume"][depth_axis_name][1] - mesh["volume"][depth_axis_name][0]),
            "thickness": float(mesh["volume"]["n"][1] - mesh["volume"]["n"][0]),
        }
        center_width = 0.5 * (mesh["volume"][width_axis_name][0] + mesh["volume"][width_axis_name][1])
        center_depth = 0.5 * (mesh["volume"][depth_axis_name][0] + mesh["volume"][depth_axis_name][1])
        centroid = _compose_world_point(
            np.array(shelf["axes"][width_axis_name], dtype=float),
            np.array(shelf["axes"][depth_axis_name], dtype=float),
            np.array(shelf["normal"], dtype=float),
            center_width,
            center_depth,
            float(shelf["height"]),
        )
        enriched_shelf["centroid"] = centroid.tolist()
        normalized_shelves.append(enriched_shelf)

    return normalized_shelves


def _build_box_part(
    part_id: str,
    kind: str,
    width_axis: np.ndarray,
    height_axis: np.ndarray,
    depth_axis: np.ndarray,
    center_width: float,
    center_height: float,
    center_depth: float,
    width: float,
    height: float,
    depth: float,
    color: str,
    opacity: float,
) -> dict:
    center = _compose_world_point(width_axis, height_axis, depth_axis, center_width, center_height, center_depth)
    half_width = width * 0.5
    half_height = height * 0.5
    half_depth = depth * 0.5
    offsets = [
        (-half_width, -half_height, -half_depth),
        (half_width, -half_height, -half_depth),
        (half_width, -half_height, half_depth),
        (-half_width, -half_height, half_depth),
        (-half_width, half_height, -half_depth),
        (half_width, half_height, -half_depth),
        (half_width, half_height, half_depth),
        (-half_width, half_height, half_depth),
    ]
    vertices = [
        (center + width_axis * offset_width + height_axis * offset_height + depth_axis * offset_depth).tolist()
        for offset_width, offset_height, offset_depth in offsets
    ]
    triangles = [
        [0, 1, 2],
        [0, 2, 3],
        [4, 6, 5],
        [4, 7, 6],
        [0, 4, 5],
        [0, 5, 1],
        [1, 5, 6],
        [1, 6, 2],
        [2, 6, 7],
        [2, 7, 3],
        [3, 7, 4],
        [3, 4, 0],
    ]
    return {
        "id": part_id,
        "kind": kind,
        "center": center.tolist(),
        "size": [float(width), float(height), float(depth)],
        "axes": {
            "width": width_axis.tolist(),
            "height": height_axis.tolist(),
            "depth": depth_axis.tolist(),
        },
        "vertices": vertices,
        "triangles": triangles,
        "style": {
            "color": color,
            "opacity": opacity,
        },
    }


def _group_by_orientation(
    candidates: Iterable[PlaneCandidate],
    max_angle_degrees: float = 12.0,
) -> list[dict]:
    """Return all orientation groups sorted by inlier support (descending)."""
    angle_threshold = cos(radians(max_angle_degrees))
    groups: list[dict] = []

    for candidate in candidates:
        assigned = False
        for group in groups:
            reference = group["normal"]
            alignment = float(np.dot(candidate.normal, reference))
            if abs(alignment) >= angle_threshold:
                oriented = candidate.normal if alignment >= 0 else -candidate.normal
                group["normal"] = _normalize(group["normal"] + oriented)
                group["planes"].append(candidate)
                group["support"] += len(candidate.inliers)
                assigned = True
                break
        if not assigned:
            groups.append(
                {
                    "normal": candidate.normal.copy(),
                    "planes": [candidate],
                    "support": len(candidate.inliers),
                }
            )

    return sorted(groups, key=lambda g: g["support"], reverse=True)


def _filter_outlier_points(points: np.ndarray, keep_percentile: float = 97.0) -> np.ndarray:
    """Remove points that are implausibly far from the scene centroid.

    COLMAP occasionally produces wildly misplaced 3D points (e.g. from bad feature
    matches).  These outliers dominate RANSAC because a plane through two outliers
    and one inlier can still accumulate many nearby-inlier counts.  Removing the
    top (100 - keep_percentile)% by distance from the median centroid is enough
    to leave the shelf region intact while eliminating pathological cases.
    """
    if len(points) < 4:
        return points
    centroid = np.median(points, axis=0)
    dists = np.linalg.norm(points - centroid, axis=1)
    radius = np.percentile(dists, keep_percentile)
    mask = dists <= radius
    filtered = points[mask]
    return filtered if len(filtered) >= 3 else points


def detect_shelf_planes(
    points: np.ndarray,
    hint_up: np.ndarray | None = None,
) -> list[dict]:
    if len(points) < 3:
        return []

    points = _filter_outlier_points(points)

    candidates = _extract_plane_candidates(points, hint_up=hint_up)
    if not candidates:
        return []

    all_groups = _group_by_orientation(candidates)
    if not all_groups:
        return []

    # Pick the orientation group whose normal is most parallel to world-up (shelves are
    # horizontal, walls/floor are not).  If no group clears the threshold, fall back to
    # the highest-support group so the pipeline always produces a result.
    dominant_group = all_groups[0]
    if hint_up is not None:
        up = _normalize(hint_up)
        for group in all_groups:
            if abs(float(np.dot(_normalize(group["normal"]), up))) >= SHELF_NORMAL_MIN_ALIGNMENT:
                dominant_group = group
                break

    dominant_normal = _normalize(dominant_group["normal"])
    dominant_candidates = dominant_group["planes"]

    heights = []
    aligned_candidates = []

    for candidate in dominant_candidates:
        normal = candidate.normal.copy()
        if np.dot(normal, dominant_normal) < 0:
            normal *= -1
        center = candidate.center
        height = float(np.dot(center, dominant_normal))
        heights.append([height])
        aligned_candidates.append(
            PlaneCandidate(
                normal=normal,
                center=center,
                inliers=candidate.inliers,
            )
        )

    height_samples = np.array(heights, dtype=float)
    clusterer = DBSCAN(eps=0.12, min_samples=1)
    labels = clusterer.fit_predict(height_samples)
    basis_u, basis_v = _plane_basis(dominant_normal)

    shelves: list[dict] = []
    for cluster_id in sorted(set(labels), key=lambda value: float(height_samples[labels == value].mean())):
        cluster_planes = [plane for plane, label in zip(aligned_candidates, labels) if label == cluster_id]
        cluster_points = np.concatenate([plane.inliers for plane in cluster_planes], axis=0)

        coords_u = cluster_points @ basis_u
        coords_v = cluster_points @ basis_v

        # Drop clusters too small to be a real shelf board (Issue B)
        raw_span_u = float(coords_u.max() - coords_u.min())
        raw_span_v = float(coords_v.max() - coords_v.min())
        if raw_span_u * raw_span_v < MIN_SHELF_AREA_M2:
            continue

        min_u, max_u, min_v, max_v = _fit_board_bounds(coords_u, coords_v)
        center_u = 0.5 * (min_u + max_u)
        center_v = 0.5 * (min_v + max_v)
        mean_height = float((cluster_points @ dominant_normal).mean())
        centroid = basis_u * center_u + basis_v * center_v + dominant_normal * mean_height
        thickness = 0.045
        shelf_id = f"shelf-{len(shelves) + 1}"
        mesh = _build_board_mesh(
            shelf_id=shelf_id,
            basis_u=basis_u,
            basis_v=basis_v,
            dominant_normal=dominant_normal,
            min_u=min_u,
            max_u=max_u,
            min_v=min_v,
            max_v=max_v,
            mean_height=mean_height,
            thickness=thickness,
        )

        shelves.append(
            {
                "id": shelf_id,
                "height": mean_height,
                "centroid": centroid.tolist(),
                "normal": dominant_normal.tolist(),
                "axes": {
                    "u": basis_u.tolist(),
                    "v": basis_v.tolist(),
                },
                "bounds": {
                    "u": [min_u, max_u],
                    "v": [min_v, max_v],
                },
                "extents": {
                    "width": max(max_u - min_u, 0.1),
                    "depth": max(max_v - min_v, 0.1),
                    "thickness": thickness,
                },
                "mesh": mesh,
                "point_count": int(len(cluster_points)),
                "_cluster_points": cluster_points.tolist(),  # used by build_rack_geometry for back-panel heuristic
            }
        )

    return _normalize_shelf_layout(shelves)


def detect_shelf_planes_from_mesh(
    mesh_path: str,
    hint_up: np.ndarray,
    normal_threshold: float = 0.75,
) -> list[dict]:
    """
    Detect shelf planes from a dense triangle mesh by clustering horizontal faces.

    Avoids RANSAC on noisy point clouds: each triangle face carries a precise
    normal, so we filter to faces that are within arccos(normal_threshold) of
    world-up and cluster their centroids by height.  This reliably finds the
    actual shelf board surfaces without fitting through walls or product faces.
    """
    import open3d as o3d

    mesh = o3d.io.read_triangle_mesh(str(mesh_path))
    mesh.compute_triangle_normals()

    triangles = np.asarray(mesh.triangles)
    vertices = np.asarray(mesh.vertices)
    tri_normals = np.asarray(mesh.triangle_normals)

    if len(tri_normals) == 0:
        return []

    up = _normalize(hint_up)
    if up is None:
        return []

    # Keep only faces whose normal aligns with world-up (horizontal surfaces)
    alignments = np.abs(tri_normals @ up)
    horiz_mask = alignments >= normal_threshold
    if horiz_mask.sum() < 50:
        return []

    face_verts = vertices[triangles[horiz_mask]]        # (N, 3, 3)
    face_centroids = face_verts.mean(axis=1)             # (N, 3)
    heights = face_centroids @ up                        # (N,)

    clusterer = DBSCAN(eps=0.04, min_samples=20)
    labels = clusterer.fit_predict(heights.reshape(-1, 1))

    basis_u, basis_v = _plane_basis(up)
    shelves: list[dict] = []

    for label in sorted(set(labels), key=lambda lb: float(heights[labels == lb].mean())):
        if label < 0:
            continue

        mask = labels == label
        cluster_pts = face_centroids[mask]
        mean_height = float(heights[mask].mean())

        coords_u = cluster_pts @ basis_u
        coords_v = cluster_pts @ basis_v

        if (coords_u.max() - coords_u.min()) * (coords_v.max() - coords_v.min()) < MIN_SHELF_AREA_M2:
            continue

        min_u, max_u, min_v, max_v = _fit_board_bounds(coords_u, coords_v)
        center_u = 0.5 * (min_u + max_u)
        center_v = 0.5 * (min_v + max_v)
        centroid = basis_u * center_u + basis_v * center_v + up * mean_height
        thickness = 0.045
        shelf_id = f"shelf-{len(shelves) + 1}"

        mesh_geo = _build_board_mesh(
            shelf_id=shelf_id,
            basis_u=basis_u,
            basis_v=basis_v,
            dominant_normal=up,
            min_u=min_u,
            max_u=max_u,
            min_v=min_v,
            max_v=max_v,
            mean_height=mean_height,
            thickness=thickness,
        )

        shelves.append(
            {
                "id": shelf_id,
                "height": mean_height,
                "centroid": centroid.tolist(),
                "normal": up.tolist(),
                "axes": {"u": basis_u.tolist(), "v": basis_v.tolist()},
                "bounds": {"u": [min_u, max_u], "v": [min_v, max_v]},
                "extents": {
                    "width": max(max_u - min_u, 0.1),
                    "depth": max(max_v - min_v, 0.1),
                    "thickness": thickness,
                },
                "mesh": mesh_geo,
                "point_count": int(mask.sum()),
                "_cluster_points": cluster_pts.tolist(),
            }
        )

    return _normalize_shelf_layout(shelves)


def shelf_local_coordinates(point: np.ndarray, shelf: dict) -> dict:
    axis_u = np.array(shelf["axes"]["u"], dtype=float)
    axis_v = np.array(shelf["axes"]["v"], dtype=float)
    normal = np.array(shelf["normal"], dtype=float)
    return {
        "u": float(np.dot(point, axis_u)),
        "v": float(np.dot(point, axis_v)),
        "n": float(np.dot(point, normal)),
    }


def pixel_to_world_ray(camera: Camera, image: ImagePose, u: float, v: float) -> np.ndarray:
    fx, fy, cx, cy = camera.intrinsics()
    ray_camera = _normalize(np.array([(u - cx) / fx, (v - cy) / fy, 1.0], dtype=float))
    ray_world = image.rotation.T @ ray_camera
    return _normalize(ray_world)


def _intersect_ray_with_shelf_plane(origin: np.ndarray, direction: np.ndarray, shelf: dict) -> tuple[float, np.ndarray] | None:
    normal = np.array(shelf["normal"], dtype=float)
    centroid = np.array(shelf["centroid"], dtype=float)
    denominator = float(np.dot(normal, direction))
    if abs(denominator) < 1e-6:
        return None

    distance = float(np.dot(normal, centroid - origin) / denominator)
    if distance <= 0:
        return None

    point = origin + direction * distance
    return distance, point


def _intersect_ray_with_shelf(origin: np.ndarray, direction: np.ndarray, shelf: dict) -> tuple[float, np.ndarray] | None:
    hit = _intersect_ray_with_shelf_plane(origin, direction, shelf)
    if hit is None:
        return None

    distance, point = hit
    local_point = shelf_local_coordinates(point, shelf)
    if not _board_contains_local(local_point, shelf):
        return None

    return distance, point


def _observed_points_in_bbox(
    reconstruction: Reconstruction,
    image: ImagePose,
    bbox: list[float],
) -> list[np.ndarray]:
    if not image.observations:
        return []

    x1, y1, x2, y2 = bbox
    observed_points: list[np.ndarray] = []
    for obs_x, obs_y, point_id in image.observations:
        if point_id == -1:
            continue
        if not (x1 <= obs_x <= x2 and y1 <= obs_y <= y2):
            continue
        point3d = reconstruction.points3d.get(point_id)
        if point3d is None:
            continue
        observed_points.append(point3d.xyz)
    return observed_points


def _local_overflow(local_point: dict[str, float], shelf: dict) -> float:
    volume = shelf["mesh"]["volume"]
    overflow_u = max(volume["u"][0] - local_point["u"], 0.0, local_point["u"] - volume["u"][1])
    overflow_v = max(volume["v"][0] - local_point["v"], 0.0, local_point["v"] - volume["v"][1])
    return float(max(overflow_u, overflow_v))


def _median_observed_local(observed_xyz: np.ndarray, shelf: dict) -> dict[str, float]:
    axis_u = np.array(shelf["axes"]["u"], dtype=float)
    axis_v = np.array(shelf["axes"]["v"], dtype=float)
    normal = np.array(shelf["normal"], dtype=float)
    return {
        "u": float(np.median(observed_xyz @ axis_u)),
        "v": float(np.median(observed_xyz @ axis_v)),
        "n": float(np.median(observed_xyz @ normal)),
    }


def _choose_shelf_from_observations(
    observed_points: list[np.ndarray],
    shelves: list[dict],
    origin: np.ndarray,
    direction: np.ndarray,
) -> tuple[dict, float, np.ndarray, dict[str, float], bool] | None:
    if len(observed_points) < OBSERVED_SHELF_MATCH_MIN_POINTS:
        return None

    observed_xyz = np.array(observed_points, dtype=float)
    best_match: tuple[tuple[float, float, float], dict, float, np.ndarray, dict[str, float], bool] | None = None
    for shelf in shelves:
        plane_hit = _intersect_ray_with_shelf_plane(origin, direction, shelf)
        if plane_hit is None:
            continue

        distance, point = plane_hit
        observed_local = _median_observed_local(observed_xyz, shelf)
        height_delta = abs(observed_local["n"] - float(shelf["height"]))
        overflow = _local_overflow(observed_local, shelf)
        if height_delta > OBSERVED_SHELF_MATCH_MAX_HEIGHT_DELTA_M:
            continue
        if overflow > OBSERVED_SHELF_MATCH_MAX_OVERFLOW_M:
            continue

        inside_shelf_volume = _board_contains_local(observed_local, shelf)
        score = (height_delta, overflow, float(distance))
        if best_match is None or score < best_match[0]:
            best_match = (score, shelf, distance, point, observed_local, inside_shelf_volume)

    if best_match is None:
        return None
    return best_match[1], best_match[2], best_match[3], best_match[4], best_match[5]


def project_detections_to_shelves(
    reconstruction: Reconstruction,
    shelves: list[dict],
    detections: list[dict],
) -> list[dict]:
    products: list[dict] = []

    for detection in detections:
        image = reconstruction.images.get(detection["image_name"])
        if image is None:
            continue
        camera = reconstruction.cameras[image.camera_id]
        x1, y1, x2, y2 = detection["bbox"]
        center_u = 0.5 * (x1 + x2)
        center_v = 0.5 * (y1 + y2)
        ray_direction = pixel_to_world_ray(camera, image, center_u, center_v)
        ray_origin = image.camera_center

        intersections = []
        for shelf in shelves:
            hit = _intersect_ray_with_shelf(ray_origin, ray_direction, shelf)
            if hit is None:
                continue
            intersections.append((shelf, *hit))

        observed_points = _observed_points_in_bbox(reconstruction, image, detection["bbox"])
        shelf_match = _choose_shelf_from_observations(
            observed_points,
            shelves,
            ray_origin,
            ray_direction,
        )
        if shelf_match is not None:
            shelf, distance, point, source_local, inside_shelf_volume = shelf_match
        else:
            if not intersections:
                continue
            shelf, distance, point = min(intersections, key=lambda item: item[1])
            source_local = shelf_local_coordinates(point, shelf)
            inside_shelf_volume = True

        anchor_point, anchor_local = _board_anchor_point(source_local, shelf)
        product_entry = {
            "id": detection["id"],
            "image_name": detection["image_name"],
            "label": detection["label"],
            "display_label": detection.get("display_label", detection["label"]),
            "model_label": detection.get("model_label"),
            "label_source": detection.get("label_source"),
            "identity_stage": detection.get("identity_stage"),
            "class_id": detection.get("class_id"),
            "class_name": detection.get("class_name"),
            "product_identity_label": bool(detection.get("product_identity_label", False)),
            "recognized_product_name": detection.get("recognized_product_name"),
            "product_identity_confidence": float(detection.get("product_identity_confidence") or 0.0),
            "product_identity_reason": detection.get("product_identity_reason"),
            "product_identity_source": detection.get("product_identity_source"),
            "confidence": detection["confidence"],
            "bbox": detection["bbox"],
            "p3d": anchor_point.tolist(),
            "projection_point": point.tolist(),
            "shelf_local": anchor_local,
            "ray_distance": float(distance),
            "shelf_id": shelf["id"],
            "crop_path": detection.get("crop_path"),
            "inside_shelf_volume": inside_shelf_volume,
        }
        # Pass through open-world identification metadata so the frontend can
        # show brand, Open Food Facts thumbnails, barcode, and the canonical
        # OFF entry without round-tripping through the cache files.
        for optional_key in (
            "brand",
            "off_code",
            "barcode",
            "reference_image_url",
            "open_world",
        ):
            if detection.get(optional_key) is not None:
                product_entry[optional_key] = detection[optional_key]
        products.append(product_entry)

    return _consensus_merge_products(products)


def _consensus_merge_products(products: list[dict], radius: float = 0.06) -> list[dict]:
    """Multi-view consensus. Detections that project within `radius` metres of
    each other on the same shelf are votes on the same physical product. Pool
    the votes, pick the winning label, and boost confidence by agreement count.
    Each cluster collapses to one merged detection that carries `multi_view`
    metadata so the frontend can show vote tallies and view counts."""
    if not products:
        return []

    by_shelf: dict[str, list[dict]] = {}
    for p in products:
        by_shelf.setdefault(p["shelf_id"], []).append(p)

    merged: list[dict] = []
    for shelf_products in by_shelf.values():
        if not shelf_products:
            continue

        positions = np.array([p["p3d"] for p in shelf_products])
        n = len(shelf_products)
        parent = list(range(n))

        def find(i: int) -> int:
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i

        def union(i: int, j: int) -> None:
            ri, rj = find(i), find(j)
            if ri != rj:
                parent[ri] = rj

        if n >= 2:
            tree = cKDTree(positions)
            for i, j in tree.query_pairs(r=radius):
                union(int(i), int(j))

        clusters: dict[int, list[int]] = {}
        for i in range(n):
            clusters.setdefault(find(i), []).append(i)

        for member_indices in clusters.values():
            cluster = [shelf_products[i] for i in member_indices]
            merged.append(_consensus_pick(cluster))

    return merged


def _consensus_pick(cluster: list[dict]) -> dict:
    """Pick the winning label across all views of a single physical product.
    Unrecognized views still count toward `view_count` but contribute no vote.
    Confidence is inflated by agreement: 1 - (1 - p)**n, capped at 0.99."""
    votes: dict[str, dict] = {}
    for det in cluster:
        if not det.get("product_identity_label"):
            continue
        label = (
            det.get("recognized_product_name")
            or det.get("display_label")
            or det.get("label")
        )
        if not label:
            continue
        det_conf = float(det.get("product_identity_confidence") or 0.0)
        weight = det_conf + 0.05 * float(det.get("confidence") or 0.0)
        bucket = votes.setdefault(
            label,
            {"weight": 0.0, "count": 0, "best_det": det, "best_conf": -1.0},
        )
        bucket["weight"] += weight
        bucket["count"] += 1
        if det_conf > bucket["best_conf"]:
            bucket["best_conf"] = det_conf
            bucket["best_det"] = det

    if votes:
        winning_label, winning_bucket = max(
            votes.items(),
            key=lambda kv: (kv[1]["weight"], kv[1]["count"]),
        )
        canonical = dict(winning_bucket["best_det"])
        n_agree = winning_bucket["count"]
        base_conf = float(canonical.get("product_identity_confidence") or 0.0)
        boosted = 1.0 - (1.0 - max(0.0, min(1.0, base_conf))) ** max(1, n_agree)
        canonical["product_identity_confidence"] = round(
            min(0.99, max(base_conf, boosted)), 4
        )
        canonical["multi_view"] = {
            "view_count": len(cluster),
            "agreeing_view_count": n_agree,
            "vote_tally": {
                lbl: {"weight": round(b["weight"], 4), "count": b["count"]}
                for lbl, b in votes.items()
            },
            "winning_label": winning_label,
            "competing_label_count": len(votes) - 1,
        }
        prior_reason = canonical.get("product_identity_reason") or "Identified."
        suffix = (
            f" Multi-view consensus: {n_agree}/{len(cluster)} views agreed"
            + (f", {len(votes) - 1} competing label(s) rejected." if len(votes) > 1 else ".")
        )
        canonical["product_identity_reason"] = prior_reason.rstrip(".") + "." + suffix
        return canonical

    fallback = max(cluster, key=lambda p: float(p.get("confidence") or 0.0))
    canonical = dict(fallback)
    canonical["multi_view"] = {
        "view_count": len(cluster),
        "agreeing_view_count": 0,
        "vote_tally": {},
        "winning_label": None,
        "competing_label_count": 0,
    }
    return canonical


def attach_products_to_shelves(shelves: list[dict], products: list[dict]) -> list[dict]:
    products_by_shelf = {shelf["id"]: [] for shelf in shelves}
    width_axis_name, depth_axis_name = _select_rack_axes(shelves)

    for product in products:
        shelf_products = products_by_shelf.get(product["shelf_id"])
        if shelf_products is None:
            continue
        shelf_products.append(
            {
                "id": product["id"],
                "label": product["label"],
                "display_label": product.get("display_label", product["label"]),
                "model_label": product.get("model_label"),
                "label_source": product.get("label_source"),
                "product_identity_label": bool(product.get("product_identity_label", False)),
                "recognized_product_name": product.get("recognized_product_name"),
                "product_identity_confidence": float(product.get("product_identity_confidence") or 0.0),
                "product_identity_reason": product.get("product_identity_reason"),
                "product_identity_source": product.get("product_identity_source"),
                "confidence": product["confidence"],
                "image_name": product["image_name"],
                "bbox": product["bbox"],
                "p3d": product["p3d"],
                "shelf_local": product["shelf_local"],
                "crop_url": product.get("crop_url"),
                "multi_view": product.get("multi_view"),
            }
        )

    enriched_shelves = []
    for shelf in shelves:
        shelf_products = products_by_shelf[shelf["id"]]
        named_products = [product for product in shelf_products if product.get("product_identity_label")]
        label_counts = Counter(product["display_label"] for product in named_products)
        enriched_shelf = dict(shelf)
        mesh = shelf["mesh"]
        enriched_shelf["product_count"] = len(shelf_products)
        enriched_shelf["product_ids"] = [product["id"] for product in shelf_products]
        enriched_shelf["products"] = shelf_products
        enriched_shelf["mesh"] = mesh
        enriched_shelf["geometry"] = {
            "kind": "shelf_board",
            "width_axis": width_axis_name,
            "depth_axis": depth_axis_name,
            "width_bounds": mesh["volume"][width_axis_name],
            "depth_bounds": mesh["volume"][depth_axis_name],
            "height_bounds": mesh["volume"]["n"],
            "anchor_height": float(mesh["volume"]["n"][1] + max(0.03, shelf["extents"]["thickness"] * 0.75)),
        }
        enriched_shelf["extents"] = {
            "width": float(mesh["volume"][width_axis_name][1] - mesh["volume"][width_axis_name][0]),
            "depth": float(mesh["volume"][depth_axis_name][1] - mesh["volume"][depth_axis_name][0]),
            "thickness": float(mesh["volume"]["n"][1] - mesh["volume"]["n"][0]),
        }
        enriched_shelf["inventory"] = {
            "product_identity_labels": bool(named_products),
            "labels": [
                {"label": label, "count": count}
                for label, count in sorted(label_counts.items(), key=lambda item: (-item[1], item[0]))
            ]
        }
        enriched_shelves.append(enriched_shelf)
    return enriched_shelves


def _scale_bounds(bounds: list[float], scale: float) -> list[float]:
    return [float(bounds[0] * scale), float(bounds[1] * scale)]


def _scale_mesh(mesh: dict, scale: float) -> dict:
    scaled_mesh = dict(mesh)
    for key in ("vertices", "top_face", "bottom_face", "top_vertices_uv"):
        if key in mesh:
            scaled_mesh[key] = [[float(component * scale) for component in vertex] for vertex in mesh[key]]
    scaled_mesh["volume"] = {
        axis_name: _scale_bounds(axis_bounds, scale)
        for axis_name, axis_bounds in mesh["volume"].items()
    }
    if "sample_step" in mesh:
        scaled_mesh["sample_step"] = float(mesh["sample_step"] * scale)
    return scaled_mesh


def _scale_shelf(shelf: dict, scale: float) -> dict:
    scaled_shelf = dict(shelf)
    scaled_shelf["height"] = float(shelf["height"] * scale)
    scaled_shelf["centroid"] = [float(component * scale) for component in shelf["centroid"]]
    scaled_shelf["bounds"] = {
        axis_name: _scale_bounds(axis_bounds, scale)
        for axis_name, axis_bounds in shelf["bounds"].items()
    }
    scaled_shelf["extents"] = {
        axis_name: float(value * scale)
        for axis_name, value in shelf["extents"].items()
    }
    scaled_shelf["mesh"] = _scale_mesh(shelf["mesh"], scale)
    return scaled_shelf


def _scale_product(product: dict, scale: float) -> dict:
    scaled_product = dict(product)
    scaled_product["p3d"] = [float(component * scale) for component in product["p3d"]]
    if product.get("projection_point") is not None:
        scaled_product["projection_point"] = [float(component * scale) for component in product["projection_point"]]
    scaled_product["shelf_local"] = {
        axis_name: float(value * scale)
        for axis_name, value in product["shelf_local"].items()
    }
    if product.get("ray_distance") is not None:
        scaled_product["ray_distance"] = float(product["ray_distance"] * scale)
    return scaled_product


def _translate_points(points: list[list[float]], delta_vector: np.ndarray) -> list[list[float]]:
    return [
        [
            float(point[0] + delta_vector[0]),
            float(point[1] + delta_vector[1]),
            float(point[2] + delta_vector[2]),
        ]
        for point in points
    ]


def _shift_shelf_height(shelf: dict, delta_height: float) -> dict:
    if abs(delta_height) < 1e-9:
        return shelf

    normal = np.array(shelf["normal"], dtype=float)
    delta_vector = normal * delta_height
    shifted_shelf = dict(shelf)
    shifted_shelf["height"] = float(shelf["height"] + delta_height)
    shifted_shelf["centroid"] = [
        float(shelf["centroid"][0] + delta_vector[0]),
        float(shelf["centroid"][1] + delta_vector[1]),
        float(shelf["centroid"][2] + delta_vector[2]),
    ]
    shifted_mesh = dict(shelf["mesh"])
    shifted_mesh["vertices"] = _translate_points(shelf["mesh"]["vertices"], delta_vector)
    if "top_face" in shelf["mesh"]:
        shifted_mesh["top_face"] = _translate_points(shelf["mesh"]["top_face"], delta_vector)
    if "bottom_face" in shelf["mesh"]:
        shifted_mesh["bottom_face"] = _translate_points(shelf["mesh"]["bottom_face"], delta_vector)
    shifted_mesh["volume"] = dict(shelf["mesh"]["volume"])
    shifted_mesh["volume"]["n"] = [
        float(shelf["mesh"]["volume"]["n"][0] + delta_height),
        float(shelf["mesh"]["volume"]["n"][1] + delta_height),
    ]
    shifted_shelf["mesh"] = shifted_mesh
    return shifted_shelf


def _shift_product_height(product: dict, normal: np.ndarray, delta_height: float) -> dict:
    if abs(delta_height) < 1e-9:
        return product

    delta_vector = normal * delta_height
    shifted_product = dict(product)
    shifted_product["p3d"] = [
        float(product["p3d"][0] + delta_vector[0]),
        float(product["p3d"][1] + delta_vector[1]),
        float(product["p3d"][2] + delta_vector[2]),
    ]
    if product.get("projection_point") is not None:
        shifted_product["projection_point"] = [
            float(product["projection_point"][0] + delta_vector[0]),
            float(product["projection_point"][1] + delta_vector[1]),
            float(product["projection_point"][2] + delta_vector[2]),
        ]
    shifted_product["shelf_local"] = dict(product["shelf_local"])
    shifted_product["shelf_local"]["n"] = float(product["shelf_local"]["n"] + delta_height)
    return shifted_product


def apply_layout_model(shelves: list[dict], products: list[dict], layout_config: dict | None) -> tuple[list[dict], list[dict], dict]:
    layout_config = layout_config or {}
    fixture_template = str(layout_config.get("fixture_template") or "gondola")
    reference_width = layout_config.get("reference_width_m")
    reference_depth = layout_config.get("reference_depth_m")
    shelf_offsets = {
        str(shelf_id): float(offset)
        for shelf_id, offset in (layout_config.get("shelf_height_offsets_m") or {}).items()
    }

    if not shelves:
        return [], products, {
            "fixture_template": fixture_template,
            "reference_width_m": reference_width,
            "reference_depth_m": reference_depth,
            "shelf_height_offsets_m": shelf_offsets,
            "scale_factor": 1.0,
            "scale_source": "scene_units",
            "measured_width": None,
            "measured_depth": None,
        }

    width_axis_name, depth_axis_name = _select_rack_axes(shelves)
    measured_width = float(
        max(shelf["mesh"]["volume"][width_axis_name][1] for shelf in shelves)
        - min(shelf["mesh"]["volume"][width_axis_name][0] for shelf in shelves)
    )
    measured_depth = float(
        max(shelf["mesh"]["volume"][depth_axis_name][1] for shelf in shelves)
        - min(shelf["mesh"]["volume"][depth_axis_name][0] for shelf in shelves)
    )

    scale_factor = 1.0
    scale_source = "scene_units"
    if reference_width:
        scale_factor = float(reference_width) / max(measured_width, 1e-6)
        scale_source = "reference_width_m"
    elif reference_depth:
        scale_factor = float(reference_depth) / max(measured_depth, 1e-6)
        scale_source = "reference_depth_m"

    transformed_shelves = [_scale_shelf(shelf, scale_factor) for shelf in shelves]
    transformed_products = [_scale_product(product, scale_factor) for product in products]

    shifted_shelves: list[dict] = []
    shelf_normals: dict[str, np.ndarray] = {}
    for shelf in transformed_shelves:
        delta_height = shelf_offsets.get(shelf["id"], 0.0)
        shifted_shelf = _shift_shelf_height(shelf, delta_height)
        shifted_shelves.append(shifted_shelf)
        shelf_normals[shelf["id"]] = np.array(shelf["normal"], dtype=float)

    shifted_products = [
        _shift_product_height(product, shelf_normals[product["shelf_id"]], shelf_offsets.get(product["shelf_id"], 0.0))
        if product["shelf_id"] in shelf_normals
        else product
        for product in transformed_products
    ]

    return shifted_shelves, shifted_products, {
        "fixture_template": fixture_template,
        "reference_width_m": reference_width,
        "reference_depth_m": reference_depth,
        "shelf_height_offsets_m": shelf_offsets,
        "scale_factor": float(scale_factor),
        "scale_source": scale_source,
        "measured_width": float(measured_width * scale_factor),
        "measured_depth": float(measured_depth * scale_factor),
    }


def build_rack_geometry(shelves: list[dict], fixture_template: str = "gondola") -> dict | None:
    if not shelves:
        return None

    width_axis_name, depth_axis_name = _select_rack_axes(shelves)
    width_axis = np.array(shelves[0]["axes"][width_axis_name], dtype=float)
    depth_axis = np.array(shelves[0]["axes"][depth_axis_name], dtype=float)
    height_axis = np.array(shelves[0]["normal"], dtype=float)

    width_bounds = [
        float(min(shelf["mesh"]["volume"][width_axis_name][0] for shelf in shelves)),
        float(max(shelf["mesh"]["volume"][width_axis_name][1] for shelf in shelves)),
    ]
    depth_bounds = [
        float(min(shelf["mesh"]["volume"][depth_axis_name][0] for shelf in shelves)),
        float(max(shelf["mesh"]["volume"][depth_axis_name][1] for shelf in shelves)),
    ]
    height_bounds = [
        float(min(shelf["mesh"]["volume"]["n"][0] for shelf in shelves)),
        float(max(shelf["mesh"]["volume"]["n"][1] for shelf in shelves)),
    ]

    width_span = width_bounds[1] - width_bounds[0]
    depth_span = depth_bounds[1] - depth_bounds[0]
    toe_kick = 0.12
    top_clearance = 0.14
    rack_bottom = height_bounds[0] - toe_kick
    rack_top = height_bounds[1] + top_clearance
    rack_height = rack_top - rack_bottom
    width_center = 0.5 * (width_bounds[0] + width_bounds[1])
    depth_center = 0.5 * (depth_bounds[0] + depth_bounds[1])
    height_center = rack_bottom + rack_height * 0.5

    side_panel_thickness = max(0.08, width_span * 0.028)
    side_panel_depth = depth_span + max(0.08, depth_span * 0.10)
    back_panel_thickness = max(0.025, depth_span * 0.035)
    plinth_height = 0.12
    plinth_depth = depth_span + max(0.10, depth_span * 0.12)
    top_cap_height = 0.05
    inner_width_span = max(width_span - side_panel_thickness * 2.0, 0.3)
    outer_width_span = width_span + side_panel_thickness * 2.0
    outer_depth_span = side_panel_depth

    # Determine which depth end is the wall side.  The wall is not reconstructed by
    # COLMAP, so fewer 3D points land near it.  Count points in the front and back 20%
    # of the depth range — the sparser end is the wall.
    back_depth = depth_bounds[0]  # fallback: assume depth[0] is the wall side
    _all_depth_coords: list[float] = []
    for _s in shelves:
        _pts = _s.get("_cluster_points")
        if _pts:
            _all_depth_coords.extend((np.array(_pts, dtype=float) @ depth_axis).tolist())
    if len(_all_depth_coords) >= 20:
        _d_arr = np.array(_all_depth_coords)
        _band = depth_span * 0.2
        _low_count = int(np.sum(_d_arr <= depth_bounds[0] + _band))
        _high_count = int(np.sum(_d_arr >= depth_bounds[1] - _band))
        if _high_count < _low_count:
            back_depth = depth_bounds[1]
    _back_sign = -1.0 if back_depth == depth_bounds[0] else 1.0
    _front_depth = depth_bounds[1] if back_depth == depth_bounds[0] else depth_bounds[0]

    side_depth_center = back_depth + _back_sign * (-side_panel_depth * 0.5 + back_panel_thickness * 0.15)
    back_panel_height = rack_height - plinth_height * 0.35
    back_panel_center_height = rack_bottom + plinth_height + back_panel_height * 0.5
    front_face_depth = _front_depth + _back_sign * (-outer_depth_span * 0.5)
    top_cap_center_height = rack_top - top_cap_height * 0.5
    side_center_height = rack_bottom + rack_height * 0.5

    parts = [
        _build_box_part(
            part_id="rack-back-panel",
            kind="back_panel",
            width_axis=width_axis,
            height_axis=height_axis,
            depth_axis=depth_axis,
            center_width=width_center,
            center_height=back_panel_center_height,
            center_depth=back_depth + _back_sign * (-back_panel_thickness * 0.5),
            width=inner_width_span,
            height=back_panel_height,
            depth=back_panel_thickness,
            color="#d1d5db",
            opacity=0.96,
        )
    ]

    if fixture_template == "gondola":
        parts.extend(
            [
                _build_box_part(
                    part_id="rack-plinth",
                    kind="plinth",
                    width_axis=width_axis,
                    height_axis=height_axis,
                    depth_axis=depth_axis,
                    center_width=width_center,
                    center_height=rack_bottom + plinth_height * 0.5,
                    center_depth=depth_center,
                    width=outer_width_span,
                    height=plinth_height,
                    depth=plinth_depth,
                    color="#cbd5e1",
                    opacity=0.98,
                ),
                _build_box_part(
                    part_id="rack-top-cap",
                    kind="top_cap",
                    width_axis=width_axis,
                    height_axis=height_axis,
                    depth_axis=depth_axis,
                    center_width=width_center,
                    center_height=top_cap_center_height,
                    center_depth=front_face_depth - outer_depth_span * 0.5,
                    width=outer_width_span,
                    height=top_cap_height,
                    depth=outer_depth_span,
                    color="#e5e7eb",
                    opacity=0.95,
                ),
                _build_box_part(
                    part_id="rack-kickplate",
                    kind="kickplate",
                    width_axis=width_axis,
                    height_axis=height_axis,
                    depth_axis=depth_axis,
                    center_width=width_center,
                    center_height=rack_bottom + plinth_height * 0.52,
                    center_depth=_front_depth + _back_sign * (-max(0.03, depth_span * 0.08)),
                    width=outer_width_span,
                    height=max(0.07, plinth_height * 0.72),
                    depth=max(0.03, back_panel_thickness * 1.4),
                    color="#94a3b8",
                    opacity=0.98,
                ),
            ]
        )
    else:
        wall_base_height = max(0.04, plinth_height * 0.45)
        parts.append(
            _build_box_part(
                part_id="rack-wall-base",
                kind="base_rail",
                width_axis=width_axis,
                height_axis=height_axis,
                depth_axis=depth_axis,
                center_width=width_center,
                center_height=rack_bottom + wall_base_height * 0.5,
                center_depth=back_depth + _back_sign * (-max(0.03, depth_span * 0.12)),
                width=inner_width_span,
                height=wall_base_height,
                depth=max(0.04, depth_span * 0.10),
                color="#94a3b8",
                opacity=0.96,
            )
        )

    for side_name, side_width in (
        ("left", width_bounds[0] - side_panel_thickness * 0.5),
        ("right", width_bounds[1] + side_panel_thickness * 0.5),
    ):
        panel_depth = side_panel_depth if fixture_template == "gondola" else depth_span + max(0.03, depth_span * 0.03)
        panel_center_depth = side_depth_center if fixture_template == "gondola" else back_depth + _back_sign * (-panel_depth * 0.5)
        parts.append(
            _build_box_part(
                part_id=f"rack-side-panel-{side_name}",
                kind="side_panel",
                width_axis=width_axis,
                height_axis=height_axis,
                depth_axis=depth_axis,
                center_width=side_width,
                center_height=side_center_height,
                center_depth=panel_center_depth,
                width=side_panel_thickness,
                height=rack_height,
                depth=panel_depth,
                color="#f3f4f6",
                opacity=0.98,
            )
        )

    return {
        "id": "rack-1",
        "kind": fixture_template,
        "axes": {
            "width": width_axis.tolist(),
            "height": height_axis.tolist(),
            "depth": depth_axis.tolist(),
        },
        "bounds": {
            "width": width_bounds,
            "height": [rack_bottom, rack_top],
            "depth": depth_bounds,
        },
        "parts": parts,
    }
