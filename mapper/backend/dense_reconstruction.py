"""
Dense 3D reconstruction from COLMAP camera poses + DepthAnything V2.

Pipeline:
  1. DepthAnything V2 Small estimates a relative depth map per image.
  2. Visible COLMAP sparse 3D points anchor each depth map to metric scale.
  3. Open3D ScalableTSDF fuses all depth maps into a dense volumetric mesh.
  4. Mesh vertices are returned as a dense (N,3) point cloud for shelf detection.

The result is cached at <job_dir>/dense/points.npy and <job_dir>/dense/mesh.ply.
Delete those files to force a rebuild.
"""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image as PILImage

logger = logging.getLogger(__name__)

# DepthAnything V2 Small — ~97 MB, downloaded once to HF cache
_DA2_MODEL_ID = "depth-anything/Depth-Anything-V2-Small-hf"

# Longest image dimension used for depth estimation and TSDF integration.
# 1024 balances speed vs. spatial resolution for shelf-scale scenes.
_PROCESS_LONG_SIDE = 1024

_depth_model = None
_depth_processor = None
_depth_device: str | None = None
_open3d = None


def _load_open3d():
    global _open3d
    if _open3d is None:
        try:
            import open3d as o3d
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "Dense reconstruction requires the optional 'open3d' dependency. "
                "Install the full backend requirements before running that pipeline."
            ) from exc
        _open3d = o3d
    return _open3d


def _load_depth_model() -> tuple:
    global _depth_model, _depth_processor, _depth_device
    if _depth_model is None:
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation

        _depth_device = "mps" if torch.backends.mps.is_available() else "cpu"
        logger.info("Loading DepthAnything V2 Small on %s …", _depth_device)
        _depth_processor = AutoImageProcessor.from_pretrained(_DA2_MODEL_ID)
        _depth_model = (
            AutoModelForDepthEstimation.from_pretrained(_DA2_MODEL_ID)
            .to(_depth_device)
            .eval()
        )
        logger.info("DepthAnything V2 ready.")
    return _depth_model, _depth_processor, _depth_device


def _estimate_depth_relative(image_rgb: np.ndarray) -> np.ndarray:
    """Return a relative depth map (H×W float32) in arbitrary units."""
    model, processor, device = _load_depth_model()
    h, w = image_rgb.shape[:2]
    pil = PILImage.fromarray(image_rgb)
    inputs = processor(images=pil, return_tensors="pt").to(device)
    with torch.no_grad():
        depth = model(**inputs).predicted_depth  # (1, H', W')
    depth_full = (
        torch.nn.functional.interpolate(
            depth.unsqueeze(1),
            size=(h, w),
            mode="bilinear",
            align_corners=False,
        )
        .squeeze()
        .cpu()
        .numpy()
        .astype(np.float32)
    )
    return depth_full


def _scale_to_metric(
    depth_rel: np.ndarray,
    img_pose,
    reconstruction,
    fx_s: float,
    fy_s: float,
    cx_s: float,
    cy_s: float,
    scale_xy: float,
) -> tuple[np.ndarray, bool]:
    """
    Align a relative depth map to metric scale using COLMAP sparse 3D points
    that are visible in this image.

    Returns (metric_depth_map, success).
    """
    h, w = depth_rel.shape
    ratios: list[float] = []

    for px_orig, py_orig, point3d_id in img_pose.observations:
        if point3d_id < 0:
            continue
        pt3d = reconstruction.points3d.get(point3d_id)
        if pt3d is None:
            continue

        # Metric depth of this 3D point in camera space
        xyz_cam = img_pose.rotation @ pt3d.xyz + img_pose.tvec
        metric_z = float(xyz_cam[2])
        if metric_z <= 0:
            continue

        # Corresponding pixel in the *processed* (downscaled) image
        ix = int(round(px_orig * scale_xy))
        iy = int(round(py_orig * scale_xy))
        if not (0 <= ix < w and 0 <= iy < h):
            continue

        predicted = float(depth_rel[iy, ix])
        if predicted > 1e-6:
            ratios.append(metric_z / predicted)

    if len(ratios) < 5:
        return depth_rel, False

    scale = float(np.median(ratios))
    return depth_rel * scale, True


def run_dense_pipeline(job_dir: Path, reconstruction) -> np.ndarray:
    """
    Build a dense point cloud and mesh for *job_dir* using camera poses from
    *reconstruction* (a colmap_text.Reconstruction) and DepthAnything V2.

    Returns an (N, 3) float64 numpy array of world-space points suitable for
    plane detection.  Falls back to the sparse COLMAP point cloud on failure.
    """
    dense_dir = job_dir / "dense"
    cache_path = dense_dir / "points.npy"
    o3d = _load_open3d()

    if cache_path.exists():
        logger.info("Loading cached dense point cloud from %s", cache_path)
        return np.load(str(cache_path))

    dense_dir.mkdir(parents=True, exist_ok=True)
    images_dir = job_dir / "images"

    # Use 2 cm voxels — fine enough for shelf-surface planarity at meter scale.
    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=0.02,
        sdf_trunc=0.05,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
    )

    n_total = len(reconstruction.images)
    n_integrated = 0

    for img_name, img_pose in sorted(reconstruction.images.items()):
        img_path = images_dir / img_name
        if not img_path.exists():
            logger.debug("Image not found, skipping: %s", img_name)
            continue

        camera = reconstruction.cameras[img_pose.camera_id]
        fx, fy, cx, cy = camera.intrinsics()

        # Resize to _PROCESS_LONG_SIDE on the longest edge
        pil_orig = PILImage.open(img_path).convert("RGB")
        w_orig, h_orig = pil_orig.size
        scale = _PROCESS_LONG_SIDE / max(w_orig, h_orig)
        w_proc = int(w_orig * scale)
        h_proc = int(h_orig * scale)
        pil_proc = pil_orig.resize((w_proc, h_proc), PILImage.LANCZOS)
        img_rgb = np.array(pil_proc)

        # Relative depth at processed size
        try:
            depth_rel = _estimate_depth_relative(img_rgb)
        except Exception as exc:
            import traceback
            logger.warning("Depth estimation failed for %s: %s", img_name, exc)
            traceback.print_exc()
            continue

        # Scale intrinsics to processed image size
        fx_s, fy_s, cx_s, cy_s = fx * scale, fy * scale, cx * scale, cy * scale

        # Scale depth to metric using sparse COLMAP anchors
        depth_metric, ok = _scale_to_metric(
            depth_rel, img_pose, reconstruction, fx_s, fy_s, cx_s, cy_s, scale
        )
        if not ok:
            logger.debug("Skipping %s — insufficient COLMAP anchor points.", img_name)
            continue

        # Clamp to plausible range for indoor/shelf scenes
        depth_metric = np.clip(depth_metric, 0.05, 10.0)

        # Build Open3D RGBD image (depth stored as millimetres in uint16)
        color_o3d = o3d.geometry.Image(img_rgb.astype(np.uint8))
        depth_mm = np.clip(depth_metric * 1000.0, 0, 65535).astype(np.uint16)
        depth_o3d = o3d.geometry.Image(depth_mm)
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color_o3d,
            depth_o3d,
            depth_scale=1000.0,
            depth_trunc=8.0,
            convert_rgb_to_intensity=False,
        )

        intrinsic = o3d.camera.PinholeCameraIntrinsic(
            w_proc, h_proc, fx_s, fy_s, cx_s, cy_s
        )

        # World-to-camera extrinsic matrix [R | t]
        extrinsic = np.eye(4)
        extrinsic[:3, :3] = img_pose.rotation
        extrinsic[:3, 3] = img_pose.tvec

        volume.integrate(rgbd, intrinsic, extrinsic)
        n_integrated += 1
        logger.info("  TSDF: integrated %d / %d (%s)", n_integrated, n_total, img_name)

    if n_integrated == 0:
        logger.warning(
            "No images integrated — falling back to sparse COLMAP point cloud."
        )
        return np.array(
            [pt.xyz for pt in reconstruction.points3d.values()], dtype=float
        )

    logger.info("Extracting mesh from TSDF volume …")
    mesh = volume.extract_triangle_mesh()
    mesh.compute_vertex_normals()

    mesh_path = dense_dir / "mesh.ply"
    o3d.io.write_triangle_mesh(str(mesh_path), mesh)
    logger.info("Saved dense mesh → %s  (%d vertices)", mesh_path, len(mesh.vertices))

    # Downsample to ~80k triangles for fast browser loading (PLYLoader)
    target_triangles = 80_000
    n_tri = len(np.asarray(mesh.triangles))
    if n_tri > target_triangles:
        reduction = target_triangles / n_tri
        mesh_small = mesh.simplify_quadric_decimation(target_triangles)
        mesh_small.compute_vertex_normals()
        small_path = dense_dir / "mesh_view.ply"
        o3d.io.write_triangle_mesh(str(small_path), mesh_small)
        logger.info(
            "Saved view mesh → %s  (%d triangles, %.1f MB)",
            small_path,
            len(np.asarray(mesh_small.triangles)),
            small_path.stat().st_size / 1e6,
        )

    points = np.asarray(mesh.vertices, dtype=float)
    np.save(str(cache_path), points)

    return points
