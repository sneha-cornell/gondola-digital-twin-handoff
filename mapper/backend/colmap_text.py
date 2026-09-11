from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import numpy as np


@dataclass(slots=True)
class Camera:
    camera_id: int
    model: str
    width: int
    height: int
    params: np.ndarray

    def intrinsics(self) -> tuple[float, float, float, float]:
        if self.model in {"SIMPLE_PINHOLE", "SIMPLE_RADIAL", "RADIAL"}:
            focal, cx, cy = self.params[:3]
            return float(focal), float(focal), float(cx), float(cy)

        if self.model in {"PINHOLE", "OPENCV", "FULL_OPENCV", "OPENCV_FISHEYE"}:
            fx, fy, cx, cy = self.params[:4]
            return float(fx), float(fy), float(cx), float(cy)

        raise ValueError(f"Unsupported camera model: {self.model}")


@dataclass(slots=True)
class ImagePose:
    image_id: int
    qvec: np.ndarray
    tvec: np.ndarray
    camera_id: int
    name: str
    rotation: np.ndarray
    camera_center: np.ndarray
    observations: tuple[tuple[float, float, int], ...] = ()


@dataclass(slots=True)
class Point3D:
    point_id: int
    xyz: np.ndarray
    error: float


@dataclass(slots=True)
class Reconstruction:
    cameras: Dict[int, Camera]
    images: Dict[str, ImagePose]
    points3d: Dict[int, Point3D]


def qvec_to_rotation(qvec: np.ndarray) -> np.ndarray:
    qw, qx, qy, qz = qvec
    return np.array(
        [
            [
                1 - 2 * qy * qy - 2 * qz * qz,
                2 * qx * qy - 2 * qw * qz,
                2 * qx * qz + 2 * qw * qy,
            ],
            [
                2 * qx * qy + 2 * qw * qz,
                1 - 2 * qx * qx - 2 * qz * qz,
                2 * qy * qz - 2 * qw * qx,
            ],
            [
                2 * qx * qz - 2 * qw * qy,
                2 * qy * qz + 2 * qw * qx,
                1 - 2 * qx * qx - 2 * qy * qy,
            ],
        ],
        dtype=float,
    )


def _iter_data_lines(path: Path) -> list[str]:
    return [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


def load_cameras(path: Path) -> Dict[int, Camera]:
    cameras: Dict[int, Camera] = {}
    for line in _iter_data_lines(path):
        parts = line.split()
        camera_id = int(parts[0])
        cameras[camera_id] = Camera(
            camera_id=camera_id,
            model=parts[1],
            width=int(parts[2]),
            height=int(parts[3]),
            params=np.array([float(value) for value in parts[4:]], dtype=float),
        )
    return cameras


def load_images(path: Path) -> Dict[str, ImagePose]:
    # COLMAP images.txt has exactly two lines per image:
    #   line 1: IMAGE_ID QW QX QY QZ TX TY TZ CAMERA_ID NAME
    #   line 2: 2D-point observations (may be an empty line for images with no matches)
    # _iter_data_lines strips blank lines, so images with empty observation lines
    # would cause an off-by-one error (the next image's header is treated as
    # observations and silently skipped).  Read raw lines here, stripping only
    # comment lines so the blank observation lines are preserved.
    data_lines = [
        line
        for line in path.read_text(encoding="utf-8").splitlines()
        if not line.lstrip().startswith("#")
    ]
    images: Dict[str, ImagePose] = {}

    index = 0
    while index < len(data_lines):
        header = data_lines[index].strip()
        index += 1
        obs_line = ""
        # Always consume the observations line (even if blank) to stay in sync
        if index < len(data_lines):
            obs_line = data_lines[index]
            index += 1

        if not header:
            continue  # blank line between entries — skip gracefully

        parts = header.split(maxsplit=9)
        if len(parts) < 10:
            continue  # malformed header

        observations: list[tuple[float, float, int]] = []
        if obs_line.strip():
            observation_parts = obs_line.strip().split()
            for obs_index in range(0, len(observation_parts) - 2, 3):
                observations.append(
                    (
                        float(observation_parts[obs_index]),
                        float(observation_parts[obs_index + 1]),
                        int(observation_parts[obs_index + 2]),
                    )
                )

        image_id = int(parts[0])
        qvec = np.array([float(value) for value in parts[1:5]], dtype=float)
        tvec = np.array([float(value) for value in parts[5:8]], dtype=float)
        camera_id = int(parts[8])
        name = parts[9]
        rotation = qvec_to_rotation(qvec)
        camera_center = -rotation.T @ tvec
        images[name] = ImagePose(
            image_id=image_id,
            qvec=qvec,
            tvec=tvec,
            camera_id=camera_id,
            name=name,
            rotation=rotation,
            camera_center=camera_center,
            observations=tuple(observations),
        )

    return images


def load_points3d(path: Path) -> Dict[int, Point3D]:
    points: Dict[int, Point3D] = {}
    for line in _iter_data_lines(path):
        parts = line.split()
        point_id = int(parts[0])
        xyz = np.array([float(value) for value in parts[1:4]], dtype=float)
        error = float(parts[7])
        points[point_id] = Point3D(point_id=point_id, xyz=xyz, error=error)
    return points


def load_text_model(text_dir: Path) -> Reconstruction:
    cameras = load_cameras(text_dir / "cameras.txt")
    images = load_images(text_dir / "images.txt")
    points3d = load_points3d(text_dir / "points3D.txt")
    return Reconstruction(cameras=cameras, images=images, points3d=points3d)
