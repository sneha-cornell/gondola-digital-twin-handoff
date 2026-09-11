from __future__ import annotations

import json
import logging
import os
import re
import shutil
import subprocess

logger = logging.getLogger(__name__)
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import uvicorn
from fastapi import BackgroundTasks, FastAPI, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from colmap_text import load_text_model
from detector import ObjectDetector
from product_classes import build_class_records
from product_identifier import ProductIdentifier
from geometry import (
    apply_layout_model,
    attach_products_to_shelves,
    build_rack_geometry,
    detect_shelf_planes,
    detect_shelf_planes_from_mesh,
    project_detections_to_shelves,
)
from dense_reconstruction import run_dense_pipeline, _load_depth_model

BASE_DIR = Path(__file__).resolve().parent
WORKSPACE_DIR = Path(os.getenv("WORKSPACE_DIR", BASE_DIR / "workspace"))
API_PREFIX = "/api"
COLMAP_BINARY = os.getenv("COLMAP_BINARY", "colmap")
HOST = os.getenv("BACKEND_HOST", "0.0.0.0")
PORT = int(os.getenv("BACKEND_PORT", "8081"))
FIXTURE_TEMPLATES = ("gondola", "wall_bay")
ALLOWED_IMAGE_SUFFIXES: frozenset[str] = frozenset({".jpg", ".jpeg", ".png"})
_JOB_ID_RE = re.compile(r"[a-zA-Z0-9][a-zA-Z0-9_\-]*$")
DEFAULT_LAYOUT_CONFIG = {
    "fixture_template": "gondola",
    "reference_width_m": None,
    "reference_depth_m": None,
    "shelf_height_offsets_m": {},
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)

app = FastAPI(title="Digital Twin Shelf Mapper", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
WORKSPACE_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/workspace", StaticFiles(directory=WORKSPACE_DIR), name="workspace")


@app.on_event("startup")
async def _startup() -> None:
    """Pre-warm the DepthAnything V2 model on the main thread at startup.

    Background tasks run in a thread pool; loading a PyTorch model for the
    first time there can race with MPS initialisation.  Loading it once here,
    synchronously, before any request arrives avoids that.
    """
    try:
        _load_depth_model()
        logger.info("DepthAnything V2 pre-warmed at startup.")
    except Exception as exc:
        logger.warning("Could not pre-warm depth model at startup: %s", exc)

# In-memory job analysis status: "idle" | "analyzing" | "done" | "error"
_job_status: dict[str, str] = {}
_job_error: dict[str, str] = {}


def _public_workspace_path(path: Path) -> str:
    return f"/workspace/{path.relative_to(WORKSPACE_DIR).as_posix()}"


def _layout_config_path(job_dir: Path) -> Path:
    return job_dir / "layout_config.json"


def _optional_positive_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    number = float(value)
    if number <= 0:
        raise ValueError("Layout calibration values must be positive numbers.")
    return number


def _normalize_layout_config(payload: dict[str, Any] | None) -> dict:
    payload = payload or {}
    template = str(payload.get("fixture_template") or DEFAULT_LAYOUT_CONFIG["fixture_template"])
    if template not in FIXTURE_TEMPLATES:
        raise ValueError(f"Unsupported fixture template '{template}'.")

    raw_offsets = payload.get("shelf_height_offsets_m") or {}
    offsets: dict[str, float] = {}
    for shelf_id, offset in raw_offsets.items():
        if offset in (None, ""):
            continue
        offsets[str(shelf_id)] = float(offset)

    return {
        "fixture_template": template,
        "reference_width_m": _optional_positive_float(payload.get("reference_width_m")),
        "reference_depth_m": _optional_positive_float(payload.get("reference_depth_m")),
        "shelf_height_offsets_m": offsets,
    }


def _load_layout_config(job_dir: Path) -> dict:
    config_path = _layout_config_path(job_dir)
    if not config_path.exists():
        return dict(DEFAULT_LAYOUT_CONFIG)
    try:
        payload = json.loads(config_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Invalid layout_config.json for job '{job_dir.name}': {exc}") from exc
    return _normalize_layout_config(payload)


def _save_layout_config(job_dir: Path, payload: dict[str, Any] | None) -> dict:
    config = _normalize_layout_config(payload)
    config_path = _layout_config_path(job_dir)
    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
    return config


def _text_model_exists(job_dir: Path) -> bool:
    text_dir = job_dir / "text"
    required = ("cameras.txt", "images.txt", "points3D.txt")
    return all((text_dir / name).exists() for name in required)


def _image_count(job_dir: Path) -> int:
    images_dir = job_dir / "images"
    if not images_dir.exists():
        return 0
    return sum(1 for path in images_dir.iterdir() if path.suffix.lower() in {".jpg", ".jpeg", ".png"})


def _colmap_available() -> bool:
    return shutil.which(COLMAP_BINARY) is not None


def _job_metadata(job_dir: Path) -> dict:
    has_text_model = _text_model_exists(job_dir)
    results_path = job_dir / "results.json"
    layout_config = _load_layout_config(job_dir)
    image_count = _image_count(job_dir)
    status = "empty"
    if has_text_model and results_path.exists():
        status = "ready"
    elif has_text_model:
        status = "ready_to_analyze"
    elif image_count > 0:
        status = "needs_reconstruction"

    return {
        "job_id": job_dir.name,
        "status": status,
        "image_count": image_count,
        "has_text_model": has_text_model,
        "has_results": results_path.exists(),
        "results_url": _public_workspace_path(results_path) if results_path.exists() else None,
        "colmap_available": _colmap_available(),
        "layout_config": layout_config,
    }


def discover_jobs() -> list[dict]:
    if not WORKSPACE_DIR.exists():
        return []
    jobs = [_job_metadata(path) for path in WORKSPACE_DIR.iterdir() if path.is_dir()]
    return sorted(jobs, key=lambda item: item["job_id"])


def _run_colmap_pipeline(job_dir: Path) -> Path:
    if not _colmap_available():
        raise RuntimeError(
            f"COLMAP binary '{COLMAP_BINARY}' is unavailable. Local development can skip COLMAP "
            "by placing cameras.txt, images.txt, and points3D.txt under the job's text/ directory."
        )

    images_dir = job_dir / "images"
    if not images_dir.exists():
        raise RuntimeError("The job has no images/ directory, so COLMAP reconstruction cannot run.")

    sparse_dir = job_dir / "sparse"
    output_dir = sparse_dir / "0"
    text_dir = job_dir / "text"
    database_path = job_dir / "database.db"
    sparse_dir.mkdir(parents=True, exist_ok=True)
    text_dir.mkdir(parents=True, exist_ok=True)

    commands = [
        [
            COLMAP_BINARY,
            "feature_extractor",
            "--database_path",
            str(database_path),
            "--image_path",
            str(images_dir),
        ],
        [
            COLMAP_BINARY,
            "exhaustive_matcher",
            "--database_path",
            str(database_path),
        ],
        [
            COLMAP_BINARY,
            "mapper",
            "--database_path",
            str(database_path),
            "--image_path",
            str(images_dir),
            "--output_path",
            str(sparse_dir),
        ],
        [
            COLMAP_BINARY,
            "model_converter",
            "--input_path",
            str(output_dir),
            "--output_path",
            str(text_dir),
            "--output_type",
            "TXT",
        ],
    ]

    _COLMAP_TIMEOUTS: dict[str, int] = {
        "feature_extractor": 600,
        "exhaustive_matcher": 3600,
        "mapper": 1800,
        "model_converter": 300,
    }

    for command in commands:
        subcommand = command[1]
        timeout = _COLMAP_TIMEOUTS.get(subcommand, 900)
        try:
            subprocess.run(command, check=True, timeout=timeout)
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(
                f"COLMAP '{subcommand}' timed out after {timeout}s. "
                "Reduce image count or increase the timeout in _run_colmap_pipeline."
            ) from exc

    return text_dir


def _ensure_text_model(job_dir: Path) -> Path:
    if _text_model_exists(job_dir):
        return job_dir / "text"
    return _run_colmap_pipeline(job_dir)


def _results_payload(
    job_id: str,
    shelves: list[dict],
    products: list[dict],
    reconstruction,
    rack: dict | None,
    layout: dict,
    labeling: dict,
    dense_mesh_url: str | None = None,
    class_records: dict | None = None,
) -> dict:
    return {
        "job_id": job_id,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "summary": {
            "image_count": len(reconstruction.images),
            "point_count": len(reconstruction.points3d),
            "shelf_count": len(shelves),
            "product_count": len(products),
            "class_count": (class_records or {}).get("class_count", 0),
        },
        "layout": layout,
        "labeling": labeling,
        "class_records": class_records,
        "rack": rack,
        "shelves": shelves,
        "products": products,
        "dense_mesh_url": dense_mesh_url,
    }


def _load_results_payload(job_id: str) -> dict:
    results_path = WORKSPACE_DIR / job_id / "results.json"
    if not results_path.exists():
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' has no results.json.")
    return json.loads(results_path.read_text(encoding="utf-8"))


def analyze_job(job_id: str) -> dict:
    job_dir = WORKSPACE_DIR / job_id
    if not job_dir.exists():
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' does not exist.")

    try:
        layout_config = _load_layout_config(job_dir)
        text_dir = _ensure_text_model(job_dir)
        reconstruction = load_text_model(text_dir)

        # Estimate world-up from camera orientations so detect_shelf_planes can
        # prefer horizontal planes (shelves) over vertical ones (walls/floor).
        # In COLMAP convention the camera +Y axis points DOWN in the image, so
        # rotation.T[:, 1] gives the world direction of gravity (image-down).
        # Negate it to obtain the world-UP direction.
        _down_vecs = np.array(
            [img.rotation.T[:, 1] for img in reconstruction.images.values()], dtype=float
        )
        hint_up: np.ndarray | None = None
        if len(_down_vecs) >= 2:
            _mean = _down_vecs.mean(axis=0)
            _signs = np.where(_down_vecs @ _mean >= 0, 1.0, -1.0)
            _raw = (_down_vecs * _signs[:, None]).mean(axis=0)
            _n = float(np.linalg.norm(_raw))
            hint_up = -(_raw / _n) if _n > 1e-6 else None

        # Dense reconstruction: DepthAnything V2 per-image depth + Open3D TSDF.
        # Falls back to sparse COLMAP points automatically if anything fails.
        try:
            points = run_dense_pipeline(job_dir, reconstruction)
        except Exception as _dense_exc:
            import traceback as _tb
            logger.warning(
                "Dense pipeline error — falling back to sparse COLMAP points.\n%s",
                _tb.format_exc(),
            )
            points = np.array(
                [pt.xyz for pt in reconstruction.points3d.values()], dtype=float
            )

        # Prefer mesh-based shelf detection (face normals are precise)
        # when a dense mesh is available; fall back to RANSAC on sparse points.
        mesh_path = job_dir / "dense" / "mesh.ply"
        if mesh_path.exists() and hint_up is not None:
            try:
                shelves = detect_shelf_planes_from_mesh(str(mesh_path), hint_up)
            except Exception as exc:
                import traceback as _tb
                logger.warning(
                    "Mesh-based shelf detection unavailable — falling back to RANSAC.\n%s",
                    _tb.format_exc(),
                )
                shelves = []
            if not shelves:
                logger.warning("Mesh-based shelf detection found nothing; trying RANSAC.")
                shelves = detect_shelf_planes(points, hint_up=hint_up)
        else:
            shelves = detect_shelf_planes(points, hint_up=hint_up)

        detector = ObjectDetector()
        detections = detector.detect_job(job_dir)
        identifier = ProductIdentifier()
        detections, identification_metadata = identifier.enrich_detections(job_dir, detections)
        class_records = build_class_records(detections, identifier.class_registry)
        products = project_detections_to_shelves(reconstruction, shelves, detections)
        shelves, products, layout = apply_layout_model(shelves, products, layout_config)

        for product in products:
            crop_path = product.get("crop_path")
            if not crop_path:
                product["crop_url"] = None
                continue
            path = Path(crop_path)
            if path.is_absolute():
                public_path = _public_workspace_path(path)
            else:
                public_path = _public_workspace_path((job_dir / path).resolve())
            product["crop_url"] = public_path

        shelves = attach_products_to_shelves(shelves, products)
        rack = build_rack_geometry(shelves, fixture_template=layout["fixture_template"])
        labeling = detector.labeling_metadata()
        labeling.update(
            {
                "detector_label_source": labeling.get("label_source"),
                "detector_product_identity_labels": labeling.get("product_identity_labels"),
                "product_identity_labels": identification_metadata["product_identity_labels"]
                or bool(labeling.get("product_identity_labels")),
                "named_product_count": identification_metadata["named_product_count"],
                "unnamed_product_count": identification_metadata["unnamed_product_count"],
                "product_name_source": identification_metadata["product_name_source"],
                "note": identification_metadata["note"],
                "pipeline": identification_metadata.get("pipeline"),
            }
        )
        # Prefer the view-optimised mesh (80k triangles) for the browser
        _view_mesh = job_dir / "dense" / "mesh_view.ply"
        _full_mesh = job_dir / "dense" / "mesh.ply"
        dense_mesh_url = _public_workspace_path(
            _view_mesh if _view_mesh.exists() else _full_mesh
        ) if (_view_mesh.exists() or _full_mesh.exists()) else None
        results = _results_payload(
            job_id,
            shelves,
            products,
            reconstruction,
            rack,
            layout,
            labeling,
            dense_mesh_url=dense_mesh_url,
            class_records=class_records,
        )
        results_path = job_dir / "results.json"
        results_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
        return results
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get(f"{API_PREFIX}/health")
def health() -> dict:
    return {
        "status": "ok",
        "workspace_dir": str(WORKSPACE_DIR),
        "colmap_available": _colmap_available(),
        "jobs": discover_jobs(),
    }


@app.get(f"{API_PREFIX}/jobs")
def list_jobs() -> dict:
    return {"jobs": discover_jobs()}


@app.get(f"{API_PREFIX}/jobs/{{job_id}}")
def get_job(job_id: str) -> dict:
    job_dir = WORKSPACE_DIR / job_id
    if not job_dir.exists():
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' does not exist.")
    job = _job_metadata(job_dir)
    results_path = job_dir / "results.json"
    if results_path.exists():
        job["results"] = json.loads(results_path.read_text(encoding="utf-8"))
    return job


@app.get(f"{API_PREFIX}/jobs/{{job_id}}/layout")
def get_job_layout(job_id: str) -> dict:
    job_dir = WORKSPACE_DIR / job_id
    if not job_dir.exists():
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' does not exist.")
    return {
        "job_id": job_id,
        "layout_config": _load_layout_config(job_dir),
        "available_templates": FIXTURE_TEMPLATES,
    }


@app.put(f"{API_PREFIX}/jobs/{{job_id}}/layout")
def update_job_layout(job_id: str, payload: dict) -> dict:
    job_dir = WORKSPACE_DIR / job_id
    if not job_dir.exists():
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' does not exist.")
    try:
        layout_config = _save_layout_config(job_dir, payload)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {
        "job_id": job_id,
        "layout_config": layout_config,
        "available_templates": FIXTURE_TEMPLATES,
    }


@app.get(f"{API_PREFIX}/results/{{job_id}}")
def get_results(job_id: str) -> dict:
    return _load_results_payload(job_id)


@app.get(f"{API_PREFIX}/jobs/{{job_id}}/shelves")
def get_job_shelves(job_id: str) -> dict:
    results = _load_results_payload(job_id)
    return {
        "job_id": job_id,
        "shelves": results.get("shelves", []),
    }


@app.get(f"{API_PREFIX}/jobs/{{job_id}}/shelves/{{shelf_id}}")
def get_job_shelf(job_id: str, shelf_id: str) -> dict:
    results = _load_results_payload(job_id)
    for shelf in results.get("shelves", []):
        if shelf["id"] == shelf_id:
            return {
                "job_id": job_id,
                "shelf": shelf,
            }
    raise HTTPException(status_code=404, detail=f"Shelf '{shelf_id}' was not found in job '{job_id}'.")


@app.post(f"{API_PREFIX}/jobs")
def create_job(payload: dict) -> dict:
    job_id = str(payload.get("job_id") or "").strip()
    if not job_id:
        raise HTTPException(status_code=400, detail="job_id is required.")
    if not _JOB_ID_RE.fullmatch(job_id):
        raise HTTPException(
            status_code=400,
            detail="job_id must start with a letter or digit and contain only letters, digits, hyphens, and underscores.",
        )
    job_dir = WORKSPACE_DIR / job_id
    if job_dir.exists():
        raise HTTPException(status_code=409, detail=f"Job '{job_id}' already exists.")
    (job_dir / "images").mkdir(parents=True, exist_ok=True)
    return _job_metadata(job_dir)


@app.post(f"{API_PREFIX}/jobs/{{job_id}}/images")
async def upload_images(job_id: str, files: list[UploadFile]) -> dict:
    job_dir = WORKSPACE_DIR / job_id
    if not job_dir.exists():
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' does not exist.")
    images_dir = job_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    saved: list[str] = []
    for upload_file in files:
        filename = Path(upload_file.filename or "").name
        suffix = Path(filename).suffix.lower()
        if suffix not in ALLOWED_IMAGE_SUFFIXES:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type '{suffix}'. Only .jpg, .jpeg, and .png are accepted.",
            )
        (images_dir / filename).write_bytes(await upload_file.read())
        saved.append(filename)
    return {
        "job_id": job_id,
        "uploaded": saved,
        "image_count": _image_count(job_dir),
    }


@app.post(f"{API_PREFIX}/reconstruct/{{job_id}}")
def reconstruct(job_id: str) -> dict:
    job_dir = WORKSPACE_DIR / job_id
    if not job_dir.exists():
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' does not exist.")
    try:
        text_dir = _run_colmap_pipeline(job_dir)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"job_id": job_id, "text_dir": str(text_dir), "status": "reconstructed"}


@app.get(f"{API_PREFIX}/jobs/{{job_id}}/analyze-status")
def get_analyze_status(job_id: str) -> dict:
    status = _job_status.get(job_id, "idle")
    return {
        "job_id": job_id,
        "status": status,
        "error": _job_error.get(job_id) if status == "error" else None,
    }


def _run_analyze_background(job_id: str) -> None:
    try:
        analyze_job(job_id)
        _job_status[job_id] = "done"
        _job_error.pop(job_id, None)
    except Exception as exc:
        _job_status[job_id] = "error"
        _job_error[job_id] = str(exc)


@app.post(f"{API_PREFIX}/analyze/{{job_id}}")
def analyze(job_id: str, background_tasks: BackgroundTasks) -> dict:
    job_dir = WORKSPACE_DIR / job_id
    if not job_dir.exists():
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' does not exist.")
    if _job_status.get(job_id) == "analyzing":
        return {"job_id": job_id, "status": "analyzing"}
    _job_status[job_id] = "analyzing"
    _job_error.pop(job_id, None)
    background_tasks.add_task(_run_analyze_background, job_id)
    return {"job_id": job_id, "status": "analyzing"}


if __name__ == "__main__":
    uvicorn.run(app, host=HOST, port=PORT)
