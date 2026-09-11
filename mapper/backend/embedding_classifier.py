"""
DINOv2-based product matcher backed by a file-based knowledge base.

The knowledge base supports both flat and multi-view layouts:

- backend/knowledge_base/<product>/*.jpg
- backend/knowledge_base/<product>/front/*.jpg
- backend/knowledge_base/<product>/angle_left/*.jpg
- backend/knowledge_base/<product>/angle_right/*.jpg

Optional aliases can be defined in backend/knowledge_base/catalog.json.
"""

from __future__ import annotations

import shutil
import threading
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

try:
    import torch
    from PIL import Image
    from sklearn.neighbors import NearestNeighbors
    from transformers import AutoImageProcessor, AutoModel

    _AVAILABLE = True
except ImportError:
    _AVAILABLE = False

from product_catalog import DEFAULT_KB_DIR, ProductCatalog, SUPPORTED_IMAGE_SUFFIXES

KB_RESCAN_INTERVAL_SECONDS = 30.0

DINO_MODEL = "facebook/dinov2-base"
N_NEIGHBORS = 5
MIN_EXACT_CONFIDENCE = 0.60
MIN_EXACT_BEST_SIMILARITY = 0.72
MIN_EXACT_MARGIN = 0.05


class EmbeddingClassifier:
    """Lazy DINOv2 catalog matcher for exact product-name recovery."""

    def __init__(self, knowledge_base_dir: Path | None = None) -> None:
        self.enabled = _AVAILABLE
        self.kb_dir = (knowledge_base_dir or DEFAULT_KB_DIR)
        self.kb_dir.mkdir(parents=True, exist_ok=True)
        self._processor = None
        self._model = None
        self._device = None
        self._model_lock = threading.Lock()
        self._knn: NearestNeighbors | None = None
        self._classes: list[str] = []
        self._reference_paths: list[Path] = []
        self._catalog = ProductCatalog(self.kb_dir)
        self._kb_mtime: float = 0.0
        self._last_mtime_check: float = 0.0

    def classify(self, image: "Image.Image") -> dict:
        """Return catalog candidates for the given crop image."""
        if not self.enabled:
            return _miss("embedding_classifier unavailable (torch/transformers not installed)")

        self._ensure_index()
        if self._knn is None:
            return _miss("knowledge base is empty")

        try:
            vec = self._embed(image)
        except Exception as exc:
            return _miss(f"embedding failed: {exc}")

        neighbor_count = min(max(N_NEIGHBORS, 12), len(self._classes))
        distances, indices = self._knn.kneighbors([vec], n_neighbors=neighbor_count)
        candidates = self._score_candidates(distances[0], indices[0])
        if not candidates:
            return _miss("no catalog candidates were produced")

        top = candidates[0]
        second = candidates[1] if len(candidates) > 1 else None
        exact = self._is_exact_match(top, second)
        if exact:
            return {
                "product_name": top["product_name"],
                "candidate": top["product_name"],
                "exact_match": True,
                "confidence": round(float(top["confidence"]), 4),
                "reason": (
                    f"Matched via DINOv2 catalog similarity using {top['support_count']} "
                    f"nearest references across {top['reference_count']} catalog images."
                ),
                "source": "embedding_classifier",
                "candidates": candidates[:5],
            }

        payload = _miss(f"top catalog candidate '{top['product_name']}' was not distinct enough")
        payload["candidate"] = top["product_name"]
        payload["confidence"] = round(float(top["confidence"]), 4)
        payload["candidates"] = candidates[:5]
        return payload

    def seed_from_crop(self, product_name: str, crop_path: Path) -> None:
        """Copy a crop into the knowledge base under product_name (idempotent)."""
        if not crop_path.exists():
            return
        dest_dir = self.kb_dir / product_name / "auto"
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / crop_path.name
        if not dest.exists():
            shutil.copy2(crop_path, dest)
            self._knn = None

    def product_count(self) -> int:
        self._catalog = ProductCatalog(self.kb_dir)
        return len(self._catalog.products_with_references())

    def _load_model(self) -> None:
        if self._model is not None:
            return
        with self._model_lock:
            if self._model is not None:
                return
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self._processor = AutoImageProcessor.from_pretrained(DINO_MODEL)
            self._model = AutoModel.from_pretrained(DINO_MODEL).to(self._device)
            self._model.eval()

    def _embed(self, image: "Image.Image") -> np.ndarray:
        self._load_model()
        inputs = self._processor(images=image.convert("RGB"), return_tensors="pt").to(self._device)
        with torch.no_grad():
            out = self._model(**inputs)
            emb = out.last_hidden_state[:, 0, :]
        emb = torch.nn.functional.normalize(emb, dim=-1)
        return emb.squeeze().cpu().numpy()

    def _mtime(self) -> float:
        mtimes: list[float] = []
        if self._catalog.catalog_path.exists():
            mtimes.append(self._catalog.catalog_path.stat().st_mtime)
        for path in self.kb_dir.rglob("*"):
            if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_SUFFIXES:
                mtimes.append(path.stat().st_mtime)
        return max(mtimes, default=0.0)

    def _ensure_index(self) -> None:
        # Walking the knowledge base to compare mtimes is far too expensive to
        # repeat for every crop in a large job; only re-check periodically.
        # invalidate_index() forces the next call to re-scan immediately.
        now = time.monotonic()
        if (
            self._knn is not None
            and self._last_mtime_check > 0.0
            and (now - self._last_mtime_check) < KB_RESCAN_INTERVAL_SECONDS
        ):
            return
        current = self._mtime()
        self._last_mtime_check = now
        if self._knn is not None and current <= self._kb_mtime:
            return
        self._rebuild_index()
        self._kb_mtime = current

    def invalidate_index(self) -> None:
        self._last_mtime_check = 0.0

    def _rebuild_index(self) -> None:
        self._catalog = ProductCatalog(self.kb_dir)
        classes: list[str] = []
        reference_paths: list[Path] = []
        vecs: list[np.ndarray] = []

        for product in self._catalog.products_with_references():
            for img_path in product.reference_images:
                try:
                    with Image.open(img_path) as img:
                        vec = self._embed(img)
                    classes.append(product.name)
                    reference_paths.append(img_path)
                    vecs.append(vec)
                except Exception:
                    continue

        if not vecs:
            self._knn = None
            self._classes = []
            self._reference_paths = []
            return

        knn = NearestNeighbors(metric="cosine", n_neighbors=min(max(N_NEIGHBORS, 12), len(vecs)))
        knn.fit(vecs)
        self._knn = knn
        self._classes = classes
        self._reference_paths = reference_paths

    def _score_candidates(self, distances: np.ndarray, indices: np.ndarray) -> list[dict]:
        neighbor_total = max(1, len(indices))
        aggregates: dict[str, dict[str, float | int]] = defaultdict(
            lambda: {
                "best_similarity": 0.0,
                "similarity_sum": 0.0,
                "weight_sum": 0.0,
                "support_count": 0,
            }
        )
        total_weight = 0.0

        for distance, index in zip(distances, indices):
            similarity = max(0.0, 1.0 - float(distance))
            weight = similarity * similarity
            product_name = self._classes[int(index)]
            bucket = aggregates[product_name]
            bucket["best_similarity"] = max(float(bucket["best_similarity"]), similarity)
            bucket["similarity_sum"] = float(bucket["similarity_sum"]) + similarity
            bucket["weight_sum"] = float(bucket["weight_sum"]) + weight
            bucket["support_count"] = int(bucket["support_count"]) + 1
            total_weight += weight

        candidates: list[dict] = []
        for product_name, bucket in aggregates.items():
            support_count = int(bucket["support_count"])
            mean_similarity = float(bucket["similarity_sum"]) / float(support_count)
            weight_share = float(bucket["weight_sum"]) / total_weight if total_weight > 0 else 0.0
            support_ratio = support_count / float(neighbor_total)
            confidence = (
                0.45 * float(bucket["best_similarity"])
                + 0.25 * mean_similarity
                + 0.20 * weight_share
                + 0.10 * support_ratio
            )

            reference_count = len(self._catalog.reference_images_for(product_name))
            candidates.append(
                {
                    "product_name": product_name,
                    "confidence": round(max(0.0, min(0.99, confidence)), 4),
                    "best_similarity": round(float(bucket["best_similarity"]), 4),
                    "mean_similarity": round(mean_similarity, 4),
                    "weight_share": round(weight_share, 4),
                    "support_count": support_count,
                    "support_ratio": round(support_ratio, 4),
                    "reference_count": reference_count,
                }
            )

        candidates.sort(
            key=lambda item: (
                float(item["confidence"]),
                float(item["best_similarity"]),
                int(item["support_count"]),
            ),
            reverse=True,
        )
        return candidates

    def _is_exact_match(self, top: dict, second: dict | None) -> bool:
        if float(top.get("best_similarity") or 0.0) < MIN_EXACT_BEST_SIMILARITY:
            return False
        if float(top.get("confidence") or 0.0) < MIN_EXACT_CONFIDENCE:
            return False

        if second is None:
            return True

        margin = float(top.get("confidence") or 0.0) - float(second.get("confidence") or 0.0)
        return margin >= MIN_EXACT_MARGIN or float(top.get("support_ratio") or 0.0) >= 0.55


def _miss(reason: str) -> dict:
    return {
        "product_name": None,
        "candidate": None,
        "exact_match": False,
        "confidence": 0.0,
        "reason": reason,
        "source": "embedding_classifier",
        "candidates": [],
    }
