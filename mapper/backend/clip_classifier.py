"""
CLIP/SigLIP zero-shot product matcher.

Scores a crop two ways:
  1. crop image  ↔  catalog reference images (image-image similarity)
  2. crop image  ↔  catalog product names    (image-text similarity, the
     zero-shot lane — works even when a product has no reference images yet)

Each candidate's confidence is the max of the two lanes, so a product wins
either by visual match or by name match. The classifier is consumed by
ProductIdentifier as an extra fallback before the VLM and as a candidate
source for the VLM's prompt.
"""

from __future__ import annotations

import os
import threading
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

try:
    import torch
    from PIL import Image
    from transformers import AutoModel, AutoProcessor

    _AVAILABLE = True
except ImportError:
    _AVAILABLE = False

from product_catalog import DEFAULT_KB_DIR, ProductCatalog, SUPPORTED_IMAGE_SUFFIXES

KB_RESCAN_INTERVAL_SECONDS = 30.0

CLIP_SOURCE = "clip_classifier"

# SigLIP-2 base. Swap via CLIP_MODEL env var (e.g. openai/clip-vit-large-patch14
# for vanilla CLIP). Anything that exposes get_image_features / get_text_features
# through transformers will work.
DEFAULT_MODEL = "google/siglip2-base-patch16-224"

PROMPT_TEMPLATES: tuple[str, ...] = (
    "a photo of {}",
    "a package of {}",
    "a product labeled {} on a grocery shelf",
    "{} packaging",
)


class CLIPClassifier:
    """Lazy CLIP/SigLIP catalog matcher. Disabled gracefully when torch or
    transformers is missing, or when the knowledge base is empty."""

    def __init__(self, knowledge_base_dir: Path | None = None) -> None:
        self.enabled = _AVAILABLE and os.getenv("CLIP_CLASSIFIER_ENABLED", "1").strip().lower() not in {
            "0",
            "false",
            "no",
            "off",
        }
        self.kb_dir = knowledge_base_dir or DEFAULT_KB_DIR
        self.kb_dir.mkdir(parents=True, exist_ok=True)
        self.model_name = os.getenv("CLIP_MODEL", DEFAULT_MODEL).strip() or DEFAULT_MODEL
        self.min_confidence = float(os.getenv("CLIP_MIN_CONFIDENCE", "0.60"))
        self.min_exact_confidence = float(os.getenv("CLIP_MIN_EXACT_CONFIDENCE", "0.76"))
        self.min_exact_margin = float(os.getenv("CLIP_MIN_EXACT_MARGIN", "0.08"))

        self._catalog = ProductCatalog(self.kb_dir)
        self._processor = None
        self._model = None
        self._device = None
        self._model_lock = threading.Lock()
        self._kb_mtime: float = 0.0
        self._last_mtime_check: float = 0.0

        # Per-product averaged text vectors (rows aligned with self._product_names)
        self._product_names: list[str] = []
        self._text_vectors: np.ndarray | None = None  # [n_products, dim]
        # Per-reference-image vectors with parallel product-name list
        self._ref_product_names: list[str] = []
        self._ref_vectors: np.ndarray | None = None  # [n_refs, dim]

    def product_count(self) -> int:
        self._catalog = ProductCatalog(self.kb_dir)
        return len(self._catalog.product_names())

    def classify(self, image: "Image.Image") -> dict:
        if not self.enabled:
            return _miss("clip_classifier unavailable (transformers/torch missing or disabled)")

        self._ensure_index()
        if self._text_vectors is None or not self._product_names:
            return _miss("clip catalog index is empty")

        try:
            img_vec = self._embed_image(image)
        except Exception as exc:
            return _miss(f"clip image embedding failed: {exc}")

        text_sims = self._text_vectors @ img_vec  # [n_products]
        ref_max: dict[str, float] = {}
        if self._ref_vectors is not None and len(self._ref_product_names) > 0:
            ref_sims = self._ref_vectors @ img_vec
            for name, sim in zip(self._ref_product_names, ref_sims):
                value = float(sim)
                if value > ref_max.get(name, -1.0):
                    ref_max[name] = value

        candidates: list[dict] = []
        for index, name in enumerate(self._product_names):
            text_sim = float(text_sims[index])
            ref_sim = ref_max.get(name, 0.0)
            combined = max(text_sim, ref_sim)
            candidates.append(
                {
                    "product_name": name,
                    "text_similarity": round(text_sim, 4),
                    "image_similarity": round(ref_sim, 4),
                    "raw_similarity": round(combined, 4),
                }
            )

        candidates.sort(key=lambda item: item["raw_similarity"], reverse=True)
        if not candidates:
            return _miss("no clip candidates were produced")

        top_sim = candidates[0]["raw_similarity"]
        min_sim = candidates[-1]["raw_similarity"]
        denom = max(1e-6, top_sim - min_sim)
        for index, item in enumerate(candidates):
            spread = (item["raw_similarity"] - min_sim) / denom
            absolute = max(0.0, min(1.0, (item["raw_similarity"] + 1.0) / 2.0))
            item["confidence"] = round(0.6 * absolute + 0.4 * spread, 4)

        top = candidates[0]
        second = candidates[1] if len(candidates) > 1 else None
        exact = self._is_exact(top, second)
        payload = {
            "product_name": top["product_name"] if exact else None,
            "candidate": top["product_name"],
            "exact_match": exact,
            "confidence": float(top["confidence"]),
            "reason": (
                f"CLIP zero-shot matched '{top['product_name']}' "
                f"(text-sim={top['text_similarity']}, image-sim={top['image_similarity']})."
                if exact
                else f"CLIP top candidate '{top['product_name']}' was not distinct enough."
            ),
            "source": CLIP_SOURCE,
            "candidates": candidates[:5],
        }
        return payload

    def _is_exact(self, top: dict, second: dict | None) -> bool:
        if float(top.get("confidence") or 0.0) < self.min_exact_confidence:
            return False
        if second is None:
            return True
        margin = float(top.get("confidence") or 0.0) - float(second.get("confidence") or 0.0)
        return margin >= self.min_exact_margin

    def _ensure_index(self) -> None:
        # Walking the knowledge base to compare mtimes is far too expensive to
        # repeat for every crop in a large job; only re-check periodically.
        # invalidate_index() forces the next call to re-scan immediately.
        now = time.monotonic()
        if (
            self._text_vectors is not None
            and self._last_mtime_check > 0.0
            and (now - self._last_mtime_check) < KB_RESCAN_INTERVAL_SECONDS
        ):
            return
        current = self._mtime()
        self._last_mtime_check = now
        if self._text_vectors is not None and current <= self._kb_mtime:
            return
        self._rebuild_index()
        self._kb_mtime = current

    def invalidate_index(self) -> None:
        self._last_mtime_check = 0.0

    def _rebuild_index(self) -> None:
        self._catalog = ProductCatalog(self.kb_dir)
        product_names = self._catalog.product_names()
        if not product_names:
            self._product_names = []
            self._text_vectors = None
            self._ref_product_names = []
            self._ref_vectors = None
            return

        self._load_model()

        text_vecs: list[np.ndarray] = []
        kept_names: list[str] = []
        for name in product_names:
            aliases = self._catalog.aliases_for(name) or (name,)
            phrases: list[str] = []
            for alias in aliases:
                for template in PROMPT_TEMPLATES:
                    phrases.append(template.format(alias))
            try:
                vecs = self._embed_texts(phrases)
            except Exception:
                continue
            mean_vec = vecs.mean(axis=0)
            mean_vec = mean_vec / (np.linalg.norm(mean_vec) + 1e-9)
            text_vecs.append(mean_vec)
            kept_names.append(name)

        if not text_vecs:
            self._product_names = []
            self._text_vectors = None
        else:
            self._product_names = kept_names
            self._text_vectors = np.stack(text_vecs, axis=0)

        ref_images: list[Image.Image] = []
        ref_names: list[str] = []
        for product in self._catalog.products_with_references():
            for ref_path in product.reference_images:
                if ref_path.suffix.lower() not in SUPPORTED_IMAGE_SUFFIXES:
                    continue
                try:
                    img = Image.open(ref_path).convert("RGB")
                    img.load()
                except Exception:
                    continue
                ref_images.append(img)
                ref_names.append(product.name)

        if ref_images:
            try:
                self._ref_vectors = self._embed_images(ref_images, batch_size=16)
                self._ref_product_names = ref_names
            except Exception:
                self._ref_vectors = None
                self._ref_product_names = []
        else:
            self._ref_vectors = None
            self._ref_product_names = []

    def _mtime(self) -> float:
        mtimes: list[float] = []
        if self._catalog.catalog_path.exists():
            mtimes.append(self._catalog.catalog_path.stat().st_mtime)
        for path in self.kb_dir.rglob("*"):
            if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_SUFFIXES:
                mtimes.append(path.stat().st_mtime)
        return max(mtimes, default=0.0)

    def _load_model(self) -> None:
        if self._model is not None:
            return
        with self._model_lock:
            if self._model is not None:
                return
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self._processor = AutoProcessor.from_pretrained(self.model_name)
            self._model = AutoModel.from_pretrained(self.model_name).to(self._device)
            self._model.eval()

    def _embed_image(self, image: "Image.Image") -> np.ndarray:
        return self._embed_images([image])[0]

    def _embed_images(self, images: list["Image.Image"], batch_size: int = 32) -> np.ndarray:
        self._load_model()
        rgbs = [img.convert("RGB") for img in images]
        out_chunks: list[np.ndarray] = []
        for start in range(0, len(rgbs), batch_size):
            batch = rgbs[start : start + batch_size]
            inputs = self._processor(images=batch, return_tensors="pt").to(self._device)
            with torch.no_grad():
                features = self._model.get_image_features(**inputs)
            features = torch.nn.functional.normalize(features, dim=-1)
            out_chunks.append(features.cpu().numpy())
        return np.concatenate(out_chunks, axis=0)

    def _embed_texts(self, texts: list[str]) -> np.ndarray:
        self._load_model()
        inputs = self._processor(
            text=texts,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
        ).to(self._device)
        with torch.no_grad():
            features = self._model.get_text_features(**inputs)
        features = torch.nn.functional.normalize(features, dim=-1)
        return features.cpu().numpy()


def _miss(reason: str) -> dict:
    return {
        "product_name": None,
        "candidate": None,
        "exact_match": False,
        "confidence": 0.0,
        "reason": reason,
        "source": CLIP_SOURCE,
        "candidates": [],
    }
