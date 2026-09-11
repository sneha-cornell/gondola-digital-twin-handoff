from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import tempfile
import unicodedata
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from difflib import get_close_matches
from pathlib import Path

from PIL import Image

from clip_classifier import CLIPClassifier
from embedding_classifier import EmbeddingClassifier
from open_world_identifier import OPEN_WORLD_SOURCE, OpenWorldIdentifier
from product_catalog import DEFAULT_KB_DIR, ProductCatalog
from product_classes import ClassRegistry
from vlm_identifier import VLMIdentifier

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ProductPattern:
    name: str
    required_groups: tuple[tuple[str, ...], ...]
    optional_groups: tuple[tuple[str, ...], ...] = ()


IDENTIFIED_LABEL_SOURCE = "ocr_center_product_identifier"
APPLE_VISION_SOURCE = "apple_vision_ocr"
GOOGLE_CLOUD_VISION_SOURCE = "google_cloud_vision_ocr"
PADDLEOCR_SOURCE = "paddleocr"
PRODUCT_IDENTIFIER_CACHE_VERSION = 10

_DISABLE_VALUES = {"0", "false", "no", "off"}
_GENERIC_WORDS = {
    "air",
    "baked",
    "barbeque",
    "bbq",
    "bohemian",
    "cheese",
    "chips",
    "chicken",
    "corn",
    "corners",
    "crisps",
    "cut",
    "deli",
    "dipped",
    "dill",
    "family",
    "flavor",
    "flavored",
    "fried",
    "gluten",
    "grain",
    "hot",
    "heat",
    "homestyle",
    "honey",
    "kettle",
    "krinkle",
    "lightly",
    "mini",
    "multi",
    "multiseed",
    "multigrain",
    "mustard",
    "original",
    "panetini",
    "pepper",
    "pepperoncini",
    "pickle",
    "pickles",
    "pink",
    "plain",
    "popcorn",
    "potato",
    "pretzel",
    "pretzels",
    "protein",
    "real",
    "riffles",
    "salt",
    "seed",
    "seasoned",
    "sea",
    "size",
    "snack",
    "spicy",
    "sticks",
    "style",
    "sweet",
    "thins",
    "twists",
    "vinegar",
    "waffles",
}
_STOPWORDS = {
    "a",
    "and",
    "brand",
    "family",
    "free",
    "gluten",
    "large",
    "mini",
    "of",
    "real",
    "size",
    "style",
    "the",
}
_BRAND_WORDS = {
    "CRUNCHMASTER",
    "DOTS",
    "HERRS",
    "HERSHEYS",
    "HIPPEAS",
    "KETTLE",
    "NEW",
    "PIPCORN",
    "POPCHIPS",
    "POPCORNERS",
    "POPLITE",
    "REESES",
    "SNYDERS",
    "STACYS",
    "TERRA",
    "WILDE",
    "YORK",
}
_LOW_SIGNAL_TOKENS = {
    "AIR",
    "BAKED",
    "BRAND",
    "CHIPS",
    "CORN",
    "CORNERS",
    "CRISPS",
    "CUT",
    "DELI",
    "FAMILY",
    "FLAVOR",
    "FLAVORED",
    "GLUTEN",
    "HEAT",
    "HOMESTYLE",
    "HOT",
    "KETTLE",
    "KRINKLE",
    "LIGHTLY",
    "MINI",
    "ORIGINAL",
    "POTATO",
    "PRETZEL",
    "PRETZELS",
    "PROTEIN",
    "REAL",
    "SEA",
    "SALT",
    "SEASONED",
    "SEED",
    "SIZE",
    "SNACK",
    "STYLE",
    "SWEET",
}
_TOKEN_NORMALIZATIONS = {
    "ARBEOUE": "BARBEQUE",
    "ARBEOUET": "BARBEQUE",
    "ARBEOUIT": "BARBEQUE",
    "BACKYART": "BACKYARD",
    "BACKYARD": "BACKYARD",
    "BARBEOUE": "BARBEQUE",
    "BARBEOUE": "BARBEQUE",
    "BARBEOUET": "BARBEQUE",
    "BARBEQUEE": "BARBEQUE",
    "BOHEMIAN": "BOHEMIAN",
    "CORNEKO": "POPCORNERS",
    "CORNEL": "POPCORNERS",
    "CRUNCHKASTER": "CRUNCHMASTER",
    "CRUNCHMASTER": "CRUNCHMASTER",
    "DETS": "DOTS",
    "DOLS": "DOTS",
    "DOIS": "DOTS",
    "DORS": "DOTS",
    "DOTS": "DOTS",
    "DOT'S": "DOTS",
    "DIPPED": "DIPPED",
    "EHETE": "KETTLE",
    "EHTE": "KETTLE",
    "FRETZELS": "PRETZELS",
    "FRIE": "FRIED",
    "HOMESTYLE": "HOMESTYLE",
    "HONEY": "HONEY",
    "HERRS": "HERRS",
    "HERSHEY": "HERSHEYS",
    "HERSHEYS": "HERSHEYS",
    "HEAT": "HEAT",
    "IPPEAS": "HIPPEAS",
    "KETTE": "KETTLE",
    "KRINKL": "KRINKLE",
    "MALAYAN": "HIMALAYAN",
    "MULTISEED": "MULTISEED",
    "MULTSEED": "MULTISEED",
    "MUSTARD": "MUSTARD",
    "NEHE": "KETTLE",
    "NOTTE": "KETTLE",
    "NWILDE": "WILDE",
    "ORIGINAL": "ORIGINAL",
    "OPCORNERS": "POPCORNERS",
    "ОРСOВS": "POPCORN",
    "ОРСOВS": "POPCORN",
    "PAMEIIMI": "PANETINI",
    "PAMEITIMI": "PANETINI",
    "PANEIIMI": "PANETINI",
    "PANETINI": "PANETINI",
    "PIPPEAS": "HIPPEAS",
    "POФWAE": "POPLITE",
    "POPCORNE": "POPCORNERS",
    "POPCORNET": "POPCORNERS",
    "POPCORNERS": "POPCORNERS",
    "POTATOCI": "POTATO",
    "PRAEALS": "PRETZELS",
    "PREIZELS": "PRETZELS",
    "PRETZAIS": "PRETZELS",
    "PRAETZELS": "PRETZELS",
    "PRAEZALS": "PRETZELS",
    "PROTEIR": "PROTEIN",
    "RETE": "KETTLE",
    "RETZAIS": "PRETZELS",
    "RNEKO": "POPCORNERS",
    "SEASALT": "SEA",
    "STYLE": "STYLE",
    "SWEET": "SWEET",
    "SNYDERS": "SNYDERS",
    "SNYDERSS": "SNYDERS",
    "SYNDERS": "SNYDERS",
    "TERSHEN": "HERSHEYS",
    "TWISTS": "TWISTS",
    "VINEBAR": "VINEGAR",
    "&VINEBAR": "VINEGAR",
    "WETE": "KETTLE",
}
_TOKEN_VOCABULARY = sorted(
    _GENERIC_WORDS
    | {
        "bohemian",
        "backyard",
        "crunchmaster",
        "barbeque",
        "dipped",
        "dots",
        "family",
        "heat",
        "herrs",
        "hersheys",
        "hippeas",
        "himalayan",
        "homestyle",
        "honey",
        "hot",
        "kettle",
        "multiseed",
        "mustard",
        "nashville",
        "new",
        "original",
        "panetini",
        "pipcorn",
        "popchips",
        "popcorners",
        "poplite",
        "pretzels",
        "protein",
        "reeses",
        "riffles",
        "sea",
        "seed",
        "size",
        "snyders",
        "stacys",
        "sweet",
        "terra",
        "twists",
        "vinegar",
        "wilde",
        "york",
    }
)
_DISPLAY_NORMALIZATIONS = {
    "BACKYARD": "Backyard",
    "BARBEQUE": "Barbeque",
    "BBQ": "BBQ",
    "BOHEMIAN": "Bohemian",
    "CRUNCHMASTER": "Crunchmaster",
    "DIPPED": "Dipped",
    "DOTS": "Dot's",
    "HEAT": "Heat",
    "HERRS": "Herr's",
    "HERSHEYS": "Hershey's",
    "HIPPEAS": "Hippeas",
    "HIMALAYAN": "Himalayan",
    "HOMESTYLE": "Homestyle",
    "HONEY": "Honey",
    "MULTISEED": "Multi-Seed",
    "MUSTARD": "Mustard",
    "NASHVILLE": "Nashville",
    "NEW": "New",
    "ORIGINAL": "Original",
    "PANETINI": "Panetini",
    "PIPCORN": "Pipcorn",
    "POPCHIPS": "Popchips",
    "POPCORNERS": "PopCorners",
    "POPLITE": "PopLite",
    "POTATO": "Potato",
    "PRETZELS": "Pretzels",
    "PROTEIN": "Protein",
    "REESES": "Reese's",
    "SALT": "Salt",
    "SEA": "Sea",
    "SIZE": "Size",
    "SNYDERS": "Snyder's",
    "STACYS": "Stacy's",
    "SWEET": "Sweet",
    "TERRA": "Terra",
    "TWISTS": "Twists",
    "WILDE": "Wilde",
    "YORK": "York",
}

_PRODUCT_PATTERNS: tuple[ProductPattern, ...] = (
    ProductPattern("Reese's Popcorn", (("REESES",), ("POPCORN",))),
    ProductPattern("Reese's Dipped Pretzels", (("REESES",), ("PRETZELS", "PRETZEL"))),
    ProductPattern("Hershey's Dipped Pretzels", (("HERSHEYS",), ("PRETZELS", "PRETZEL"))),
    ProductPattern("Hippeas Bohemian Barbeque", (("HIPPEAS",), ("BOHEMIAN",), ("BARBEQUE", "BBQ"))),
    ProductPattern("Hippeas Barbeque", (("HIPPEAS",), ("BARBEQUE", "BBQ"))),
    ProductPattern("Snyder's Family Size Pretzel Sticks", (("SNYDERS",), ("FAMILY",), ("SIZE",), ("STICKS",))),
    ProductPattern("Snyder's Pretzel Sticks", (("SNYDERS",), ("STICKS",))),
    ProductPattern("Dot's Homestyle Pretzels Honey Mustard", (("DOTS",), ("PRETZELS", "PRETZEL"), ("HONEY",), ("MUSTARD",))),
    ProductPattern("Dot's Homestyle Pretzels BBQ", (("DOTS",), ("PRETZELS", "PRETZEL"), ("BBQ", "BARBEQUE"))),
    ProductPattern("Dot's Homestyle Pretzels", (("DOTS",), ("PRETZELS", "PRETZEL"))),
    ProductPattern("Crunchmaster Multi-Seed", (("CRUNCHMASTER",), ("MULTISEED",))),
    ProductPattern("New York Style Panetini", (("NEW",), ("YORK",), ("PANETINI",))),
    ProductPattern("Wilde Protein Chips Sea Salt & Vinegar", (("WILDE",), ("PROTEIN",), ("CHIPS",), ("SEA",), ("SALT",), ("VINEGAR",))),
    ProductPattern("Wilde Protein Chips Himalayan Pink Salt", (("WILDE",), ("PROTEIN",), ("CHIPS",), ("HIMALAYAN",), ("SALT",))),
    ProductPattern("Wilde Protein Chips Nashville Hot", (("WILDE",), ("PROTEIN",), ("CHIPS",), ("NASHVILLE",), ("HOT",))),
    ProductPattern("Wilde Protein Chips", (("WILDE",), ("PROTEIN",), ("CHIPS",))),
    ProductPattern("Popchips Sweet Heat", (("POPCHIPS",), ("SWEET",), ("HEAT",))),
    ProductPattern("Popchips BBQ", (("POPCHIPS",), ("BBQ", "BARBEQUE"))),
    ProductPattern("Popchips Sea Salt & Vinegar", (("POPCHIPS",), ("SEA",), ("SALT",), ("VINEGAR",))),
    ProductPattern("Popchips", (("POPCHIPS",),)),
    ProductPattern("PopCorners Sea Salt", (("POPCORNERS",), ("SEA",), ("SALT",))),
    ProductPattern("PopCorners", (("POPCORNERS",),)),
    ProductPattern("Kettle Brand Krinkle Cut Dill Pickle", (("KETTLE",), ("KRINKLE",), ("DILL",))),
    ProductPattern("Kettle Brand Krinkle Cut Dill Pickle", (("KRINKLE",), ("DILL",))),
    ProductPattern("Kettle Brand Pepperoncini", (("KETTLE",), ("PEPPERONCINI",))),
    ProductPattern("Kettle Brand Pepperoncini", (("PEPPERONCINI",),)),
    ProductPattern("Kettle Brand Dill Pickle", (("KETTLE",), ("DILL",), ("PICKLE", "PICKLES"))),
    ProductPattern("Kettle Brand Backyard Barbeque", (("KETTLE",), ("BACKYARD",), ("BARBEQUE", "BBQ"))),
    ProductPattern("Kettle Brand Backyard Barbeque", (("BACKYARD",), ("BARBEQUE", "BBQ"))),
    ProductPattern("Kettle Brand Riffles", (("KETTLE",), ("RIFFLES",))),
    ProductPattern("Kettle Brand Riffles", (("RIFFLES",),)),
    ProductPattern("Kettle Brand Sea Salt", (("KETTLE",), ("SEA",), ("SALT",))),
)


@dataclass(frozen=True)
class OCRLine:
    text: str
    min_x: float
    min_y: float
    width: float
    height: float
    confidence: float | None = None

    @property
    def center_x(self) -> float:
        return self.min_x + 0.5 * self.width

    @property
    def center_y(self) -> float:
        return self.min_y + 0.5 * self.height

    @property
    def area(self) -> float:
        return self.width * self.height


class ProductIdentifier:
    def __init__(self) -> None:
        enabled_flag = os.getenv("PRODUCT_NAME_OCR", "1").strip().lower()
        self.enabled = enabled_flag not in _DISABLE_VALUES
        self.min_image_side = int(os.getenv("PRODUCT_NAME_MIN_SIDE", "140"))
        self.max_analysis_side = int(os.getenv("PRODUCT_NAME_MAX_SIDE", "1400"))
        self.timeout_seconds = float(os.getenv("PRODUCT_NAME_TIMEOUT_SECONDS", "15"))
        self.confidence_threshold = float(os.getenv("PRODUCT_NAME_CONFIDENCE", "0.62"))
        self.max_ocr_workers = max(1, int(os.getenv("PRODUCT_NAME_OCR_MAX_WORKERS", "6")))
        self.visual_only_confidence_threshold = float(
            os.getenv("PRODUCT_NAME_VISUAL_ONLY_CONFIDENCE", "0.75")
        )
        self.visual_only_similarity_threshold = float(
            os.getenv("PRODUCT_NAME_VISUAL_ONLY_BEST_SIMILARITY", "0.95")
        )
        # The DINOv2 catalog matcher is the highest-yield naming stage, so it is
        # on by default; set PRODUCT_NAME_EMBEDDING_FALLBACK=0 to opt out.
        fallback_flag = os.getenv("PRODUCT_NAME_EMBEDDING_FALLBACK", "1").strip().lower()
        self.embedding_fallback_enabled = fallback_flag not in _DISABLE_VALUES
        self._vision_binary_path: Path | None = None
        self._engine_reason: str | None = None
        self._engine = self._detect_engine() if self.enabled else None
        if not self.enabled:
            self._engine_reason = "disabled via PRODUCT_NAME_OCR"
        self._catalog = ProductCatalog()
        self._class_registry = ClassRegistry(catalog=self._catalog)
        self._embedding_classifier = EmbeddingClassifier()
        self._clip_classifier = CLIPClassifier()
        self._catalog_aliases = self._load_catalog_aliases()
        self._vlm = VLMIdentifier(self._catalog)
        self._open_world = OpenWorldIdentifier(self._catalog, self._vlm)

    @property
    def class_registry(self) -> ClassRegistry:
        return self._class_registry

    @property
    def engine(self) -> str | None:
        return self._engine

    def enrich_detections(self, job_dir: Path, detections: list[dict]) -> tuple[list[dict], dict]:
        if not detections:
            return detections, self.summary_metadata(0, 0)

        stage_stats: dict[str, dict[str, int]] = {
            stage: {"attempted": 0, "named": 0}
            for stage in ("ocr", "embedding", "clip", "vlm", "open_world")
        }
        for stage in self.pipeline_status():
            if not stage["enabled"]:
                logger.warning(
                    "Identification stage '%s' is disabled: %s",
                    stage["stage"],
                    stage.get("reason"),
                )

        cache_dir = job_dir / "product_names"
        cache_dir.mkdir(parents=True, exist_ok=True)

        # Separate detections that need OCR from those that don't.
        # Build a list of (index, updated_dict, resolved_path, cache_path) for OCR candidates.
        enriched: list[dict | None] = [None] * len(detections)
        ocr_queue: list[tuple[int, dict, Path, Path | None, Path]] = []

        for index, detection in enumerate(detections):
            updated = dict(detection)

            if updated.get("product_identity_label"):
                updated.setdefault("product_identity_source", updated.get("label_source"))
                enriched[index] = updated
                continue

            crop_path = updated.get("crop_path")
            if not crop_path:
                enriched[index] = updated
                continue

            resolved_crop_path = Path(crop_path)
            if not resolved_crop_path.is_absolute():
                resolved_crop_path = (job_dir / resolved_crop_path).resolve()
            if not resolved_crop_path.exists():
                enriched[index] = updated
                continue

            if not self.enabled or self._engine is None or not self._is_large_enough(resolved_crop_path):
                enriched[index] = updated
                continue

            cache_path = cache_dir / f"{updated['id']}.json"
            cached_result = self._load_cache(cache_path)
            if cached_result is not None:
                # Cache hit — apply immediately without queuing.
                stage_stats["ocr"]["attempted"] += 1
                self._apply_ocr_result(updated, cached_result)
                if updated.get("product_identity_label"):
                    stage_stats["ocr"]["named"] += 1
                enriched[index] = updated
                continue

            resolved_raw_crop_path: Path | None = None
            raw_crop_path = updated.get("raw_crop_path")
            if raw_crop_path:
                raw_candidate = Path(raw_crop_path)
                if not raw_candidate.is_absolute():
                    raw_candidate = (job_dir / raw_candidate).resolve()
                if raw_candidate.exists():
                    resolved_raw_crop_path = raw_candidate

            ocr_queue.append((index, updated, resolved_crop_path, resolved_raw_crop_path, cache_path))

        # Run all cache-miss OCR calls concurrently (Vision/Tesseract are subprocess-bound).
        if ocr_queue:
            def _run_ocr(args: tuple[int, dict, Path, Path | None, Path]) -> tuple[int, dict, dict]:
                idx, det, crop, raw_crop, cache = args
                result = self.identify_detection_crop(crop, raw_crop)
                return idx, det, result

            with ThreadPoolExecutor(max_workers=min(self.max_ocr_workers, len(ocr_queue))) as pool:
                futures = {pool.submit(_run_ocr, item): item for item in ocr_queue}
                for future in as_completed(futures):
                    idx, updated, result = future.result()
                    futures[future][4].write_text(json.dumps(result, indent=2), encoding="utf-8")
                    stage_stats["ocr"]["attempted"] += 1
                    self._apply_ocr_result(updated, result)
                    if updated.get("product_identity_label"):
                        stage_stats["ocr"]["named"] += 1
                    enriched[idx] = updated

        # Embedding fallback: for detections that OCR couldn't name, try the DINOv2
        # knowledge-base classifier. This covers products with unreadable text or
        # products not yet in the OCR catalog.
        if (
            self.embedding_fallback_enabled
            and self._embedding_classifier.enabled
            and self._embedding_classifier.product_count() > 0
        ):
            for det in enriched:
                if det is None or det.get("product_identity_label"):
                    continue
                crop_path = det.get("crop_path")
                if not crop_path:
                    continue
                resolved = Path(crop_path)
                if not resolved.exists():
                    continue
                stage_stats["embedding"]["attempted"] += 1
                try:
                    with Image.open(resolved) as img:
                        emb_result = self._embedding_classifier.classify(img.convert("RGB"))
                except Exception:
                    continue
                if emb_result.get("exact_match") and emb_result.get("product_name"):
                    det["recognized_product_name"] = emb_result["product_name"]
                    det["product_identity_confidence"] = float(emb_result.get("confidence") or 0.0)
                    det["product_identity_reason"] = emb_result.get("reason")
                    det["product_identity_source"] = emb_result.get("source")
                    det["label"] = emb_result["product_name"]
                    det["display_label"] = emb_result["product_name"]
                    det["label_source"] = IDENTIFIED_LABEL_SOURCE
                    det["product_identity_label"] = True
                    det["identity_stage"] = "embedding"
                    stage_stats["embedding"]["named"] += 1

        # CLIP/SigLIP zero-shot fallback. Runs on detections that still have no
        # confident label. Even when CLIP itself isn't confident enough to claim
        # an exact match, its top-k feeds the VLM's candidate shortlist.
        if self._clip_classifier.enabled and self._clip_classifier.product_count() > 0:
            for det in enriched:
                if det is None:
                    continue
                crop_path = det.get("crop_path")
                if not crop_path:
                    continue
                resolved = Path(crop_path)
                if not resolved.is_absolute():
                    resolved = (job_dir / resolved).resolve()
                if not resolved.exists():
                    continue
                stage_stats["clip"]["attempted"] += 1
                try:
                    with Image.open(resolved) as img:
                        clip_result = self._clip_classifier.classify(img.convert("RGB"))
                except Exception:
                    continue

                clip_candidates = clip_result.get("candidates") or []
                if clip_candidates:
                    det["clip_candidates"] = clip_candidates
                    existing = det.get("catalog_candidates") or []
                    seen = {c.get("product_name") for c in existing if isinstance(c, dict)}
                    for clip_cand in clip_candidates:
                        name = clip_cand.get("product_name")
                        if not name or name in seen:
                            continue
                        existing.append(
                            {
                                "product_name": name,
                                "confidence": clip_cand.get("confidence"),
                                "source": "clip",
                            }
                        )
                        seen.add(name)
                    det["catalog_candidates"] = existing
                    if not det.get("catalog_candidate"):
                        det["catalog_candidate"] = clip_candidates[0]["product_name"]

                if det.get("product_identity_label"):
                    continue
                if not (clip_result.get("exact_match") and clip_result.get("product_name")):
                    continue
                det["recognized_product_name"] = clip_result["product_name"]
                det["product_identity_confidence"] = float(clip_result.get("confidence") or 0.0)
                det["product_identity_reason"] = clip_result.get("reason")
                det["product_identity_source"] = clip_result.get("source")
                det["label"] = clip_result["product_name"]
                det["display_label"] = clip_result["product_name"]
                det["label_source"] = IDENTIFIED_LABEL_SOURCE
                det["product_identity_label"] = True
                det["identity_stage"] = "clip"
                stage_stats["clip"]["named"] += 1

        # VLM fallback (Option A): Anthropic vision model decides only the hard
        # cases — detections with no confident label, or whose existing
        # confidence is below the VLM threshold. Cached per detection id.
        if self._vlm.enabled:
            vlm_cache_dir = job_dir / "vlm_names"
            vlm_cache_dir.mkdir(parents=True, exist_ok=True)
            for det in enriched:
                if det is None:
                    continue
                existing_conf = float(det.get("product_identity_confidence") or 0.0)
                if det.get("product_identity_label") and existing_conf >= self._vlm.confidence_threshold:
                    continue
                crop_path = det.get("crop_path")
                if not crop_path:
                    continue
                resolved = Path(crop_path)
                if not resolved.is_absolute():
                    resolved = (job_dir / resolved).resolve()
                if not resolved.exists():
                    continue

                stage_stats["vlm"]["attempted"] += 1
                cache_path = vlm_cache_dir / f"{det['id']}.json"
                vlm_result: dict | None = None
                if cache_path.exists():
                    try:
                        vlm_result = json.loads(cache_path.read_text(encoding="utf-8"))
                    except json.JSONDecodeError:
                        vlm_result = None

                if vlm_result is None:
                    candidates: list[str] = []
                    catalog_candidate = det.get("catalog_candidate")
                    if catalog_candidate:
                        candidates.append(str(catalog_candidate))
                    for cand in det.get("catalog_candidates") or []:
                        name = cand.get("product_name") if isinstance(cand, dict) else None
                        if name and name not in candidates:
                            candidates.append(str(name))
                    if det.get("recognized_product_name") and det["recognized_product_name"] not in candidates:
                        candidates.append(str(det["recognized_product_name"]))
                    ocr_text = det.get("recognized_product_name") or ""
                    vlm_result = self._vlm.identify(resolved, ocr_text, candidates)
                    if vlm_result is not None:
                        cache_path.write_text(json.dumps(vlm_result, indent=2), encoding="utf-8")

                if not vlm_result or not vlm_result.get("exact_match"):
                    continue
                product_name = vlm_result.get("product_name")
                if not product_name:
                    continue
                det["recognized_product_name"] = product_name
                det["product_identity_confidence"] = float(vlm_result.get("confidence") or 0.0)
                det["product_identity_reason"] = vlm_result.get("reason")
                det["product_identity_source"] = vlm_result.get("source")
                det["label"] = product_name
                det["display_label"] = product_name
                det["label_source"] = IDENTIFIED_LABEL_SOURCE
                det["product_identity_label"] = True
                det["identity_stage"] = "vlm"
                stage_stats["vlm"]["named"] += 1

        # Open-world fallback: identify any grocery product on Earth by chaining
        # a free-form Claude call with Open Food Facts verification (and
        # barcode lookup when pyzbar is available). Runs only on detections
        # the closed-set pipeline could not name.
        if self._open_world.enabled:
            open_world_cache_dir = job_dir / "open_world"
            open_world_cache_dir.mkdir(parents=True, exist_ok=True)
            for det in enriched:
                if det is None or det.get("product_identity_label"):
                    continue
                crop_path = det.get("crop_path")
                if not crop_path:
                    continue
                resolved = Path(crop_path)
                if not resolved.is_absolute():
                    resolved = (job_dir / resolved).resolve()
                if not resolved.exists():
                    continue

                resolved_raw: Path | None = None
                raw_crop_path = det.get("raw_crop_path")
                if raw_crop_path:
                    raw_candidate = Path(raw_crop_path)
                    if not raw_candidate.is_absolute():
                        raw_candidate = (job_dir / raw_candidate).resolve()
                    if raw_candidate.exists():
                        resolved_raw = raw_candidate

                stage_stats["open_world"]["attempted"] += 1
                cache_path = open_world_cache_dir / f"{det['id']}.json"
                ow_result: dict | None = None
                if cache_path.exists():
                    try:
                        ow_result = json.loads(cache_path.read_text(encoding="utf-8"))
                    except json.JSONDecodeError:
                        ow_result = None

                if ow_result is None:
                    candidate_names: list[str] = []
                    catalog_candidate = det.get("catalog_candidate")
                    if catalog_candidate:
                        candidate_names.append(str(catalog_candidate))
                    for cand in det.get("catalog_candidates") or []:
                        name = cand.get("product_name") if isinstance(cand, dict) else None
                        if name and name not in candidate_names:
                            candidate_names.append(str(name))
                    if (
                        det.get("recognized_product_name")
                        and det["recognized_product_name"] not in candidate_names
                    ):
                        candidate_names.append(str(det["recognized_product_name"]))
                    ow_ocr_text = det.get("recognized_product_name") or ""
                    try:
                        ow_result = self._open_world.identify(
                            crop_path=resolved,
                            raw_crop_path=resolved_raw,
                            ocr_text=ow_ocr_text,
                            catalog_candidates=candidate_names,
                        )
                    except Exception as exc:
                        ow_result = {
                            "source": OPEN_WORLD_SOURCE,
                            "product_name": None,
                            "exact_match": False,
                            "confidence": 0.0,
                            "reason": f"open-world stage raised {type(exc).__name__}: {exc}",
                        }
                    if ow_result is not None:
                        try:
                            cache_path.write_text(
                                json.dumps(ow_result, indent=2),
                                encoding="utf-8",
                            )
                        except OSError:
                            pass

                if not ow_result:
                    continue

                # Always surface the open-world payload on the detection so the
                # frontend / debugging tools can inspect Open Food Facts metadata
                # even when the confidence isn't high enough to promote a label.
                det["open_world"] = ow_result

                product_name = ow_result.get("product_name")
                if not (ow_result.get("exact_match") and product_name):
                    continue
                det["recognized_product_name"] = product_name
                det["product_identity_confidence"] = float(ow_result.get("confidence") or 0.0)
                det["product_identity_reason"] = ow_result.get("reason")
                det["product_identity_source"] = ow_result.get("source") or OPEN_WORLD_SOURCE
                det["label"] = product_name
                det["display_label"] = product_name
                det["label_source"] = IDENTIFIED_LABEL_SOURCE
                det["product_identity_label"] = True
                det["identity_stage"] = "open_world"
                stage_stats["open_world"]["named"] += 1
                if ow_result.get("brand"):
                    det["brand"] = ow_result["brand"]
                if ow_result.get("off_image_url"):
                    det["reference_image_url"] = ow_result["off_image_url"]
                if ow_result.get("off_code"):
                    det["off_code"] = ow_result["off_code"]
                if ow_result.get("barcode"):
                    det["barcode"] = ow_result["barcode"]

        named_count = 0
        source_counts: dict[str, int] = {}
        for det in enriched:
            if det is None:
                continue
            # Stamp canonical class identity onto every record so downstream
            # consumers (results.json, planogram, Blender) get class-based rows.
            self._class_registry.annotate(det)
            if not det.get("product_identity_label"):
                continue
            named_count += 1
            source = str(det.get("product_identity_source") or "unknown")
            source_counts[source] = source_counts.get(source, 0) + 1

        summary = self.summary_metadata(
            named_count,
            len(detections),
            source_counts=source_counts,
        )
        summary["pipeline"] = self.pipeline_status(stage_stats)
        return [det for det in enriched if det is not None], summary

    def _apply_ocr_result(self, detection: dict, result: dict) -> None:
        detection["recognized_product_name"] = result.get("product_name")
        detection["product_identity_confidence"] = float(result.get("confidence") or 0.0)
        detection["product_identity_reason"] = result.get("reason")
        detection["product_identity_source"] = result.get("source")

        if result.get("exact_match") and result.get("product_name"):
            detection["label"] = result["product_name"]
            detection["display_label"] = result["product_name"]
            detection["label_source"] = IDENTIFIED_LABEL_SOURCE
            detection["product_identity_label"] = True
            detection["identity_stage"] = "ocr"
            # Auto-seed the embedding knowledge base so future crops of this product
            # can be classified even when package text is unreadable.
            crop_path = result.get("selected_crop_path") or detection.get("crop_path")
            if crop_path and result.get("selected_crop_type") == "raw":
                self._embedding_classifier.seed_from_crop(result["product_name"], Path(crop_path))

    def identify_crop(self, crop_path: Path) -> dict:
        with Image.open(crop_path) as image:
            rgb = image.convert("RGB")
            variants = self._build_variants(rgb)

        best_result: dict | None = None
        for variant_name, variant in variants:
            try:
                lines = self._ocr_lines(variant)
            except Exception as exc:
                return {
                    "generated_at": datetime.now(timezone.utc).isoformat(),
                    "source": self._engine,
                    "product_name": None,
                    "exact_match": False,
                    "confidence": 0.0,
                    "reason": f"OCR failed: {exc}",
                    "variant": variant_name,
                }

            result = self._infer_name(lines)
            result["variant"] = variant_name
            result["source"] = self._engine
            if best_result is None or self._result_key(result) > self._result_key(best_result):
                best_result = result

        assert best_result is not None
        best_result = self._combine_with_catalog(crop_path, best_result)
        best_result["cache_version"] = PRODUCT_IDENTIFIER_CACHE_VERSION
        best_result["generated_at"] = datetime.now(timezone.utc).isoformat()
        return best_result

    def identify_detection_crop(
        self,
        focus_crop_path: Path,
        raw_crop_path: Path | None = None,
    ) -> dict:
        focus_result = self.identify_crop(focus_crop_path)
        focus_result["selected_crop_path"] = str(focus_crop_path)
        focus_result["selected_crop_type"] = "focus"

        if raw_crop_path is None or raw_crop_path == focus_crop_path or not raw_crop_path.exists():
            return focus_result

        raw_result = self.identify_crop(raw_crop_path)
        raw_result["selected_crop_path"] = str(raw_crop_path)
        raw_result["selected_crop_type"] = "raw"

        focus_key = self._selection_key(focus_result)
        raw_key = self._selection_key(raw_result)
        if raw_key > focus_key:
            return raw_result
        return focus_result

    def _combine_with_catalog(self, crop_path: Path, ocr_result: dict) -> dict:
        combined = dict(ocr_result)
        if not self._embedding_classifier.enabled or self._embedding_classifier.product_count() <= 0:
            return combined

        observed_tokens = combined.get("observed_tokens") or []
        if not observed_tokens and not combined.get("exact_match"):
            try:
                with Image.open(crop_path) as image:
                    catalog_result = self._embedding_classifier.classify(image.convert("RGB"))
            except Exception as exc:
                combined["catalog_reason"] = f"Catalog matching failed: {exc}"
                return combined

            candidates = catalog_result.get("candidates") or []
            if candidates:
                combined["catalog_candidates"] = candidates
                combined["catalog_candidate"] = candidates[0]["product_name"]
                combined["catalog_confidence"] = float(candidates[0].get("confidence") or 0.0)

            top = candidates[0] if candidates else None
            if (
                catalog_result.get("exact_match")
                and catalog_result.get("product_name")
                and top is not None
                and float(catalog_result.get("confidence") or 0.0) >= self.visual_only_confidence_threshold
                and float(top.get("best_similarity") or 0.0) >= self.visual_only_similarity_threshold
            ):
                combined["product_name"] = catalog_result["product_name"]
                combined["exact_match"] = True
                combined["confidence"] = round(float(catalog_result.get("confidence") or 0.0), 4)
                combined["reason"] = (
                    "Catalog image similarity strongly matched a known product when package text was unreadable."
                )
                combined["source"] = "embedding_classifier"
                return combined

            combined["catalog_reason"] = "Skipped exact catalog promotion because package text was unreadable and the visual match was not strong enough."
            return combined

        try:
            with Image.open(crop_path) as image:
                catalog_result = self._embedding_classifier.classify(image.convert("RGB"))
        except Exception as exc:
            combined["catalog_reason"] = f"Catalog matching failed: {exc}"
            return combined

        candidates = catalog_result.get("candidates") or []
        if candidates:
            combined["catalog_candidates"] = candidates
            combined["catalog_candidate"] = candidates[0]["product_name"]
            combined["catalog_confidence"] = float(candidates[0].get("confidence") or 0.0)

        ocr_name = combined.get("product_name")
        catalog_name = catalog_result.get("product_name") or catalog_result.get("candidate")
        if combined.get("exact_match") and ocr_name:
            top_catalog_name = str(combined.get("catalog_candidate") or "").strip()
            if (
                top_catalog_name
                and top_catalog_name != ocr_name
                and self._should_demote_ocr_exact(
                    observed_tokens=combined.get("observed_tokens") or [],
                    catalog_confidence=float(combined.get("catalog_confidence") or 0.0),
                )
            ):
                combined["candidate"] = ocr_name
                combined["product_name"] = None
                combined["exact_match"] = False
                combined["reason"] = (
                    "Package text suggested a generic product name, but catalog similarity disagreed."
                )
                combined["catalog_disagreement"] = top_catalog_name
                return combined

            if catalog_result.get("product_name") == ocr_name:
                combined["confidence"] = round(
                    max(
                        float(combined.get("confidence") or 0.0),
                        float(catalog_result.get("confidence") or 0.0),
                        min(
                            0.99,
                            (
                                0.55 * float(combined.get("confidence") or 0.0)
                                + 0.45 * float(catalog_result.get("confidence") or 0.0)
                                + 0.06
                            ),
                        ),
                    ),
                    4,
                )
                combined["reason"] = (
                    "Package text and catalog image similarity agreed on the same product."
                )
                combined["source"] = f"{self._engine}+embedding_classifier"
            elif catalog_result.get("product_name"):
                combined["catalog_disagreement"] = catalog_result.get("product_name")
            return combined

        if not catalog_name:
            return combined

        alignment = self._catalog_alignment_score(observed_tokens, catalog_name)
        catalog_confidence = float(catalog_result.get("confidence") or 0.0)
        low_signal_observations = bool(observed_tokens) and not self._high_signal_tokens(observed_tokens)
        if catalog_result.get("product_name") and (
            alignment >= 0.55
            or (not observed_tokens and catalog_confidence >= 0.76)
            or (low_signal_observations and catalog_confidence >= 0.73)
        ):
            combined["product_name"] = catalog_result["product_name"]
            combined["exact_match"] = True
            combined["confidence"] = round(
                max(
                    catalog_confidence,
                    min(
                        0.98,
                        (0.68 * catalog_confidence) + (0.22 * alignment) + 0.10,
                    ),
                ),
                4,
            )
            combined["reason"] = (
                "Catalog image similarity matched a known product and the OCR text aligned with that catalog entry."
                if observed_tokens and not low_signal_observations
                else (
                    "Catalog image similarity matched a known product when OCR only recovered low-signal package text."
                    if observed_tokens
                    else "Catalog image similarity matched a known product when package text was unreadable."
                )
            )
            combined["source"] = f"{self._engine}+embedding_classifier"
            return combined

        return combined

    def _load_catalog_aliases(self) -> dict[str, tuple[str, ...]]:
        names = {pattern.name for pattern in _PRODUCT_PATTERNS}
        names.update(self._catalog.product_names())
        alias_map: dict[str, tuple[str, ...]] = {}
        for name in sorted(names):
            aliases = [name, *self._catalog.aliases_for(name)]
            alias_map[name] = tuple(dict.fromkeys(alias.strip() for alias in aliases if alias.strip()))
        return alias_map

    def _catalog_alignment_score(self, observed_tokens: list[str], product_name: str) -> float:
        if not observed_tokens:
            return 0.0

        observed = set(observed_tokens)
        aliases = self._catalog_aliases.get(product_name, (product_name,))
        best_score = 0.0
        for alias in aliases:
            alias_phrase = self._normalize_phrase(alias)
            alias_tokens = set(self._normalized_tokens(alias_phrase))
            if not alias_tokens:
                continue
            required_tokens = {token for token in alias_tokens if token in _BRAND_WORDS or token not in _GENERIC_WORDS}
            if not required_tokens:
                required_tokens = alias_tokens
            hits = len(required_tokens & observed)
            brand_hits = len((required_tokens & observed) & _BRAND_WORDS)
            if brand_hits == 0 and hits < 2:
                continue
            if hits <= 0:
                continue
            score = hits / float(len(required_tokens))
            best_score = max(best_score, score)
        return round(best_score, 4)

    def _selection_key(self, result: dict) -> tuple[int, int, int, float, float, int]:
        observed_tokens = result.get("observed_tokens") or []
        candidate_name = result.get("product_name") or result.get("candidate") or ""
        candidate_tokens = self._normalized_tokens(self._normalize_phrase(candidate_name))
        return (
            1 if result.get("exact_match") and result.get("product_name") else 0,
            len(self._high_signal_tokens(candidate_tokens)),
            len(self._high_signal_tokens(observed_tokens)),
            float(result.get("catalog_confidence") or 0.0),
            float(result.get("confidence") or 0.0),
            len(candidate_tokens),
        )

    def _high_signal_tokens(self, observed_tokens: list[str] | set[str]) -> set[str]:
        return {
            token
            for token in observed_tokens
            if token and token not in _LOW_SIGNAL_TOKENS and token not in _BRAND_WORDS
        }

    def _should_demote_ocr_exact(
        self,
        observed_tokens: list[str],
        catalog_confidence: float,
    ) -> bool:
        if catalog_confidence < 0.55:
            return False
        token_set = {token for token in observed_tokens if token}
        if not token_set:
            return False

        high_signal_tokens = self._high_signal_tokens(token_set)
        brand_tokens = {token for token in token_set if token in _BRAND_WORDS}

        if high_signal_tokens:
            return False
        return len(brand_tokens) <= 1

    def pipeline_status(self, stage_stats: dict[str, dict[str, int]] | None = None) -> list[dict]:
        """Report every identification stage with its enabled state and, when
        disabled, the concrete reason — no more silent stage failures."""
        embedding_ready = (
            self.embedding_fallback_enabled and self._embedding_classifier.enabled
        )
        if not self.embedding_fallback_enabled:
            embedding_reason = "disabled via PRODUCT_NAME_EMBEDDING_FALLBACK"
        elif not self._embedding_classifier.enabled:
            embedding_reason = "torch/transformers not installed"
        elif self._embedding_classifier.product_count() <= 0:
            embedding_ready = False
            embedding_reason = "knowledge base has no reference images"
        else:
            embedding_reason = None

        clip_ready = self._clip_classifier.enabled
        if clip_ready and self._clip_classifier.product_count() <= 0:
            clip_ready = False
            clip_reason = "knowledge base has no reference images"
        elif not clip_ready:
            clip_reason = "torch/transformers not installed or disabled via CLIP_CLASSIFIER_ENABLED"
        else:
            clip_reason = None

        local_backend = getattr(self._vlm, "_local_backend", None)
        if self._vlm.enabled:
            vlm_reason = None
        elif getattr(self._vlm, "_local_backend_requested", False):
            vlm_reason = "LOCAL_VLM_ENABLED is set but the local VLM failed to initialize"
        else:
            vlm_reason = "ANTHROPIC_API_KEY not set and LOCAL_VLM_ENABLED not enabled"

        stages = [
            {
                "stage": "ocr",
                "enabled": bool(self.enabled and self._engine),
                "detail": self._engine,
                "reason": self._engine_reason,
            },
            {
                "stage": "embedding",
                "enabled": embedding_ready,
                "detail": f"{self._embedding_classifier.product_count()} catalog products"
                if self._embedding_classifier.enabled
                else None,
                "reason": embedding_reason,
            },
            {
                "stage": "clip",
                "enabled": clip_ready,
                "detail": self._clip_classifier.model_name,
                "reason": clip_reason,
            },
            {
                "stage": "vlm",
                "enabled": self._vlm.enabled,
                "detail": (
                    f"local {local_backend.model_id}"
                    if local_backend is not None and local_backend.enabled
                    else self._vlm.model
                )
                if self._vlm.enabled
                else None,
                "reason": vlm_reason,
            },
            {
                "stage": "open_world",
                "enabled": self._open_world.enabled,
                "detail": None,
                "reason": None
                if self._open_world.enabled
                else "requires an enabled VLM stage (or disabled via OPEN_WORLD_ENABLED)",
            },
        ]
        if stage_stats:
            for stage in stages:
                stats = stage_stats.get(stage["stage"])
                if stats is not None:
                    stage["attempted"] = stats["attempted"]
                    stage["named"] = stats["named"]
        return stages

    def summary_metadata(
        self,
        named_count: int,
        total_count: int,
        source_counts: dict[str, int] | None = None,
    ) -> dict:
        source_counts = source_counts or {}
        product_name_sources = sorted(source_counts)
        product_name_source = (
            product_name_sources[0]
            if len(product_name_sources) == 1
            else (product_name_sources if product_name_sources else self._engine)
        )
        if total_count <= 0:
            return {
                "product_identity_labels": False,
                "named_product_count": 0,
                "unnamed_product_count": 0,
                "product_name_source": product_name_source,
                "product_name_sources": source_counts,
                "note": "No projected products are available to name.",
            }

        if not self.enabled:
            return {
                "product_identity_labels": False,
                "named_product_count": 0,
                "unnamed_product_count": total_count,
                "product_name_source": product_name_source,
                "product_name_sources": source_counts,
                "note": "Local product naming is unavailable, so unnamed products stay neutral.",
            }

        unnamed_count = max(0, total_count - named_count)
        if named_count <= 0:
            note = (
                "Local product naming ran on each crop, but exact package names were not confirmed."
            )
        elif unnamed_count <= 0:
            note = (
                "Exact product names were recognized from local visual models and catalog references "
                "for every projected product."
            )
        else:
            note = (
                f"Exact product names were recognized from local visual models or catalog references for "
                f"{named_count} of {total_count} projected products. Unreadable products stay neutral."
            )

        return {
            "product_identity_labels": named_count > 0,
            "named_product_count": named_count,
            "unnamed_product_count": unnamed_count,
            "product_name_source": product_name_source,
            "product_name_sources": source_counts,
            "note": note,
        }

    def _detect_engine(self) -> str | None:
        requested = os.getenv("PRODUCT_NAME_OCR_ENGINE", "auto").strip().lower()
        if requested in _DISABLE_VALUES or requested in {"none", "off", "disabled"}:
            self._engine_reason = "disabled via PRODUCT_NAME_OCR_ENGINE"
            return None
        missing: list[str] = []
        # GCV — best accuracy on any OS, use when credentials are configured.
        gcv_creds = os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "").strip()
        gcv_project = os.getenv("GOOGLE_CLOUD_PROJECT", "").strip()
        if gcv_creds or gcv_project:
            try:
                import google.cloud.vision  # noqa: F401
                return GOOGLE_CLOUD_VISION_SOURCE
            except ImportError:
                missing.append("google-cloud-vision package not installed despite GCV credentials")
        else:
            missing.append("Google Cloud Vision credentials not configured")
        # Apple Vision — macOS only, excellent quality for English product text.
        if sys.platform == "darwin":
            if shutil.which("swiftc"):
                return APPLE_VISION_SOURCE
            missing.append("swiftc not found for Apple Vision")
        else:
            missing.append("Apple Vision requires macOS")
        # PaddleOCR — Linux/AWS fallback when GCV is not configured.
        try:
            from paddleocr import PaddleOCR  # noqa: F401
            return PADDLEOCR_SOURCE
        except ImportError:
            missing.append("paddleocr not installed")
        self._engine_reason = "no OCR engine available: " + "; ".join(missing)
        logger.warning("Product-name OCR is disabled — %s", self._engine_reason)
        return None

    def _ensure_vision_binary(self) -> Path:
        if self._vision_binary_path and self._vision_binary_path.exists():
            return self._vision_binary_path

        script_path = Path(__file__).with_name("vision_ocr.swift")
        if not script_path.exists():
            raise RuntimeError("vision_ocr.swift is missing from the backend directory.")

        binary_path = Path(tempfile.gettempdir()) / "shelf_mapper_vision_ocr"
        source_mtime = script_path.stat().st_mtime
        if binary_path.exists() and binary_path.stat().st_mtime >= source_mtime:
            self._vision_binary_path = binary_path
            return binary_path

        subprocess.run(
            ["swiftc", str(script_path), "-O", "-o", str(binary_path)],
            check=True,
            timeout=max(30.0, self.timeout_seconds * 2),
        )
        self._vision_binary_path = binary_path
        return binary_path

    def _ocr_lines(self, image: Image.Image) -> list[OCRLine]:
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as temp_file:
            temp_path = Path(temp_file.name)
        try:
            prepared = self._prepare_image(image)
            prepared.save(temp_path, format="JPEG", quality=88)
            if self._engine == GOOGLE_CLOUD_VISION_SOURCE:
                return self._google_cloud_vision_ocr(temp_path)
            if self._engine == APPLE_VISION_SOURCE:
                return self._vision_ocr(temp_path)
            if self._engine == PADDLEOCR_SOURCE:
                return self._paddleocr_ocr(temp_path)
            return []
        finally:
            temp_path.unlink(missing_ok=True)

    def _prepare_image(self, image: Image.Image) -> Image.Image:
        rgb = image.convert("RGB")
        width, height = rgb.size
        longest_side = max(width, height)
        if longest_side <= self.max_analysis_side:
            return rgb

        scale = self.max_analysis_side / float(longest_side)
        resized_size = (
            max(1, int(round(width * scale))),
            max(1, int(round(height * scale))),
        )
        return rgb.resize(resized_size, Image.Resampling.LANCZOS)

    def _build_variants(self, image: Image.Image) -> list[tuple[str, Image.Image]]:
        width, height = image.size
        variants = [("center", self._center_crop(image, 0.58, 0.54))]
        if width >= 2 * self.min_image_side and height >= 2 * self.min_image_side:
            variants.append(("full", image))
        return variants

    def _center_crop(self, image: Image.Image, width_ratio: float, height_ratio: float) -> Image.Image:
        width, height = image.size
        crop_width = max(self.min_image_side, int(round(width * width_ratio)))
        crop_height = max(self.min_image_side, int(round(height * height_ratio)))
        crop_width = min(width, crop_width)
        crop_height = min(height, crop_height)

        left = max(0, (width - crop_width) // 2)
        top = max(0, (height - crop_height) // 2)
        right = left + crop_width
        bottom = top + crop_height
        return image.crop((left, top, right, bottom))

    def _vision_ocr(self, image_path: Path) -> list[OCRLine]:
        binary_path = self._ensure_vision_binary()
        result = subprocess.run(
            [str(binary_path), str(image_path)],
            check=True,
            capture_output=True,
            encoding="utf-8",
            timeout=self.timeout_seconds,
        )
        payload = json.loads(result.stdout or "[]")
        return [
            OCRLine(
                text=str(item.get("text", "")),
                min_x=float(item.get("min_x", 0.0)),
                min_y=float(item.get("min_y", 0.0)),
                width=float(item.get("width", 0.0)),
                height=float(item.get("height", 0.0)),
                confidence=float(item["confidence"]) if item.get("confidence") is not None else None,
            )
            for item in payload
        ]

    def _paddleocr_ocr(self, image_path: Path) -> list[OCRLine]:
        from paddleocr import PaddleOCR

        if not hasattr(self, "_paddle_model"):
            self._paddle_model = PaddleOCR(
                lang="en",
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                use_textline_orientation=False,
            )

        # Upscale small crops so PaddleOCR can read fine text reliably.
        with Image.open(image_path) as img:
            orig_w, orig_h = img.width, img.height
            min_side = min(orig_w, orig_h)
            if min_side < 400:
                scale = 400 / min_side
                new_w = max(1, int(round(orig_w * scale)))
                new_h = max(1, int(round(orig_h * scale)))
                upscaled = img.convert("RGB").resize((new_w, new_h), Image.Resampling.LANCZOS)
            else:
                upscaled = img.convert("RGB")
                new_w, new_h = orig_w, orig_h

        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
            upscaled_path = Path(f.name)
        try:
            upscaled.save(upscaled_path, format="JPEG", quality=95)
            result = self._paddle_model.predict(str(upscaled_path))
        finally:
            upscaled_path.unlink(missing_ok=True)

        if not result:
            return []

        page = result[0]
        rec_texts = page.get("rec_texts") or []
        rec_scores = page.get("rec_scores") or []
        rec_polys = page.get("rec_polys") or []

        if not rec_texts:
            return []

        img_w, img_h = float(new_w), float(new_h)
        observations: list[OCRLine] = []
        for text, score, poly in zip(rec_texts, rec_scores, rec_polys):
            text = text.strip()
            if not text:
                continue
            xs = [p[0] for p in poly]
            ys = [p[1] for p in poly]
            min_x = min(xs) / img_w
            max_x = max(xs) / img_w
            min_y = min(ys) / img_h
            max_y = max(ys) / img_h
            observations.append(
                OCRLine(
                    text=text,
                    min_x=min_x,
                    min_y=1.0 - max_y,
                    width=max_x - min_x,
                    height=max_y - min_y,
                    confidence=float(score) * 100.0,
                )
            )

        return observations

    def _gcv_cache_dir(self) -> Path:
        cache_dir = DEFAULT_KB_DIR / "gcv_ocr_cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        return cache_dir

    def _gcv_image_hash(self, image_bytes: bytes) -> str:
        return hashlib.sha256(image_bytes).hexdigest()

    def _gcv_cache_load(self, image_hash: str) -> list[OCRLine] | None:
        cache_path = self._gcv_cache_dir() / f"{image_hash}.json"
        if not cache_path.exists():
            return None
        try:
            payload = json.loads(cache_path.read_text(encoding="utf-8"))
            return [
                OCRLine(
                    text=item["text"],
                    min_x=item["min_x"],
                    min_y=item["min_y"],
                    width=item["width"],
                    height=item["height"],
                    confidence=item.get("confidence"),
                )
                for item in payload
            ]
        except Exception:
            return None

    def _gcv_cache_save(self, image_hash: str, lines: list[OCRLine]) -> None:
        cache_path = self._gcv_cache_dir() / f"{image_hash}.json"
        cache_path.write_text(
            json.dumps(
                [
                    {
                        "text": line.text,
                        "min_x": line.min_x,
                        "min_y": line.min_y,
                        "width": line.width,
                        "height": line.height,
                        "confidence": line.confidence,
                    }
                    for line in lines
                ],
                indent=2,
            ),
            encoding="utf-8",
        )

    def _google_cloud_vision_ocr(self, image_path: Path) -> list[OCRLine]:
        from google.cloud import vision

        with open(image_path, "rb") as f:
            content = f.read()

        image_hash = self._gcv_image_hash(content)
        cached = self._gcv_cache_load(image_hash)
        if cached is not None:
            return cached

        client = vision.ImageAnnotatorClient()
        image = vision.Image(content=content)
        response = client.document_text_detection(image=image)

        if response.error.message:
            raise RuntimeError(f"Google Cloud Vision error: {response.error.message}")

        full_text = response.full_text_annotation
        if not full_text.pages:
            self._gcv_cache_save(image_hash, [])
            return []

        page = full_text.pages[0]
        img_w = float(page.width) if page.width else 1.0
        img_h = float(page.height) if page.height else 1.0

        observations: list[OCRLine] = []
        for block in page.blocks:
            for paragraph in block.paragraphs:
                word_texts: list[str] = []
                min_x, min_y = img_w, img_h
                max_x, max_y = 0.0, 0.0
                conf_sum = 0.0
                word_count = 0

                for word in paragraph.words:
                    word_text = "".join(s.text for s in word.symbols).strip()
                    if not word_text:
                        continue
                    word_texts.append(word_text)
                    verts = word.bounding_box.vertices
                    xs = [v.x for v in verts]
                    ys = [v.y for v in verts]
                    min_x = min(min_x, min(xs))
                    min_y = min(min_y, min(ys))
                    max_x = max(max_x, max(xs))
                    max_y = max(max_y, max(ys))
                    conf_sum += word.confidence
                    word_count += 1

                if not word_texts:
                    continue

                observations.append(
                    OCRLine(
                        text=" ".join(word_texts),
                        min_x=min_x / img_w,
                        min_y=1.0 - (max_y / img_h),
                        width=(max_x - min_x) / img_w,
                        height=(max_y - min_y) / img_h,
                        confidence=(conf_sum / word_count) * 100.0,
                    )
                )

        self._gcv_cache_save(image_hash, observations)
        return observations

    def _image_size(self, image_path: Path) -> tuple[int, int]:
        with Image.open(image_path) as image:
            return image.size

    def _infer_name(self, lines: list[OCRLine]) -> dict:
        cleaned_lines = [self._clean_line(line) for line in lines]
        cleaned_lines = [line for line in cleaned_lines if line is not None]
        if not cleaned_lines:
            return self._unknown("No readable package text near the centered product.")

        anchors = self._candidate_anchors(cleaned_lines)
        best_match: dict | None = None
        best_alias_match: dict | None = None
        best_fallback: dict | None = None
        for anchor in anchors:
            group = self._anchor_group(anchor, cleaned_lines)
            result = self._match_catalog_product(anchor, group)
            if result is not None and (
                best_match is None or self._result_key(result) > self._result_key(best_match)
            ):
                best_match = result

            alias_result = self._match_catalog_alias(anchor, group)
            if alias_result is not None and (
                best_alias_match is None or self._result_key(alias_result) > self._result_key(best_alias_match)
            ):
                best_alias_match = alias_result

            fallback = self._fallback_result(anchor, group)
            if fallback is not None and (
                best_fallback is None or self._result_key(fallback) > self._result_key(best_fallback)
            ):
                best_fallback = fallback

        if best_match is not None:
            return best_match
        if best_alias_match is not None:
            return best_alias_match
        if best_fallback is not None:
            return best_fallback
        return self._unknown("Package text was detected but could not be normalized.")

    def _candidate_anchors(self, lines: list[OCRLine]) -> list[OCRLine]:
        scored: list[tuple[tuple[float, float, float], OCRLine]] = []
        for line in lines:
            tokens = self._normalized_tokens(line.text)
            if not tokens:
                continue
            brand_hits = sum(1 for token in tokens if token in _BRAND_WORDS)
            distinctive_hits = sum(1 for token in tokens if token not in _GENERIC_WORDS)
            score = self._line_score(line) + (0.32 * brand_hits) + (0.06 * distinctive_hits)
            scored.append(((float(brand_hits), score, line.area), line))

        scored.sort(key=lambda item: item[0], reverse=True)
        anchors: list[OCRLine] = []
        for _, line in scored:
            if any(self._is_duplicate_text(line.text, existing.text) for existing in anchors):
                continue
            anchors.append(line)
            if len(anchors) >= 6:
                break

        if anchors:
            return anchors
        return [self._best_anchor(lines)] if lines else []

    def _best_anchor(self, lines: list[OCRLine]) -> OCRLine:
        return max(lines, key=self._line_score)

    def _anchor_group(self, anchor: OCRLine, lines: list[OCRLine]) -> list[OCRLine]:
        group = [
            line
            for line in lines
            if abs(line.center_x - anchor.center_x) <= max(0.16, anchor.width * 1.3, line.width * 1.1)
            and abs(line.center_y - anchor.center_y) <= max(0.16, anchor.height * 4.2, line.height * 2.8)
        ]
        group.sort(
            key=lambda item: (
                abs(item.center_x - anchor.center_x) + (1.15 * abs(item.center_y - anchor.center_y)),
                -item.area,
            )
        )
        trimmed: list[OCRLine] = []
        for line in group:
            if any(self._is_duplicate_text(line.text, existing.text) for existing in trimmed):
                continue
            trimmed.append(line)
            if len(trimmed) >= 6:
                break
        trimmed.sort(key=lambda item: (-item.center_y, item.min_x))
        return trimmed

    def _match_catalog_product(self, anchor: OCRLine, group: list[OCRLine]) -> dict | None:
        if not group:
            return None

        anchor_tokens = set(self._normalized_tokens(anchor.text))
        observed_text = self._group_text(group)
        group_tokens = [token for line in group for token in self._normalized_tokens(line.text)]
        token_set = set(group_tokens)
        brand_tokens = {token for token in token_set if token in _BRAND_WORDS}

        best_pattern: ProductPattern | None = None
        best_score: tuple[float, int, int, int] | None = None
        for pattern in _PRODUCT_PATTERNS:
            required_hits = 0
            matched_groups: list[tuple[str, ...]] = []
            for variants in pattern.required_groups:
                if any(variant in token_set for variant in variants):
                    required_hits += 1
                    matched_groups.append(variants)

            if required_hits != len(pattern.required_groups):
                continue

            brand_group = pattern.required_groups[0]
            brand_match = any(variant in anchor_tokens for variant in brand_group)
            if brand_tokens and not any(variant in brand_tokens for variant in brand_group):
                continue
            if brand_tokens and not brand_match and not any(variant in anchor_tokens for group in matched_groups for variant in group):
                continue

            optional_hits = self._pattern_optional_hits(pattern, token_set)
            unmatched_brands = sum(1 for token in brand_tokens if token not in brand_group)
            pattern_score = (
                required_hits * 100
                + optional_hits * 8
                + (12 if brand_match else 0)
                - unmatched_brands * 6
            )
            tie_break = (
                float(pattern_score),
                len(pattern.required_groups),
                optional_hits,
                -unmatched_brands,
            )
            if best_score is None or tie_break > best_score:
                best_score = tie_break
                best_pattern = pattern

        if best_pattern is None:
            return None

        word_count = len(token_set)
        brand_token_count = sum(1 for token in token_set if token in _BRAND_WORDS)
        confidence = self._confidence(anchor, group, brand_token_count, word_count)
        confidence += min(0.18, 0.04 * len(best_pattern.required_groups))
        confidence += min(0.06, 0.02 * self._pattern_optional_hits(best_pattern, token_set))
        confidence = max(self.confidence_threshold, min(0.99, confidence))
        return {
            "product_name": best_pattern.name,
            "exact_match": True,
            "confidence": confidence,
            "reason": "Package text matched a known catalog product near the centered crop.",
            "observed_text": observed_text,
            "observed_tokens": sorted(token_set),
        }

    def _match_catalog_alias(self, anchor: OCRLine, group: list[OCRLine]) -> dict | None:
        if not group:
            return None

        observed_text = self._group_text(group)
        token_set = set(self._normalized_tokens(observed_text))
        if not token_set:
            return None

        best_name: str | None = None
        best_score: tuple[float, int, int] | None = None
        for product_name, aliases in self._catalog_aliases.items():
            for alias in aliases:
                alias_text = self._normalize_phrase(alias)
                alias_tokens = set(self._normalized_tokens(alias_text))
                if not alias_tokens:
                    continue
                required_tokens = {
                    token
                    for token in alias_tokens
                    if token in _BRAND_WORDS or token not in _GENERIC_WORDS
                }
                if not required_tokens:
                    required_tokens = alias_tokens
                if len(required_tokens) < 2:
                    continue

                hits = len(required_tokens & token_set)
                if hits != len(required_tokens):
                    continue

                score = (
                    len(alias_tokens & token_set),
                    len(required_tokens),
                    sum(1 for token in alias_tokens if token in _BRAND_WORDS),
                )
                if best_score is None or score > best_score:
                    best_score = score
                    best_name = product_name

        if best_name is None:
            return None

        brand_token_count = sum(1 for token in token_set if token in _BRAND_WORDS)
        word_count = len(token_set)
        confidence = self._confidence(anchor, group, brand_token_count, word_count)
        confidence += 0.10
        confidence = max(self.confidence_threshold, min(0.97, confidence))
        return {
            "product_name": best_name,
            "exact_match": True,
            "confidence": confidence,
            "reason": "Package text matched a catalog alias near the centered crop.",
            "observed_text": observed_text,
            "observed_tokens": sorted(token_set),
        }

    def _pattern_optional_hits(self, pattern: ProductPattern, token_set: set[str]) -> int:
        hits = 0
        for variants in pattern.optional_groups:
            if any(variant in token_set for variant in variants):
                hits += 1
        return hits

    def _fallback_result(self, anchor: OCRLine, group: list[OCRLine]) -> dict | None:
        if not group:
            return None

        combined = self._group_text(group)
        normalized_tokens = self._normalized_tokens(combined)
        if not normalized_tokens:
            return None

        brand_token_count = sum(1 for token in normalized_tokens if token in _BRAND_WORDS)
        word_count = len(normalized_tokens)
        confidence = self._confidence(anchor, group, brand_token_count, word_count)
        candidate = self._display_phrase(combined)
        if not candidate:
            return None
        return self._unknown(
            "Package text did not match a known catalog product.",
            confidence=confidence,
            candidate=candidate,
            observed_text=combined,
            observed_tokens=normalized_tokens,
        )

    def _unknown(
        self,
        reason: str,
        confidence: float = 0.0,
        candidate: str | None = None,
        observed_text: str | None = None,
        observed_tokens: list[str] | None = None,
    ) -> dict:
        payload = {
            "product_name": None,
            "exact_match": False,
            "confidence": max(0.0, min(1.0, confidence)),
            "reason": reason,
        }
        if candidate:
            payload["candidate"] = candidate
        if observed_text:
            payload["observed_text"] = observed_text
        if observed_tokens:
            payload["observed_tokens"] = observed_tokens
        return payload

    def _group_text(self, group: list[OCRLine]) -> str:
        parts: list[str] = []
        for line in group:
            if parts and any(self._is_duplicate_text(line.text, existing) for existing in parts):
                continue
            parts.append(line.text)
        return self._normalize_phrase(" ".join(parts))

    def _clean_line(self, line: OCRLine) -> OCRLine | None:
        text = unicodedata.normalize("NFKD", line.text)
        text = text.encode("ascii", "ignore").decode("ascii")
        text = re.sub(r"[^A-Za-z0-9'&+/ -]+", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        if len(text) < 3:
            return None

        tokens = []
        for raw_token in text.upper().split():
            token = self._normalize_token(raw_token)
            if len(token) < 3:
                continue
            if token.lower() in _STOPWORDS and len(text.split()) == 1:
                continue
            tokens.append(token)

        if not tokens:
            return None

        return OCRLine(
            text=" ".join(tokens),
            min_x=line.min_x,
            min_y=line.min_y,
            width=line.width,
            height=line.height,
            confidence=line.confidence,
        )

    def _normalize_token(self, token: str) -> str:
        compact = re.sub(r"[^A-Z0-9&'+]", "", token.upper())
        if len(compact) < 2:
            return ""
        if compact in _TOKEN_NORMALIZATIONS:
            return _TOKEN_NORMALIZATIONS[compact]

        cutoff = 0.92 if len(compact) <= 4 else 0.8
        matches = get_close_matches(compact.lower(), _TOKEN_VOCABULARY, n=1, cutoff=cutoff)
        if matches:
            return matches[0].upper()
        # Pass unknown tokens through so new brands aren't silently dropped.
        return compact

    def _normalize_phrase(self, text: str) -> str:
        tokens = []
        previous = None
        for token in text.split():
            normalized = self._normalize_token(token)
            if not normalized:
                continue
            if previous == normalized:
                continue
            tokens.append(normalized)
            previous = normalized
        return " ".join(tokens)

    def _display_phrase(self, text: str) -> str:
        display_tokens = []
        for token in text.split():
            if token in _DISPLAY_NORMALIZATIONS:
                display_tokens.append(_DISPLAY_NORMALIZATIONS[token])
            elif token == "&":
                display_tokens.append("&")
            elif token.isupper() and len(token) <= 3:
                display_tokens.append(token)
            else:
                display_tokens.append(token.title())
        return " ".join(display_tokens)

    def _normalized_tokens(self, text: str) -> list[str]:
        return [token for token in text.split() if token and token.lower() not in _STOPWORDS]

    def _line_score(self, line: OCRLine) -> float:
        distance = ((line.center_x - 0.5) ** 2 + (line.center_y - 0.5) ** 2) ** 0.5
        centrality = max(0.0, 1.0 - distance / 0.65)
        size_bonus = min(1.2, (line.area * 18.0) + (line.height * 4.0))
        confidence_bonus = ((line.confidence or 60.0) / 100.0) * 0.2
        return centrality * (0.65 + size_bonus) + confidence_bonus

    def _confidence(
        self,
        anchor: OCRLine,
        group: list[OCRLine],
        brand_token_count: int,
        word_count: int,
    ) -> float:
        anchor_score = self._line_score(anchor)
        group_bonus = min(0.2, 0.05 * max(0, len(group) - 1))
        brand_bonus = 0.25 if brand_token_count >= 1 else 0.0
        phrase_bonus = 0.08 if word_count >= 3 else 0.0
        length_penalty = 0.12 if word_count > 6 else 0.0
        confidence = 0.24 + min(0.34, anchor_score * 0.26) + group_bonus + brand_bonus + phrase_bonus - length_penalty
        return max(0.0, min(0.97, confidence))

    def _result_key(self, result: dict) -> tuple[int, float]:
        return (
            1 if result.get("exact_match") else 0,
            float(result.get("confidence") or 0.0),
        )

    def _is_duplicate_text(self, current: str, previous: str) -> bool:
        if current == previous:
            return True
        return current in previous or previous in current

    def _is_large_enough(self, image_path: Path) -> bool:
        with Image.open(image_path) as image:
            width, height = image.size
        return min(width, height) >= self.min_image_side

    def _load_cache(self, cache_path: Path) -> dict | None:
        if not cache_path.exists():
            return None
        try:
            payload = json.loads(cache_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return None
        if int(payload.get("cache_version") or 0) != PRODUCT_IDENTIFIER_CACHE_VERSION:
            return None
        return payload
