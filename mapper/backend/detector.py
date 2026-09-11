from __future__ import annotations
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

try:
    import cv2
except ImportError:  # pragma: no cover - optional dependency in local dev
    cv2 = None

try:
    import numpy as np
except ImportError:  # pragma: no cover - optional dependency in local dev
    np = None

from PIL import Image

from product_identifier import ProductIdentifier

try:
    from ultralytics import YOLO
except ImportError:  # pragma: no cover - optional dependency in local dev
    YOLO = None

# COCO class IDs that can plausibly appear on a grocery store shelf.
# Set YOLO_CLASS_IDS env var (comma-separated ints) to override, or set YOLO_MODEL
# to a grocery-trained checkpoint to disable this filter entirely.
GROCERY_COCO_CLASS_IDS: frozenset[int] = frozenset({
    39, 40, 41, 45,                           # bottle, wine glass, cup, bowl
    46, 47, 48, 49, 50, 51, 52, 53, 54, 55,   # banana→cake (produce & prepared food)
})

_COCO_MODEL_NAMES: frozenset[str] = frozenset({
    "yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "yolov8l.pt", "yolov8x.pt",
    "yolo11n.pt", "yolo11s.pt", "yolo11m.pt", "yolo11l.pt", "yolo11x.pt",
})
# Custom shelf-product detectors: better than COCO for grocery shelves but produce
# a generic "object" class — product identity still comes from OCR / embeddings.
_SHELF_DETECTOR_MODELS: frozenset[str] = frozenset({
    "best.pt",
})
_NON_IDENTITY_MODELS: frozenset[str] = _COCO_MODEL_NAMES | _SHELF_DETECTOR_MODELS
GENERIC_LABEL_SOURCE = "generic_object_detector"
PRODUCT_LABEL_SOURCE = "product_classifier"
OCR_TEXT_PROPOSAL_SOURCE = "ocr_text_proposal_detector"
_OCR_BRAND_TOKENS: frozenset[str] = frozenset({
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
})


class ObjectDetector:
    def __init__(
        self,
        model_name: str | None = None,
        confidence: float | None = None,
        class_ids: list[int] | None = None,
    ) -> None:
        # Default to the custom shelf-product detector. Override with YOLO_MODEL env var.
        # Use a bigger COCO backbone (yolo11l by default) only as a last resort —
        # generic COCO classes miss most packaged goods, but the larger backbone
        # at least improves recall on dense shelves. Users can override with
        # YOLO_FALLBACK_MODEL (e.g., "yolo11x.pt" for maximum recall, or
        # "yolo11m.pt" / "yolo11s.pt" for faster inference).
        _default_model = str(Path(__file__).resolve().parent / "models" / "best.pt")
        if not Path(_default_model).exists():
            _default_model = os.getenv("YOLO_FALLBACK_MODEL", "yolo11l.pt").strip() or "yolo11l.pt"
        self.model_name = model_name or os.getenv("YOLO_MODEL", _default_model)
        self.model_key = Path(self.model_name).name
        # Only models that output product-specific class names produce identity labels.
        # Both COCO models and shelf-detector models output generic/object classes.
        self.product_identity_labels = self.model_key not in _NON_IDENTITY_MODELS
        confidence_env = os.getenv("YOLO_CONFIDENCE", "").strip()
        if confidence_env:
            self.confidence = float(confidence_env)
        elif confidence is not None:
            self.confidence = confidence
        elif self.model_key in _SHELF_DETECTOR_MODELS:
            # Favor recall for the grocery shelf detector; product naming can demote
            # bad boxes later, but missed products cannot be recovered downstream.
            self.confidence = 0.10
        else:
            self.confidence = 0.20
        imgsz_env = os.getenv("YOLO_IMGSZ", "").strip()
        if imgsz_env:
            self.imgsz = max(320, int(imgsz_env))
        elif self.model_key in _SHELF_DETECTOR_MODELS:
            self.imgsz = 960
        else:
            self.imgsz = 640
        self._crop_identifier = ProductIdentifier()
        self._ocr_crop_enabled = (
            self._crop_identifier.enabled
            and cv2 is not None
            and np is not None
        )
        # Use OCR text proposals for detection only when no specialized shelf model is
        # available — best.pt already finds product bounding boxes reliably.
        self._use_ocr_text_detector = (
            not self.product_identity_labels
            and self.model_key not in _SHELF_DETECTOR_MODELS
            and self._ocr_crop_enabled
        )
        self.label_source = (
            PRODUCT_LABEL_SOURCE
            if self.product_identity_labels
            else (OCR_TEXT_PROPOSAL_SOURCE if self._use_ocr_text_detector else GENERIC_LABEL_SOURCE)
        )
        self.detection_strategy = (
            OCR_TEXT_PROPOSAL_SOURCE
            if self._use_ocr_text_detector
            else "yolo_detector"
        )
        self.max_text_anchors = max(
            1,
            int(os.getenv("PRODUCT_DETECTION_TEXT_ANCHOR_LIMIT", "48")),
        )
        self.max_text_detections = max(
            1,
            int(os.getenv("PRODUCT_DETECTION_TEXT_MAX_DETECTIONS", "40")),
        )
        self.segment_min_pixels = max(
            1,
            int(os.getenv("PRODUCT_DETECTION_SEGMENT_MIN_PIXELS", "40000")),
        )
        self.segment_max_pixels = max(
            self.segment_min_pixels,
            int(os.getenv("PRODUCT_DETECTION_SEGMENT_MAX_PIXELS", "250000")),
        )
        self.segment_max_side = max(
            1,
            int(os.getenv("PRODUCT_DETECTION_SEGMENT_MAX_SIDE", "900")),
        )
        self.segment_iterations = max(
            1,
            int(os.getenv("PRODUCT_DETECTION_SEGMENT_ITERS", "2")),
        )
        self.image_stride = max(
            1,
            int(os.getenv("PRODUCT_DETECTION_IMAGE_STRIDE", "2" if self._use_ocr_text_detector else "1")),
        )

        # Class filter: explicit arg > YOLO_CLASS_IDS env var > auto (COCO model) > none.
        # best.pt only has one class (object), so no filter is needed.
        env_ids = os.getenv("YOLO_CLASS_IDS", "").strip()
        if class_ids is not None:
            self.class_ids: list[int] | None = class_ids
        elif env_ids:
            self.class_ids = [int(x.strip()) for x in env_ids.split(",") if x.strip()]
        elif self.model_key in _COCO_MODEL_NAMES:
            self.class_ids = sorted(GROCERY_COCO_CLASS_IDS)
        else:
            self.class_ids = None

        # Warm the model now so the first analyze call doesn't pay load latency.
        self._model = None
        if not self._use_ocr_text_detector and YOLO is not None:
            self._get_model()

    def labeling_metadata(self) -> dict:
        if self.product_identity_labels:
            note = "Labels come from a product-specific detector."
        elif self._use_ocr_text_detector:
            note = "Detections come from OCR-guided product proposals, and labels should not be treated as product names."
        else:
            note = "Detections come from a generic shelf-product detector; product names are added later via OCR or embeddings."

        return {
            "model_name": self.model_name,
            "model_key": self.model_key,
            "label_source": self.label_source,
            "product_identity_labels": self.product_identity_labels,
            "detection_strategy": self.detection_strategy,
            "note": note,
        }

    def detect_job(self, job_dir: Path) -> list[dict]:
        cached = self._load_cached_detections(job_dir, expected_strategy=self.detection_strategy)
        if cached:
            return self._ensure_focus_crops(job_dir, cached)

        images_dir = job_dir / "images"
        if not images_dir.exists():
            raise RuntimeError(
                "No cached detections were found and the job is missing an images/ directory."
            )

        if not self._use_ocr_text_detector and YOLO is None:
            raise RuntimeError(
                "ultralytics is not installed. Install backend/requirements-automation.txt "
                "or pre-populate detections/ JSON files."
            )

        model = None if self._use_ocr_text_detector else self._get_model()
        detections_dir = job_dir / "detections"
        crops_dir = job_dir / "crops"
        focus_crops_dir = job_dir / "focus_crops"
        detections_dir.mkdir(parents=True, exist_ok=True)
        crops_dir.mkdir(parents=True, exist_ok=True)
        focus_crops_dir.mkdir(parents=True, exist_ok=True)

        image_paths = sorted(
            path for path in images_dir.iterdir() if path.suffix.lower() in {".jpg", ".jpeg", ".png"}
        )
        if self.image_stride > 1:
            image_paths = image_paths[::self.image_stride]

        detector_meta = self.labeling_metadata()

        def _process(img_path: Path) -> tuple[Path, list[dict]]:
            return img_path, self._detect_single_image(model, img_path, crops_dir, focus_crops_dir)

        detections: list[dict] = []
        env_workers = int(os.getenv("PRODUCT_DETECTION_MAX_WORKERS", "0") or "0")
        if env_workers > 0:
            max_workers = min(env_workers, len(image_paths)) if image_paths else 1
        elif self._use_ocr_text_detector:
            max_workers = min(4, len(image_paths)) if image_paths else 1
        else:
            # Ultralytics model instances are not safe to share across threads during
            # predict() on this checkpoint; use serial inference unless explicitly
            # overridden.
            max_workers = 1
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {pool.submit(_process, p): p for p in image_paths}
            for future in as_completed(futures):
                img_path, image_detections = future.result()
                (detections_dir / f"{img_path.stem}.json").write_text(
                    json.dumps(
                        {
                            "image_name": img_path.name,
                            "detector": detector_meta,
                            "detections": image_detections,
                        },
                        indent=2,
                    ),
                    encoding="utf-8",
                )
                detections.extend(image_detections)

        return detections

    def _get_model(self):
        if self._model is None:
            self._model = YOLO(self.model_name)
        return self._model

    def _detect_single_image(
        self,
        model,
        image_path: Path,
        crops_dir: Path,
        focus_crops_dir: Path,
    ) -> list[dict]:
        if self._use_ocr_text_detector:
            with Image.open(image_path) as image:
                rgb = image.convert("RGB")
                text_lines = self._ocr_text_lines(rgb)
                return self._detect_single_image_from_text(
                    image_path=image_path,
                    image=rgb,
                    text_lines=text_lines,
                    crops_dir=crops_dir,
                    focus_crops_dir=focus_crops_dir,
                )

        predict_kwargs: dict = {
            "source": str(image_path),
            "conf": self.confidence,
            "imgsz": self.imgsz,
            "iou": 0.45,
            "agnostic_nms": True,
            "verbose": False,
        }
        if self.class_ids is not None:
            predict_kwargs["classes"] = self.class_ids
        results = model.predict(**predict_kwargs)
        names = results[0].names

        detections: list[dict] = []
        with Image.open(image_path) as image:
            rgb = image.convert("RGB")
            img_w, img_h = rgb.size
            # Only run scene-level OCR when focus-crop refinement is actually possible.
            text_lines = self._ocr_text_lines(rgb) if self._ocr_crop_enabled else []
            for index, box in enumerate(results[0].boxes):
                x1, y1, x2, y2 = [float(v) for v in box.xyxy[0].tolist()]
                # PIL.Image.crop requires integer pixel coordinates
                cx1 = max(0, int(round(x1)))
                cy1 = max(0, int(round(y1)))
                cx2 = min(img_w, int(round(x2)))
                cy2 = min(img_h, int(round(y2)))
                detection_id = f"{image_path.stem}-{index:03d}"
                raw_crop_path = crops_dir / f"{detection_id}.jpg"
                rgb.crop((cx1, cy1, cx2, cy2)).save(raw_crop_path, quality=92)
                focus_crop_path = self._save_focus_crop(
                    rgb,
                    focus_crops_dir,
                    detection_id,
                    (x1, y1, x2, y2),
                    text_lines=text_lines,
                )

                cls_id = int(box.cls[0].item())
                model_label = names.get(cls_id, str(cls_id))
                display_label = model_label if self.product_identity_labels else "Detected product"
                detections.append(
                    {
                        "id": detection_id,
                        "image_name": image_path.name,
                        "label": display_label,
                        "display_label": display_label,
                        "model_label": model_label,
                        "label_source": self.label_source,
                        "product_identity_label": self.product_identity_labels,
                        "confidence": float(box.conf[0].item()),
                        "bbox": [x1, y1, x2, y2],
                        "crop_path": str(focus_crop_path),
                        "raw_crop_path": str(raw_crop_path),
                    }
                )

        return detections

    def _detect_single_image_from_text(
        self,
        image_path: Path,
        image: Image.Image,
        text_lines: list,
        crops_dir: Path,
        focus_crops_dir: Path,
    ) -> list[dict]:
        if not text_lines:
            return []

        ranked_anchors = sorted(text_lines, key=self._text_anchor_sort_key, reverse=True)
        candidate_anchors = [
            anchor
            for anchor in ranked_anchors
            if self._line_has_brand_token(anchor.text)
            or self._crop_identifier._line_score(anchor) >= 0.48
        ]
        if not candidate_anchors:
            candidate_anchors = ranked_anchors[:self.max_text_anchors]
        else:
            candidate_anchors = candidate_anchors[:self.max_text_anchors]

        proposals: list[tuple[float, tuple[int, int, int, int], tuple[int, int, int, int], str]] = []
        for anchor in candidate_anchors:
            group = self._group_text_lines(text_lines, anchor)
            if not group:
                continue

            score = self._text_proposal_score(anchor, group)
            if score < 0.18:
                continue

            group_box = self._group_box_pixels(group, image.size)
            synthetic_bbox = group_box
            seed_window = self._ocr_seed_window(group, synthetic_bbox, image.size)
            refined = self._segment_product_bounds(image, seed_window, group_box) or seed_window
            if not self._text_proposal_is_valid(refined, image.size):
                continue

            anchor_text = group[0].text
            proposals.append((score, seed_window, refined, anchor_text))

        proposals.sort(key=lambda item: item[0], reverse=True)
        detections: list[dict] = []
        kept_bounds: list[tuple[int, int, int, int]] = []
        for score, seed_window, refined, anchor_text in proposals:
            if any(self._bounds_iou(refined, kept) >= 0.45 for kept in kept_bounds):
                continue

            detection_id = f"{image_path.stem}-{len(detections):03d}"
            raw_crop_path = crops_dir / f"{detection_id}.jpg"
            focus_crop_path = focus_crops_dir / f"{detection_id}.jpg"
            image.crop(seed_window).save(raw_crop_path, quality=92)
            image.crop(refined).save(focus_crop_path, quality=92)

            detections.append(
                {
                    "id": detection_id,
                    "image_name": image_path.name,
                    "label": "Detected product",
                    "display_label": "Detected product",
                    "model_label": anchor_text,
                    "label_source": self.label_source,
                    "product_identity_label": False,
                    "confidence": float(score),
                    "bbox": [float(value) for value in refined],
                    "crop_path": str(focus_crop_path),
                    "raw_crop_path": str(raw_crop_path),
                }
            )
            kept_bounds.append(refined)
            if len(detections) >= self.max_text_detections:
                break

        return detections

    def _ensure_focus_crops(self, job_dir: Path, detections: list[dict]) -> list[dict]:
        images_dir = job_dir / "images"
        if not images_dir.exists():
            return detections

        focus_crops_dir = job_dir / "focus_crops"
        focus_crops_dir.mkdir(parents=True, exist_ok=True)

        detections_by_image: dict[str, list[dict]] = {}
        for detection in detections:
            detections_by_image.setdefault(detection["image_name"], []).append(detection)

        for image_name, image_detections in detections_by_image.items():
            image_path = images_dir / image_name
            if not image_path.exists():
                continue

            with Image.open(image_path) as image:
                rgb = image.convert("RGB")
                text_lines = self._ocr_text_lines(rgb)
                for detection in image_detections:
                    raw_crop_path = detection.get("raw_crop_path") or detection.get("crop_path")
                    if raw_crop_path:
                        detection["raw_crop_path"] = raw_crop_path

                    focus_crop_path = self._save_focus_crop(
                        rgb,
                        focus_crops_dir,
                        detection["id"],
                        tuple(detection["bbox"]),
                        text_lines=text_lines,
                    )
                    detection["crop_path"] = str(focus_crop_path)

        return detections

    def _save_focus_crop(
        self,
        image: Image.Image,
        focus_crops_dir: Path,
        detection_id: str,
        bbox: tuple[float, float, float, float],
        text_lines: list | None = None,
    ) -> Path:
        crop_bounds = self._focus_crop_bounds(
            bbox,
            image=image,
            image_size=image.size,
            text_lines=text_lines or [],
        )
        crop_bounds = self._cap_focus_crop_growth(
            crop_bounds,
            bbox=bbox,
            image_size=image.size,
        )
        focus_crop_path = focus_crops_dir / f"{detection_id}.jpg"
        image.crop(crop_bounds).save(focus_crop_path, quality=92)
        return focus_crop_path

    def _focus_crop_bounds(
        self,
        bbox: tuple[float, float, float, float],
        image: Image.Image,
        image_size: tuple[int, int],
        text_lines: list,
    ) -> tuple[int, int, int, int]:
        x1, y1, x2, y2 = bbox
        img_w, img_h = image_size
        box_w = max(1.0, x2 - x1)
        box_h = max(1.0, y2 - y1)
        touches_boundary = (
            x1 <= img_w * 0.02
            or y1 <= img_h * 0.02
            or x2 >= img_w * 0.98
            or y2 >= img_h * 0.98
        )

        if (
            img_w * 0.12 <= box_w <= img_w * 0.32
            and img_h * 0.16 <= box_h <= img_h * 0.36
            and not touches_boundary
        ):
            return (
                max(0, int(round(x1))),
                max(0, int(round(y1))),
                min(img_w, int(round(x2))),
                min(img_h, int(round(y2))),
            )

        ocr_bounds = self._ocr_refined_crop_bounds(
            bbox=bbox,
            image=image,
            image_size=image_size,
            text_lines=text_lines,
        )
        if ocr_bounds is not None:
            return ocr_bounds

        return self._default_focus_crop_bounds(bbox, image_size)

    def _default_focus_crop_bounds(
        self,
        bbox: tuple[float, float, float, float],
        image_size: tuple[int, int],
    ) -> tuple[int, int, int, int]:
        x1, y1, x2, y2 = bbox
        img_w, img_h = image_size
        box_w = max(1.0, x2 - x1)
        box_h = max(1.0, y2 - y1)
        center_x = 0.5 * (x1 + x2)
        center_y = 0.5 * (y1 + y2)
        # Never crop tighter than the bbox itself; cap expansion for small boxes only
        focus_w = max(box_w * 1.05, min(img_w * 0.12, max(img_w * 0.04, box_w * 1.10)))
        focus_h = max(box_h * 1.05, min(img_h * 0.16, max(img_h * 0.05, box_h * 1.15)))

        if focus_h < focus_w * 1.05:
            focus_h = max(box_h * 1.05, min(img_h * 0.16, focus_w * 1.15))
        if focus_w > focus_h * 0.92:
            focus_w = max(box_w * 1.05, max(img_w * 0.04, focus_h * 0.90))

        left = max(0, int(round(center_x - 0.5 * focus_w)))
        top = max(0, int(round(center_y - 0.5 * focus_h)))
        right = min(img_w, int(round(left + focus_w)))
        bottom = min(img_h, int(round(top + focus_h)))

        if right - left < self._minimum_focus_width(img_w):
            left = max(0, int(round(center_x - 0.5 * self._minimum_focus_width(img_w))))
            right = min(img_w, int(round(left + self._minimum_focus_width(img_w))))
        if bottom - top < self._minimum_focus_height(img_h):
            top = max(0, int(round(center_y - 0.5 * self._minimum_focus_height(img_h))))
            bottom = min(img_h, int(round(top + self._minimum_focus_height(img_h))))

        return left, top, right, bottom

    def _cap_focus_crop_growth(
        self,
        crop_bounds: tuple[int, int, int, int],
        bbox: tuple[float, float, float, float],
        image_size: tuple[int, int],
    ) -> tuple[int, int, int, int]:
        crop_left, crop_top, crop_right, crop_bottom = crop_bounds
        crop_width = max(1, crop_right - crop_left)
        crop_height = max(1, crop_bottom - crop_top)
        x1, y1, x2, y2 = bbox
        box_width = max(1.0, x2 - x1)
        box_height = max(1.0, y2 - y1)
        img_width, img_height = image_size

        max_width = max(self._minimum_focus_width(img_width), box_width * 1.30)
        max_height = max(self._minimum_focus_height(img_height), box_height * 1.35)
        if crop_width <= max_width and crop_height <= max_height:
            return crop_bounds

        return self._default_focus_crop_bounds(bbox, image_size)

    def _ocr_refined_crop_bounds(
        self,
        bbox: tuple[float, float, float, float],
        image: Image.Image,
        image_size: tuple[int, int],
        text_lines: list,
    ) -> tuple[int, int, int, int] | None:
        if not self._ocr_crop_enabled or not text_lines:
            return None

        anchor = self._select_text_anchor(text_lines, bbox, image_size)
        if anchor is None:
            return None

        group = self._group_text_lines(text_lines, anchor)
        if not group:
            return None

        seed_window = self._ocr_seed_window(group, bbox, image_size)
        group_box = self._group_box_pixels(group, image_size)
        refined = self._segment_product_bounds(image, seed_window, group_box)
        if refined is None:
            return seed_window
        return refined

    def _ocr_text_lines(self, image: Image.Image) -> list:
        if not self._ocr_crop_enabled:
            return []
        try:
            roi_image, roi_bounds = self._detection_ocr_region(image)
            observations = self._crop_identifier._ocr_lines(
                self._prepare_detection_ocr_image(roi_image)
            )
        except Exception:
            return []

        cleaned = [
            self._crop_identifier._clean_line(
                self._remap_detection_ocr_line(line, image.size, roi_bounds)
            )
            for line in observations
        ]
        return [line for line in cleaned if line is not None]

    def _prepare_detection_ocr_image(self, image: Image.Image) -> Image.Image:
        rgb = image.convert("RGB")
        width, height = rgb.size
        longest_side = max(width, height)
        max_side = 840
        if longest_side <= max_side:
            return rgb

        scale = max_side / float(longest_side)
        resized_size = (
            max(1, int(round(width * scale))),
            max(1, int(round(height * scale))),
        )
        return rgb.resize(resized_size, Image.Resampling.LANCZOS)

    def _detection_ocr_region(
        self,
        image: Image.Image,
    ) -> tuple[Image.Image, tuple[int, int, int, int]]:
        width, height = image.size
        left = int(round(width * 0.02))
        top = int(round(height * 0.14))
        right = int(round(width * 0.98))
        bottom = int(round(height * 0.88))
        bounds = (left, top, right, bottom)
        return image.crop(bounds), bounds

    def _remap_detection_ocr_line(
        self,
        line,
        image_size: tuple[int, int],
        roi_bounds: tuple[int, int, int, int],
    ):
        image_width, image_height = image_size
        roi_left, roi_top, roi_right, roi_bottom = roi_bounds
        roi_width = max(1, roi_right - roi_left)
        roi_height = max(1, roi_bottom - roi_top)
        roi_bottom_from_bottom = image_height - roi_bottom

        return line.__class__(
            text=line.text,
            min_x=(roi_left + (line.min_x * roi_width)) / float(image_width),
            min_y=(roi_bottom_from_bottom + (line.min_y * roi_height)) / float(image_height),
            width=(line.width * roi_width) / float(image_width),
            height=(line.height * roi_height) / float(image_height),
            confidence=line.confidence,
        )

    def _select_text_anchor(
        self,
        text_lines: list,
        bbox: tuple[float, float, float, float],
        image_size: tuple[int, int],
    ):
        img_w, img_h = image_size
        target_x = (bbox[0] + bbox[2]) / (2.0 * img_w)
        target_y = 1.0 - ((bbox[1] + bbox[3]) / (2.0 * img_h))

        best_line = None
        best_score = None
        for line in text_lines:
            distance = ((line.center_x - target_x) ** 2 + (line.center_y - target_y) ** 2) ** 0.5
            brand_bonus = 0.28 if self._line_has_brand_token(line.text) else 0.0
            score = brand_bonus + self._crop_identifier._line_score(line) - (distance * 1.25)
            if best_line is None or score > best_score:
                best_line = line
                best_score = score
        return best_line

    def _group_text_lines(self, text_lines: list, anchor) -> list:
        grouped = [
            line
            for line in text_lines
            if abs(line.center_x - anchor.center_x) <= max(0.07, anchor.width * 0.8)
            and abs(line.center_y - anchor.center_y) <= max(0.09, anchor.height * 4.0)
        ]
        grouped.sort(key=lambda item: (-item.center_y, item.min_x))
        return grouped

    def _ocr_seed_window(
        self,
        group: list,
        bbox: tuple[float, float, float, float],
        image_size: tuple[int, int],
    ) -> tuple[int, int, int, int]:
        img_w, img_h = image_size
        box_w = max(1.0, bbox[2] - bbox[0])
        box_h = max(1.0, bbox[3] - bbox[1])

        group_left, group_top, group_right, group_bottom = self._group_box_pixels(group, image_size)
        group_w = max(1.0, group_right - group_left)
        group_h = max(1.0, group_bottom - group_top)

        center_x = 0.5 * (group_left + group_right)
        crop_w = max(img_w * 0.15, group_w * 2.5)
        crop_h = max(img_h * 0.20, group_h * 6.5)

        if box_w < img_w * 0.42:
            crop_w = max(crop_w, box_w * 0.9)
        if box_h < img_h * 0.46:
            crop_h = max(crop_h, box_h * 0.9)

        crop_w = min(img_w * 0.26, crop_w)
        crop_h = min(img_h * 0.32, crop_h)
        if crop_h < crop_w * 1.35:
            crop_h = min(img_h * 0.32, crop_w * 1.45)

        center_y = group_top + (crop_h * 0.28)
        left = int(round(center_x - (0.5 * crop_w)))
        top = int(round(center_y - (0.5 * crop_h)))
        right = int(round(left + crop_w))
        bottom = int(round(top + crop_h))

        if left < 0:
            right -= left
            left = 0
        if top < 0:
            bottom -= top
            top = 0
        if right > img_w:
            left -= right - img_w
            right = img_w
        if bottom > img_h:
            top -= bottom - img_h
            bottom = img_h

        return max(0, left), max(0, top), min(img_w, right), min(img_h, bottom)

    def _group_box_pixels(
        self,
        group: list,
        image_size: tuple[int, int],
    ) -> tuple[float, float, float, float]:
        img_w, img_h = image_size
        left = min(line.min_x for line in group) * img_w
        right = max(line.min_x + line.width for line in group) * img_w
        top = min(1.0 - (line.min_y + line.height) for line in group) * img_h
        bottom = max(1.0 - line.min_y for line in group) * img_h
        return left, top, right, bottom

    def _segment_product_bounds(
        self,
        image: Image.Image,
        seed_window: tuple[int, int, int, int],
        group_box: tuple[float, float, float, float],
    ) -> tuple[int, int, int, int] | None:
        if cv2 is None or np is None:
            return None

        crop_left, crop_top, crop_right, crop_bottom = seed_window
        local_width = crop_right - crop_left
        local_height = crop_bottom - crop_top
        if local_width <= 0 or local_height <= 0:
            return None

        local_pixels = local_width * local_height

        # GrabCut adds marginal value for tiny windows and becomes prohibitively slow
        # on very large OCR seed windows. In both cases, keep the seed crop.
        if local_pixels < self.segment_min_pixels:
            return None
        if local_pixels > self.segment_max_pixels:
            return None
        if local_width > self.segment_max_side or local_height > self.segment_max_side:
            return None

        rgb = image.crop(seed_window).convert("RGB")
        bgr = cv2.cvtColor(np.array(rgb), cv2.COLOR_RGB2BGR)

        group_left, group_top, group_right, group_bottom = group_box
        group_left -= crop_left
        group_right -= crop_left
        group_top -= crop_top
        group_bottom -= crop_top
        if group_right <= group_left or group_bottom <= group_top:
            return None

        mask = np.full((local_height, local_width), cv2.GC_PR_BGD, dtype=np.uint8)
        mask[:8, :] = cv2.GC_BGD
        mask[-8:, :] = cv2.GC_BGD
        mask[:, :8] = cv2.GC_BGD
        mask[:, -8:] = cv2.GC_BGD

        fg_left = max(10, int(round(group_left - ((group_right - group_left) * 0.35))))
        fg_right = min(local_width - 10, int(round(group_right + ((group_right - group_left) * 0.35))))
        fg_top = max(10, int(round(group_top - ((group_bottom - group_top) * 0.55))))
        fg_bottom = min(local_height - 10, int(round(group_bottom + ((group_bottom - group_top) * 1.50))))
        mask[fg_top:fg_bottom, fg_left:fg_right] = cv2.GC_PR_FGD

        core_left = max(10, int(round(group_left + ((group_right - group_left) * 0.10))))
        core_right = min(local_width - 10, int(round(group_right - ((group_right - group_left) * 0.10))))
        core_top = max(10, int(round(group_top - ((group_bottom - group_top) * 0.05))))
        core_bottom = min(local_height - 10, int(round(group_bottom + ((group_bottom - group_top) * 0.35))))
        mask[core_top:core_bottom, core_left:core_right] = cv2.GC_FGD

        background_model = np.zeros((1, 65), np.float64)
        foreground_model = np.zeros((1, 65), np.float64)
        try:
            cv2.grabCut(
                bgr,
                mask,
                None,
                background_model,
                foreground_model,
                self.segment_iterations,
                cv2.GC_INIT_WITH_MASK,
            )
        except cv2.error:
            return None

        foreground = np.where(
            (mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD),
            255,
            0,
        ).astype("uint8")
        foreground = cv2.morphologyEx(foreground, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
        foreground = cv2.morphologyEx(foreground, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))

        component_box = self._foreground_component_box(
            foreground,
            seed_x=int(round((group_left + group_right) * 0.5)),
            seed_y=int(round((group_top + group_bottom) * 0.5)),
        )
        if component_box is None:
            return None

        component_left, component_top, component_right, component_bottom = component_box
        component_w = max(1.0, component_right - component_left)
        component_h = max(1.0, component_bottom - component_top)
        group_w = max(1.0, group_right - group_left)
        group_h = max(1.0, group_bottom - group_top)

        refined_left = max(
            0,
            int(round(min(component_left - (component_w * 0.08), group_left - (group_w * 0.55)))),
        )
        refined_top = max(
            0,
            int(round(min(component_top - (component_h * 0.18), group_top - (group_h * 0.75)))),
        )
        refined_right = min(
            local_width,
            int(round(max(component_right + (component_w * 0.08), group_right + (group_w * 0.35)))),
        )
        refined_bottom = min(
            local_height,
            int(round(max(component_bottom + (component_h * 0.12), group_bottom + (group_h * 1.45)))),
        )

        return (
            int(crop_left + refined_left),
            int(crop_top + refined_top),
            int(crop_left + refined_right),
            int(crop_top + refined_bottom),
        )

    def _foreground_component_box(
        self,
        foreground,
        seed_x: int,
        seed_y: int,
    ) -> tuple[int, int, int, int] | None:
        if cv2 is None:
            return None

        component_count, labels, stats, _ = cv2.connectedComponentsWithStats(foreground)
        if component_count <= 1:
            return None

        clamped_x = min(max(seed_x, 0), foreground.shape[1] - 1)
        clamped_y = min(max(seed_y, 0), foreground.shape[0] - 1)
        component_id = int(labels[clamped_y, clamped_x])

        if component_id == 0:
            best_component = None
            best_score = None
            for index in range(1, component_count):
                x, y, width, height, area = stats[index]
                if area < 5000:
                    continue
                center_x = x + (0.5 * width)
                center_y = y + (0.5 * height)
                distance = ((center_x - clamped_x) ** 2 + (center_y - clamped_y) ** 2) ** 0.5
                score = float(area) - (distance * 35.0)
                if best_component is None or score > best_score:
                    best_component = index
                    best_score = score
            component_id = best_component or 0
            if component_id == 0:
                return None

        x, y, width, height, _ = stats[component_id]
        return int(x), int(y), int(x + width), int(y + height)

    def _line_has_brand_token(self, text: str) -> bool:
        return any(token in _OCR_BRAND_TOKENS for token in text.split())

    def _text_anchor_sort_key(self, line) -> tuple[int, float, float]:
        return (
            1 if self._line_has_brand_token(line.text) else 0,
            self._crop_identifier._line_score(line),
            line.area,
        )

    def _text_proposal_score(self, anchor, group: list) -> float:
        score = self._crop_identifier._line_score(anchor)
        if self._line_has_brand_token(anchor.text):
            score += 0.22
        if any(self._line_has_brand_token(line.text) for line in group):
            score += 0.10
        score += min(0.12, 0.04 * max(0, len(group) - 1))
        return max(0.0, min(0.99, score))

    def _text_proposal_is_valid(
        self,
        bounds: tuple[int, int, int, int],
        image_size: tuple[int, int],
    ) -> bool:
        left, top, right, bottom = bounds
        width = max(1, right - left)
        height = max(1, bottom - top)
        image_width, image_height = image_size
        area_ratio = (width * height) / float(image_width * image_height)
        aspect_ratio = width / float(height)

        if width < image_width * 0.08 or height < image_height * 0.10:
            return False
        if area_ratio > 0.16:
            return False
        if aspect_ratio < 0.22 or aspect_ratio > 1.55:
            return False
        return True

    def _bounds_iou(
        self,
        first: tuple[int, int, int, int],
        second: tuple[int, int, int, int],
    ) -> float:
        ax1, ay1, ax2, ay2 = first
        bx1, by1, bx2, by2 = second
        inter_left = max(ax1, bx1)
        inter_top = max(ay1, by1)
        inter_right = min(ax2, bx2)
        inter_bottom = min(ay2, by2)
        inter_width = max(0, inter_right - inter_left)
        inter_height = max(0, inter_bottom - inter_top)
        intersection = inter_width * inter_height
        if intersection <= 0:
            return 0.0

        first_area = max(1, (ax2 - ax1) * (ay2 - ay1))
        second_area = max(1, (bx2 - bx1) * (by2 - by1))
        union = first_area + second_area - intersection
        return intersection / float(union)

    def _minimum_focus_width(self, image_width: int) -> float:
        return image_width * 0.04

    def _minimum_focus_height(self, image_height: int) -> float:
        return image_height * 0.05

    def _load_cached_detections(
        self,
        job_dir: Path,
        expected_strategy: str | None = None,
    ) -> list[dict]:
        detections_dir = job_dir / "detections"
        if not detections_dir.exists():
            return []

        detections: list[dict] = []
        for path in sorted(detections_dir.glob("*.json")):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError as exc:
                raise RuntimeError(
                    f"Malformed detection cache file '{path.name}': {exc}. "
                    "Delete the detections/ directory to re-run detection."
                ) from exc
            if isinstance(payload, list):
                if expected_strategy is not None:
                    return []
                raw_detections = payload
                detector_metadata = None
            else:
                raw_detections = payload.get("detections", [])
                detector_metadata = payload.get("detector") if isinstance(payload.get("detector"), dict) else None
                if (
                    expected_strategy is not None
                    and detector_metadata
                    and detector_metadata.get("detection_strategy") != expected_strategy
                ):
                    return []
                if expected_strategy is not None and detector_metadata:
                    cached_model = detector_metadata.get("model_key") or detector_metadata.get("model_name")
                    if cached_model and Path(str(cached_model)).name != self.model_key:
                        return []

            detections.extend(
                self._normalize_detection(detection, detector_metadata)
                for detection in raw_detections
            )

        return detections

    def _normalize_detection(self, detection: dict, detector_metadata: dict | None) -> dict:
        raw_label = (
            detection.get("model_label")
            or detection.get("raw_label")
            or detection.get("label")
            or "object"
        )
        product_identity_label = detection.get("product_identity_label")
        if product_identity_label is None:
            if detector_metadata and "product_identity_labels" in detector_metadata:
                product_identity_label = bool(detector_metadata["product_identity_labels"])
            else:
                product_identity_label = self.product_identity_labels

        label_source = detection.get("label_source")
        if not label_source:
            if detector_metadata and detector_metadata.get("label_source"):
                label_source = str(detector_metadata["label_source"])
            else:
                label_source = self.label_source

        display_label = detection.get("display_label")
        if not display_label:
            display_label = raw_label if product_identity_label else "Detected product"

        normalized = dict(detection)
        normalized["label"] = display_label
        normalized["display_label"] = display_label
        normalized["model_label"] = raw_label
        normalized["label_source"] = label_source
        normalized["product_identity_label"] = bool(product_identity_label)
        normalized["raw_crop_path"] = detection.get("raw_crop_path")
        return normalized
