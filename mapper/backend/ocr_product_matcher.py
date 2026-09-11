"""
OCR-based product name extraction from shelf crop images.

Uses EasyOCR to read text from a crop, then:
1. Fuzzy-matches against knowledge-base product names  → "ocr_kb_match"
2. Returns raw OCR text when no KB match found         → "ocr_text_only"

Priority in the pipeline:
  OCR KB match (conf >= threshold)  >  OCR + propagation agree  >  propagation alone
"""

from __future__ import annotations

import re
import tempfile
from difflib import SequenceMatcher
from pathlib import Path

from PIL import Image


_STOP_WORDS = {
    "a", "an", "the", "and", "or", "of", "in", "on", "at", "to", "for",
    "with", "by", "as", "oz", "g", "lb", "lbs", "ct", "pk", "pack",
    "new", "free", "natural", "organic", "made",
}


def _normalize(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9& ]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _token_set(text: str) -> set[str]:
    return {t for t in _normalize(text).split() if t not in _STOP_WORDS and len(t) > 1}


def _score(ocr_text: str, product_name: str) -> float:
    """
    Combines token overlap (0.6 weight) + sequence ratio (0.4 weight).
    Token overlap = intersection / product_name_tokens (precision on product side).
    """
    ocr_tokens = _token_set(ocr_text)
    prod_tokens = _token_set(product_name)
    if not prod_tokens or not ocr_tokens:
        return 0.0
    token_overlap = len(ocr_tokens & prod_tokens) / len(prod_tokens)
    seq = SequenceMatcher(None, _normalize(ocr_text), _normalize(product_name)).ratio()
    return round(token_overlap * 0.65 + seq * 0.35, 4)


class OCRProductMatcher:
    """Lazy-initialised EasyOCR reader + KB fuzzy matcher."""

    def __init__(
        self,
        kb_dir: Path | None = None,
        gpu: bool | None = None,
        kb_match_threshold: float = 0.52,
    ) -> None:
        self._reader = None
        self._gpu = gpu  # None = auto-detect on first use
        self.threshold = kb_match_threshold
        self._product_names: list[str] = []
        if kb_dir:
            self.load_kb(Path(kb_dir))

    def load_kb(self, kb_dir: Path) -> None:
        self._product_names = sorted(
            d.name
            for d in kb_dir.iterdir()
            if d.is_dir() and not d.name.startswith(("open_world", "_"))
        )

    def _get_reader(self):
        if self._reader is None:
            import easyocr
            if self._gpu is None:
                try:
                    import torch
                    self._gpu = torch.cuda.is_available()
                except ImportError:
                    self._gpu = False
            self._reader = easyocr.Reader(["en"], gpu=self._gpu, verbose=False)
        return self._reader

    def _upscale(self, image: Image.Image, min_side: int = 320) -> Image.Image:
        w, h = image.size
        shortest = min(w, h)
        if shortest < min_side:
            scale = min_side / shortest
            return image.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
        return image

    def extract_text(self, image: Image.Image) -> list[tuple[str, float]]:
        """Return list of (text, confidence) from EasyOCR, sorted by confidence desc."""
        image = self._upscale(image.convert("RGB"))
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
            tmp = Path(f.name)
        try:
            image.save(tmp, format="JPEG", quality=92)
            results = self._get_reader().readtext(str(tmp), detail=1)
            return sorted(
                [(str(r[1]), float(r[2])) for r in results if float(r[2]) > 0.3],
                key=lambda x: -x[1],
            )
        finally:
            tmp.unlink(missing_ok=True)

    def match(self, image: Image.Image) -> dict:
        """
        Run OCR and fuzzy-match against KB.

        Returns dict with keys:
          product_name  — KB name if matched, else None
          ocr_text      — all OCR lines joined (useful even when no KB match)
          confidence    — match score 0–1
          source        — "ocr_kb_match" | "ocr_text_only" | "ocr_no_text"
        """
        lines = self.extract_text(image)
        if not lines:
            return {"product_name": None, "ocr_text": "", "confidence": 0.0, "source": "ocr_no_text"}

        full_text = " ".join(t for t, _ in lines)

        if not self._product_names:
            return {
                "product_name": None,
                "ocr_text": full_text,
                "confidence": 0.0,
                "source": "ocr_text_only",
            }

        best_name, best_score = max(
            ((name, _score(full_text, name)) for name in self._product_names),
            key=lambda x: x[1],
        )

        if best_score >= self.threshold:
            return {
                "product_name": best_name,
                "ocr_text": full_text,
                "confidence": best_score,
                "source": "ocr_kb_match",
            }

        return {
            "product_name": None,
            "ocr_text": full_text,
            "confidence": best_score,
            "source": "ocr_text_only",
        }
