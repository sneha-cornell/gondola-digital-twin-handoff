from __future__ import annotations

import base64
import json
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

VLM_SOURCE = "anthropic_vlm_fallback"
VLM_OPEN_SOURCE = "anthropic_vlm_open"

_DISABLE_VALUES = {"0", "false", "no", "off"}
_IMAGE_MEDIA_TYPES = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".webp": "image/webp",
    ".gif": "image/gif",
}


class VLMIdentifier:
    """Anthropic vision-language fallback for crops the OCR/embedding stack
    could not name confidently. Activated only when `ANTHROPIC_API_KEY` is set."""

    def __init__(self, catalog) -> None:
        self.api_key = os.getenv("ANTHROPIC_API_KEY", "").strip()
        enabled_flag = os.getenv("VLM_FALLBACK_ENABLED", "1").strip().lower()
        self.model = os.getenv("VLM_FALLBACK_MODEL", "claude-sonnet-4-6").strip() or "claude-sonnet-4-6"
        self.confidence_threshold = float(os.getenv("VLM_FALLBACK_THRESHOLD", "0.92"))
        self.accept_threshold = float(os.getenv("VLM_FALLBACK_ACCEPT_THRESHOLD", "0.80"))
        self.max_tokens = int(os.getenv("VLM_FALLBACK_MAX_TOKENS", "300"))
        self._catalog = catalog
        self._client = None
        self._catalog_names: tuple[str, ...] = tuple(catalog.product_names())

        # Optional local (open-source) VLM backend. When ``LOCAL_VLM_ENABLED=1``
        # is set, both .identify and .identify_open delegate to a Qwen2.5-VL
        # model running locally instead of calling Anthropic. The local
        # backend is enabled even when ANTHROPIC_API_KEY is missing, since it
        # has no upstream dependency.
        self._local_backend = None
        local_env = os.getenv("LOCAL_VLM_ENABLED", "0").strip().lower()
        self._local_backend_requested = local_env not in _DISABLE_VALUES
        if self._local_backend_requested:
            try:
                from local_vlm import LocalVLM
            except ImportError as exc:
                logger.warning(
                    "LOCAL_VLM_ENABLED=1 but local_vlm could not be imported: %s",
                    exc,
                )
            else:
                self._local_backend = LocalVLM(catalog)

        # The aggregate "enabled" flag is true when *any* backend can run:
        # either the local Qwen model, or the Anthropic API.
        anthropic_enabled = bool(self.api_key) and enabled_flag not in _DISABLE_VALUES
        self._anthropic_enabled = anthropic_enabled
        self.enabled = anthropic_enabled or bool(
            self._local_backend and self._local_backend.enabled
        )

    def _ensure_client(self):
        if self._client is None:
            try:
                from anthropic import Anthropic
            except ImportError as exc:
                raise RuntimeError(
                    "anthropic package not installed. Add `anthropic` to backend/requirements.txt."
                ) from exc
            self._client = Anthropic(api_key=self.api_key)
        return self._client

    def _system_prompt(self) -> list[dict]:
        catalog_lines = "\n".join(f"- {name}" for name in self._catalog_names)
        text = (
            "You identify packaged retail products from cropped shelf photos.\n"
            "You receive: a single product crop, OCR text excerpts, and candidate "
            "product names produced by earlier pipeline stages.\n\n"
            "CATALOG (return product_name verbatim from this list, or null):\n"
            f"{catalog_lines}\n\n"
            "Rules:\n"
            "1. Return product_name exactly as listed above, or null if no catalog "
            "product is a confident visual match.\n"
            "2. Prefer null over guessing. Confidence below 0.6 means null.\n"
            "3. Use packaging color, brand mark, and flavor text to disambiguate "
            "similar SKUs. The OCR text may be wrong — trust the image first.\n"
            "4. Reply with strict JSON only: "
            '{"product_name": <string or null>, "confidence": <0.0-1.0>, '
            '"reason": <short string>}.'
        )
        return [{"type": "text", "text": text, "cache_control": {"type": "ephemeral"}}]

    def identify(
        self,
        crop_path: Path,
        ocr_text: str | None,
        candidates: list[str],
    ) -> dict | None:
        if not self.enabled:
            return None

        # Local backend takes priority when enabled — keeps everything on-box
        # and avoids any Anthropic billing.
        if self._local_backend is not None and self._local_backend.enabled:
            return self._local_backend.identify(crop_path, ocr_text, candidates)

        if not self._anthropic_enabled:
            return None

        try:
            client = self._ensure_client()
            with open(crop_path, "rb") as fh:
                image_bytes = fh.read()
            image_b64 = base64.standard_b64encode(image_bytes).decode("ascii")
            media_type = _IMAGE_MEDIA_TYPES.get(crop_path.suffix.lower(), "image/jpeg")

            user_lines: list[str] = []
            if ocr_text:
                user_lines.append(f"OCR text: {ocr_text}")
            if candidates:
                trimmed = [c for c in candidates if c][:5]
                if trimmed:
                    user_lines.append("Top candidates from earlier stages: " + "; ".join(trimmed))
            user_lines.append(
                'Identify this product. Reply with strict JSON only: '
                '{"product_name": "<exact catalog name or null>", '
                '"confidence": <0.0-1.0>, "reason": "<short>"}'
            )

            response = client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                system=self._system_prompt(),
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": media_type,
                                    "data": image_b64,
                                },
                            },
                            {"type": "text", "text": "\n".join(user_lines)},
                        ],
                    }
                ],
            )
        except Exception as exc:
            return {
                "product_name": None,
                "exact_match": False,
                "confidence": 0.0,
                "reason": f"VLM call failed: {exc}",
                "source": VLM_SOURCE,
                "error": str(exc),
            }

        text = "".join(getattr(block, "text", "") for block in response.content)
        payload = self._parse_json(text)
        if payload is None:
            return {
                "product_name": None,
                "exact_match": False,
                "confidence": 0.0,
                "reason": "VLM response was not valid JSON.",
                "source": VLM_SOURCE,
                "raw": text[:500],
            }

        product_name = payload.get("product_name")
        if not isinstance(product_name, str) or not product_name.strip():
            product_name = None
        else:
            product_name = product_name.strip()
            if product_name not in self._catalog_names:
                product_name = None

        try:
            confidence = float(payload.get("confidence") or 0.0)
        except (TypeError, ValueError):
            confidence = 0.0
        confidence = max(0.0, min(1.0, confidence))

        accepted = bool(product_name) and confidence >= self.accept_threshold
        return {
            "product_name": product_name if accepted else None,
            "exact_match": accepted,
            "confidence": confidence,
            "reason": str(payload.get("reason") or "VLM fallback identification."),
            "source": VLM_SOURCE,
            "raw_product_name": payload.get("product_name"),
        }

    # ------------------------------------------------------------------
    # Open-mode identification (no catalog constraint)
    # ------------------------------------------------------------------

    def _open_system_prompt(self) -> list[dict]:
        text = (
            "You identify packaged retail products from cropped shelf photos.\n"
            "Unlike the closed-set mode, you may name ANY commercially available "
            "grocery / packaged consumer product on Earth (food, beverages, "
            "snacks, cleaning products, personal care, etc.).\n\n"
            "You receive: a single product crop and (optionally) OCR text "
            "excerpts plus prior-stage candidate names. The candidate names "
            "are hints — they may be wrong; do not feel obliged to use them.\n\n"
            "Rules:\n"
            "1. Identify the product as precisely as you can: brand, product "
            "line, and the exact flavor/variant.\n"
            "2. Use packaging color, brand mark, logo, and visible text to "
            "disambiguate similar SKUs.\n"
            "3. OCR text may be wrong — trust the image first.\n"
            "4. If you cannot read the brand or product line confidently, "
            "return null for that field; do NOT invent a brand.\n"
            "5. Confidence is your own self-assessed likelihood that the "
            "product_name is exactly correct (not just \"plausibly close\").\n"
            "6. Reply with strict JSON only: "
            '{"brand": <string or null>, "product_name": <string or null>, '
            '"variant": <string or null>, "confidence": <0.0-1.0>, '
            '"reason": <short string>}\n'
            "   - product_name should be a complete consumer-facing name "
            "without the brand prefix (e.g., \"Classic Potato Chips\" not "
            "\"Lay's Classic Potato Chips\").\n"
            "   - variant captures size / flavor modifier when visible "
            "(e.g., \"Family Size\", \"Spicy\")."
        )
        return [{"type": "text", "text": text, "cache_control": {"type": "ephemeral"}}]

    def identify_open(
        self,
        crop_path: Path,
        ocr_text: str | None = None,
        candidates: list[str] | None = None,
    ) -> dict | None:
        """Free-form identification: return a dict with ``brand``,
        ``product_name``, ``variant``, ``confidence``, ``reason`` (or ``None``
        when disabled / API call fails).

        Unlike :meth:`identify`, the returned ``product_name`` is *not*
        constrained to the local catalog. The open-world identifier verifies
        the guess against Open Food Facts."""

        if not self.enabled:
            return None

        # Local backend takes priority when enabled.
        if self._local_backend is not None and self._local_backend.enabled:
            return self._local_backend.identify_open(crop_path, ocr_text, candidates)

        if not self._anthropic_enabled:
            return None

        try:
            client = self._ensure_client()
            with open(crop_path, "rb") as fh:
                image_bytes = fh.read()
            image_b64 = base64.standard_b64encode(image_bytes).decode("ascii")
            media_type = _IMAGE_MEDIA_TYPES.get(crop_path.suffix.lower(), "image/jpeg")

            user_lines: list[str] = []
            if ocr_text:
                user_lines.append(f"OCR text: {ocr_text}")
            if candidates:
                trimmed = [c for c in candidates if c][:5]
                if trimmed:
                    user_lines.append(
                        "Earlier-stage candidate names (hints, may be wrong): "
                        + "; ".join(trimmed)
                    )
            user_lines.append(
                'Identify this packaged product. Reply with strict JSON only: '
                '{"brand": <string or null>, "product_name": <string or null>, '
                '"variant": <string or null>, "confidence": <0.0-1.0>, '
                '"reason": "<short>"}'
            )

            response = client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                system=self._open_system_prompt(),
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": media_type,
                                    "data": image_b64,
                                },
                            },
                            {"type": "text", "text": "\n".join(user_lines)},
                        ],
                    }
                ],
            )
        except Exception as exc:
            return {
                "brand": None,
                "product_name": None,
                "variant": None,
                "confidence": 0.0,
                "reason": f"VLM open call failed: {exc}",
                "source": VLM_OPEN_SOURCE,
                "error": str(exc),
            }

        text = "".join(getattr(block, "text", "") for block in response.content)
        payload = self._parse_json(text)
        if payload is None:
            return {
                "brand": None,
                "product_name": None,
                "variant": None,
                "confidence": 0.0,
                "reason": "VLM open response was not valid JSON.",
                "source": VLM_OPEN_SOURCE,
                "raw": text[:500],
            }

        brand = payload.get("brand")
        product_name = payload.get("product_name")
        variant = payload.get("variant")
        if not isinstance(brand, str) or not brand.strip():
            brand = None
        else:
            brand = brand.strip()
        if not isinstance(product_name, str) or not product_name.strip():
            product_name = None
        else:
            product_name = product_name.strip()
        if not isinstance(variant, str) or not variant.strip():
            variant = None
        else:
            variant = variant.strip()

        try:
            confidence = float(payload.get("confidence") or 0.0)
        except (TypeError, ValueError):
            confidence = 0.0
        confidence = max(0.0, min(1.0, confidence))

        return {
            "brand": brand,
            "product_name": product_name,
            "variant": variant,
            "confidence": confidence,
            "reason": str(payload.get("reason") or "VLM open-mode identification."),
            "source": VLM_OPEN_SOURCE,
            "raw": text[:500],
        }

    @staticmethod
    def _parse_json(text: str) -> dict | None:
        if not text:
            return None
        stripped = text.strip()
        if stripped.startswith("```"):
            stripped = stripped.strip("`")
            if "\n" in stripped:
                stripped = stripped.split("\n", 1)[1]
            if stripped.endswith("```"):
                stripped = stripped[:-3]
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start < 0 or end <= start:
            return None
        try:
            return json.loads(stripped[start : end + 1])
        except json.JSONDecodeError:
            return None
