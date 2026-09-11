"""Local open-source vision-language model backend.

Drop-in replacement for the Anthropic VLM stage. Wraps Qwen2.5-VL via
HuggingFace transformers, with first-class support for Apple Silicon (MPS).

Activated by setting ``LOCAL_VLM_ENABLED=1`` in the environment. When active,
both the catalog VLM stage (closed-set, against the local knowledge base) and
the open-world VLM stage (free-form product identification) use this backend
instead of calling Anthropic.

Tested on Apple Silicon with the 3B variant, which runs in ~6 GB of unified
memory. Bump to ``Qwen/Qwen2.5-VL-7B-Instruct`` if you have 32 GB+ RAM.
"""

from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from threading import Lock

logger = logging.getLogger(__name__)

LOCAL_VLM_SOURCE = "local_vlm_qwen2_5"
LOCAL_VLM_OPEN_SOURCE = "local_vlm_qwen2_5_open"

_DISABLE_VALUES = {"0", "false", "no", "off"}
_DEFAULT_MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"

# Qwen2.5-VL counts visual tokens in 28x28 patches; the processor ships with
# generous defaults (max_pixels = 1280*28*28 ≈ 1M tokens) that overrun MPS on
# 16 GB Macs for full-resolution shelf shots. Capping at 512*28*28 (≈ 401k
# tokens, roughly 640×640) keeps memory bounded while preserving enough
# resolution to read package text. Tweak via LOCAL_VLM_MAX_PIXELS.
_DEFAULT_MIN_PIXELS = 256 * 28 * 28
_DEFAULT_MAX_PIXELS = 512 * 28 * 28


def _is_disabled(env_name: str, default: str = "1") -> bool:
    return os.getenv(env_name, default).strip().lower() in _DISABLE_VALUES


def _resolve_dtype(device: str):
    """Return the torch dtype that matches the chosen device."""
    import torch

    if device == "cuda":
        # Use bf16 on Ampere+ (A100, H100, RTX 30/40 series); fp16 elsewhere.
        if torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8:
            return torch.bfloat16
        return torch.float16
    if device == "mps":
        # MPS does not support bf16 reliably as of torch 2.4. fp16 is faster
        # than fp32 on M1/M2/M3 GPUs and fits the 3B model in ~6 GB.
        return torch.float16
    return torch.float32


def _select_device() -> str:
    """Apple Silicon → MPS, NVIDIA → CUDA, otherwise CPU."""
    override = os.getenv("LOCAL_VLM_DEVICE", "").strip().lower()
    if override:
        return override
    try:
        import torch
    except ImportError:
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class LocalVLM:
    """Thin Qwen2.5-VL wrapper with the same surface area as
    :class:`backend.vlm_identifier.VLMIdentifier` for the two methods used by
    the identification pipeline (``identify`` and ``identify_open``).

    The model is lazy-loaded on first call so importing this module is cheap
    and the closed-set OCR/embedding stages do not pay for VLM weights they
    never use."""

    def __init__(self, catalog) -> None:
        self.enabled = (
            not _is_disabled("LOCAL_VLM_ENABLED", default="0")
            and os.getenv("LOCAL_VLM_ENABLED", "0").strip().lower() not in _DISABLE_VALUES
        )
        # Re-evaluate so the default ("0") explicitly disables.
        self.enabled = os.getenv("LOCAL_VLM_ENABLED", "0").strip().lower() not in _DISABLE_VALUES
        self.model_id = os.getenv("LOCAL_VLM_MODEL", _DEFAULT_MODEL_ID).strip() or _DEFAULT_MODEL_ID
        self.max_new_tokens = int(os.getenv("LOCAL_VLM_MAX_TOKENS", "320"))
        self.accept_threshold = float(os.getenv("VLM_FALLBACK_ACCEPT_THRESHOLD", "0.80"))
        self.min_pixels = int(os.getenv("LOCAL_VLM_MIN_PIXELS", str(_DEFAULT_MIN_PIXELS)))
        self.max_pixels = int(os.getenv("LOCAL_VLM_MAX_PIXELS", str(_DEFAULT_MAX_PIXELS)))
        # Defensive PIL resize before the processor sees the image. Set to 0
        # to disable. Default of 768 px on the long side keeps even 4K shelf
        # photos well under the 512*28*28 budget.
        self.long_side_px = int(os.getenv("LOCAL_VLM_LONG_SIDE_PX", "768"))
        self._device = _select_device()
        self._dtype = None  # resolved on first load
        self._catalog = catalog
        self._catalog_names: tuple[str, ...] = tuple(catalog.product_names())
        self._model = None
        self._processor = None
        self._load_attempted = False
        self._load_error: str | None = None
        self._lock = Lock()

    # ------------------------------------------------------------------ load

    def _ensure_loaded(self) -> bool:
        with self._lock:
            if self._load_attempted:
                return self._model is not None
            self._load_attempted = True
            if not self.enabled:
                return False
            try:
                import torch
                from transformers import AutoProcessor

                # Qwen2.5-VL has a dedicated class in transformers >= 4.49.
                try:
                    from transformers import Qwen2_5_VLForConditionalGeneration as _VLClass
                except ImportError:
                    # Fallback for earlier transformers releases that only
                    # shipped Qwen2-VL (the 2.0 architecture).
                    from transformers import Qwen2VLForConditionalGeneration as _VLClass
                    logger.warning(
                        "transformers does not expose Qwen2_5_VLForConditionalGeneration; "
                        "falling back to Qwen2VLForConditionalGeneration. Upgrade to "
                        "transformers>=4.49 for Qwen2.5-VL support."
                    )

                self._dtype = _resolve_dtype(self._device)
                logger.info(
                    "Loading local VLM %s on %s (%s)",
                    self.model_id,
                    self._device,
                    self._dtype,
                )

                # Allow MPS to use the full unified memory pool. Without this,
                # the default high-water-mark (~75% of RAM) trips OOMs even on
                # 16 GB Macs once the vision tower allocates activations.
                if self._device == "mps":
                    os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.0")

                processor_kwargs = {"trust_remote_code": True}
                if self.min_pixels > 0:
                    processor_kwargs["min_pixels"] = self.min_pixels
                if self.max_pixels > 0:
                    processor_kwargs["max_pixels"] = self.max_pixels
                self._processor = AutoProcessor.from_pretrained(
                    self.model_id,
                    **processor_kwargs,
                )
                load_kwargs = {
                    "torch_dtype": self._dtype,
                    "trust_remote_code": True,
                    "low_cpu_mem_usage": True,
                }
                # FlashAttention-2 only works on CUDA. Eager attention works
                # everywhere and is the only safe option on MPS.
                if self._device != "cuda":
                    load_kwargs["attn_implementation"] = "eager"
                self._model = _VLClass.from_pretrained(self.model_id, **load_kwargs)
                if self._device != "cpu":
                    self._model = self._model.to(self._device)
                self._model.eval()
                return True
            except Exception as exc:  # noqa: BLE001
                self._load_error = f"{type(exc).__name__}: {exc}"
                logger.warning(
                    "Local VLM disabled: failed to load %s — %s",
                    self.model_id,
                    self._load_error,
                )
                self._model = None
                self._processor = None
                return False

    # ------------------------------------------------------------------ generate

    def _generate(self, prompt: str, image_path: Path) -> str:
        """Run a single image+text Qwen2.5-VL forward pass and return the
        decoded model response (after the last assistant turn)."""
        from PIL import Image
        import torch

        image = Image.open(image_path).convert("RGB")
        # Belt-and-suspenders: pre-shrink very large images so the processor
        # never has to allocate a full-resolution tensor. The vision tower
        # already downsamples to a multiple of 28, so a 768-px long side is
        # plenty for OCR while keeping MPS activations well below the OOM
        # watermark.
        if self.long_side_px > 0:
            longest = max(image.size)
            if longest > self.long_side_px:
                scale = self.long_side_px / float(longest)
                new_size = (
                    max(1, int(round(image.size[0] * scale))),
                    max(1, int(round(image.size[1] * scale))),
                )
                image = image.resize(new_size, Image.LANCZOS)
        # Qwen2.5-VL chat template expects messages of the form
        # [{"role": "user", "content": [{"type":"image"}, {"type":"text","text":...}]}]
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        text = self._processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self._processor(
            text=[text],
            images=[image],
            padding=True,
            return_tensors="pt",
        )
        if self._device != "cpu":
            inputs = {k: v.to(self._device) for k, v in inputs.items()}

        with torch.inference_mode():
            output_ids = self._model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                temperature=1.0,
            )
        # Trim the prompt tokens so we only decode the model's reply.
        prompt_len = inputs["input_ids"].shape[1]
        trimmed = output_ids[:, prompt_len:]
        decoded = self._processor.batch_decode(
            trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        # Release intermediate tensors so the next forward pass starts with a
        # clean memory slate. MPS in particular hangs on to activations
        # aggressively without an explicit cache flush.
        del inputs, output_ids, trimmed
        if self._device == "mps" and hasattr(torch, "mps"):
            try:
                torch.mps.empty_cache()
            except Exception:  # noqa: BLE001
                pass
        elif self._device == "cuda":
            try:
                torch.cuda.empty_cache()
            except Exception:  # noqa: BLE001
                pass
        return decoded[0] if decoded else ""

    # ------------------------------------------------------------------ identify

    def _catalog_prompt(self, ocr_text: str | None, candidates: list[str]) -> str:
        catalog_lines = "\n".join(f"- {name}" for name in self._catalog_names)
        hint_lines: list[str] = []
        if ocr_text:
            hint_lines.append(f"OCR text: {ocr_text}")
        if candidates:
            trimmed = [c for c in candidates if c][:5]
            if trimmed:
                hint_lines.append("Earlier-stage candidates: " + "; ".join(trimmed))
        hint_block = "\n".join(hint_lines) + ("\n" if hint_lines else "")
        return (
            "You identify packaged retail products from cropped shelf photos.\n\n"
            "CATALOG (return product_name verbatim from this list, or null):\n"
            f"{catalog_lines}\n\n"
            f"{hint_block}"
            "Rules:\n"
            "1. Return product_name exactly as listed above, or null if no "
            "catalog product is a confident visual match.\n"
            "2. Prefer null over guessing. Confidence below 0.6 means null.\n"
            "3. Use packaging color, brand mark, and flavor text to "
            "disambiguate similar SKUs. The OCR text may be wrong — trust "
            "the image first.\n"
            "4. Reply with strict JSON only: "
            '{"product_name": <string or null>, "confidence": <0.0-1.0>, '
            '"reason": <short string>}'
        )

    def identify(
        self,
        crop_path: Path,
        ocr_text: str | None,
        candidates: list[str],
    ) -> dict | None:
        if not self._ensure_loaded():
            return None
        try:
            text = self._generate(self._catalog_prompt(ocr_text, candidates), crop_path)
        except Exception as exc:
            return {
                "product_name": None,
                "exact_match": False,
                "confidence": 0.0,
                "reason": f"Local VLM call failed: {exc}",
                "source": LOCAL_VLM_SOURCE,
                "error": str(exc),
            }
        payload = _parse_json(text)
        if payload is None:
            return {
                "product_name": None,
                "exact_match": False,
                "confidence": 0.0,
                "reason": "Local VLM response was not valid JSON.",
                "source": LOCAL_VLM_SOURCE,
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
            "reason": str(payload.get("reason") or "Local VLM identification."),
            "source": LOCAL_VLM_SOURCE,
            "raw_product_name": payload.get("product_name"),
        }

    # --------------------------------------------------------------- open

    def _open_prompt(self, ocr_text: str | None, candidates: list[str]) -> str:
        hint_lines: list[str] = []
        if ocr_text:
            hint_lines.append(f"OCR text: {ocr_text}")
        if candidates:
            trimmed = [c for c in candidates if c][:5]
            if trimmed:
                hint_lines.append(
                    "Earlier-stage candidate names (hints, may be wrong): "
                    + "; ".join(trimmed)
                )
        hint_block = "\n".join(hint_lines) + ("\n" if hint_lines else "")
        return (
            "You identify packaged retail products from cropped shelf photos.\n"
            "You may name ANY commercially available grocery / packaged "
            "consumer product on Earth — there is no fixed catalog.\n\n"
            f"{hint_block}"
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
            "6. Reply with exactly one strict JSON object only. Do not return "
            "an array, markdown, commentary, or multiple candidates: "
            '{"brand": <string or null>, "product_name": <string or null>, '
            '"variant": <string or null>, "confidence": <0.0-1.0>, '
            '"reason": <short string>}\n'
            "   - product_name should be a complete consumer-facing name "
            "without the brand prefix (e.g., \"Classic Potato Chips\" not "
            "\"Lay's Classic Potato Chips\").\n"
            "   - variant captures size / flavor modifier when visible "
            "(e.g., \"Family Size\", \"Spicy\")."
        )

    def identify_open(
        self,
        crop_path: Path,
        ocr_text: str | None = None,
        candidates: list[str] | None = None,
    ) -> dict | None:
        if not self._ensure_loaded():
            return None
        try:
            text = self._generate(
                self._open_prompt(ocr_text, candidates or []),
                crop_path,
            )
        except Exception as exc:
            return {
                "brand": None,
                "product_name": None,
                "variant": None,
                "confidence": 0.0,
                "reason": f"Local VLM open call failed: {exc}",
                "source": LOCAL_VLM_OPEN_SOURCE,
                "error": str(exc),
            }
        payload = _parse_json(text)
        if payload is None:
            return {
                "brand": None,
                "product_name": None,
                "variant": None,
                "confidence": 0.0,
                "reason": "Local VLM open response was not valid JSON.",
                "source": LOCAL_VLM_OPEN_SOURCE,
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
            "reason": str(payload.get("reason") or "Local VLM open-mode identification."),
            "source": LOCAL_VLM_OPEN_SOURCE,
            "raw": text[:500],
        }

    # --------------------------------------------------------------- diag

    def status(self) -> dict:
        return {
            "enabled": self.enabled,
            "model_id": self.model_id,
            "device": self._device,
            "loaded": self._model is not None,
            "load_error": self._load_error,
        }


# --------------------------------------------------------------- helpers


_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE | re.MULTILINE)


def _parse_json(text: str) -> dict | None:
    if not text:
        return None
    stripped = _FENCE_RE.sub("", text.strip()).strip()
    try:
        payload = json.loads(stripped)
        if isinstance(payload, dict):
            return payload
        if isinstance(payload, list):
            return next((item for item in payload if isinstance(item, dict)), None)
    except json.JSONDecodeError:
        pass

    start = stripped.find("{")
    if start < 0:
        return None
    depth = 0
    in_string = False
    escape = False
    for index in range(start, len(stripped)):
        char = stripped[index]
        if escape:
            escape = False
            continue
        if char == "\\":
            escape = True
            continue
        if char == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                try:
                    payload = json.loads(stripped[start : index + 1])
                except json.JSONDecodeError:
                    return None
                return payload if isinstance(payload, dict) else None
    return None
