"""Open-world product identifier.

The closed-set pipeline (OCR pattern match, DINOv2 KNN, SigLIP zero-shot,
catalog-constrained VLM) can only name products already present in
`backend/knowledge_base/`. This module identifies *any* grocery product on
Earth by combining:

1. A free-form vision-language guess (Anthropic Claude, no catalog constraint).
2. Open Food Facts (and Open Beauty / Products Facts) verification + brand /
   image enrichment.
3. An optional barcode reader for the rare shelf shot where a UPC/EAN is
   actually visible.
4. An optional knowledge-base auto-grow step so identified products fall back
   to the cheap DINOv2 / SigLIP path on subsequent runs.

The stage is invoked by `ProductIdentifier.enrich_detections` for every crop
that survives the closed-set fallbacks unlabeled.

Environment knobs
-----------------
- ``OPEN_WORLD_ENABLED``           master switch (default on; auto-disabled
                                   without an Anthropic key)
- ``OPEN_WORLD_USER_AGENT``        HTTP User-Agent for the Open*Facts APIs
- ``OPEN_WORLD_PROVIDERS``         comma list, any of:
                                   ``openfoodfacts``, ``openbeautyfacts``,
                                   ``openproductsfacts`` (default first)
- ``OPEN_WORLD_HTTP_TIMEOUT``      seconds, per HTTP call (default 8)
- ``OPEN_WORLD_MIN_INTERVAL``      seconds, min gap between API calls
                                   (default 0.25)
- ``OPEN_WORLD_MIN_SIDE``          int, skip crops shorter than this side
                                   (default 96)
- ``OPEN_WORLD_ACCEPT_CONFIDENCE`` float, accept threshold for the combined
                                   open-world identification (default 0.55)
- ``OPEN_WORLD_BARCODE_ENABLED``   "1"/"0", run pyzbar barcode reader if the
                                   library + libzbar are available (default 1)
- ``OPEN_WORLD_AUTO_CATALOG``      "1"/"0", auto-grow the knowledge base on
                                   confident open-world hits (default 0)
- ``OPEN_WORLD_MATCH_THRESHOLD``   float, fuzzy-match similarity required
                                   between the VLM guess and the OFF
                                   canonical name to canonicalize
                                   (default 0.55)
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import shutil
import threading
import time
import urllib.parse
from dataclasses import dataclass, field
from datetime import datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

OPEN_WORLD_SOURCE = "open_world_identifier"
OFF_SOURCE = "open_food_facts"
BARCODE_SOURCE = "barcode_lookup"

_DISABLE_VALUES = {"0", "false", "no", "off"}

_OFF_BASE_URLS: dict[str, str] = {
    "openfoodfacts": "https://world.openfoodfacts.org",
    "openbeautyfacts": "https://world.openbeautyfacts.org",
    "openproductsfacts": "https://world.openproductsfacts.org",
}

# Search-a-licious is the official fast text-search endpoint for Open Food
# Facts. The legacy /cgi/search.pl service is known to hang for 20-60s on
# popular queries; search-a-licious responds in <1s.
_SEARCH_A_LICIOUS_BASE = "https://search.openfoodfacts.org"

_DEFAULT_USER_AGENT = (
    "DigitalTwinShelfMapper/0.1 "
    "(+https://github.com/DriverAI-Team/digital-twin-shelf-mapper)"
)

_SAFE_NAME_RE = re.compile(r"[^A-Za-z0-9 ._'&-]+")


def _is_disabled(env_name: str, default: str = "1") -> bool:
    return os.getenv(env_name, default).strip().lower() in _DISABLE_VALUES


def _norm_for_match(text: str | None) -> str:
    if not text:
        return ""
    return re.sub(r"\s+", " ", text.strip().lower())


def _fuzzy_ratio(a: str | None, b: str | None) -> float:
    a_n, b_n = _norm_for_match(a), _norm_for_match(b)
    if not a_n or not b_n:
        return 0.0
    return SequenceMatcher(None, a_n, b_n).ratio()


def _safe_dirname(name: str) -> str:
    """Sanitize a product name to a filesystem-safe directory name."""
    cleaned = _SAFE_NAME_RE.sub(" ", name).strip()
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned[:120] or "Unknown Product"


@dataclass
class OFFRecord:
    """Normalized record from the Open*Facts family of APIs."""

    code: str | None = None
    product_name: str | None = None
    brand: str | None = None
    generic_name: str | None = None
    image_url: str | None = None
    categories: list[str] = field(default_factory=list)
    countries: list[str] = field(default_factory=list)
    provider: str = "openfoodfacts"
    score: float = 0.0  # fuzzy similarity vs the query
    raw: dict = field(default_factory=dict)

    @property
    def display_name(self) -> str | None:
        if self.product_name and self.brand:
            if self.brand.lower() not in self.product_name.lower():
                return f"{self.brand} {self.product_name}".strip()
        return self.product_name

    def to_dict(self) -> dict:
        return {
            "code": self.code,
            "product_name": self.product_name,
            "brand": self.brand,
            "generic_name": self.generic_name,
            "image_url": self.image_url,
            "categories": list(self.categories),
            "countries": list(self.countries),
            "provider": self.provider,
            "score": round(self.score, 4),
            "display_name": self.display_name,
        }


def _coerce_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _split_csv(value: Any) -> list[str]:
    if not value:
        return []
    if isinstance(value, list):
        return [str(v).strip() for v in value if str(v).strip()]
    text = str(value).strip()
    if not text:
        return []
    return [piece.strip() for piece in text.split(",") if piece.strip()]


def _flatten_field(value: Any) -> str | None:
    """Coerce an Open*Facts field value to a string. Search-a-licious often
    returns list values where the legacy ``/cgi/search.pl`` returns CSV
    strings; both must reduce to a clean ``str | None``."""
    if value is None:
        return None
    if isinstance(value, list):
        joined = ", ".join(str(v).strip() for v in value if str(v).strip())
        return joined or None
    text = str(value).strip()
    return text or None


def _flatten_off_product(payload: dict, provider: str) -> OFFRecord:
    name = (
        _flatten_field(payload.get("product_name_en"))
        or _flatten_field(payload.get("product_name"))
        or _flatten_field(payload.get("generic_name_en"))
        or _flatten_field(payload.get("generic_name"))
    )
    brand_field = _flatten_field(payload.get("brands"))
    brand = brand_field.split(",")[0].strip() if brand_field else None
    generic = _flatten_field(payload.get("generic_name")) or _flatten_field(
        payload.get("generic_name_en")
    )
    image_url = (
        _flatten_field(payload.get("image_front_url"))
        or _flatten_field(payload.get("image_url"))
        or _flatten_field(payload.get("image_small_url"))
    )
    categories = _split_csv(payload.get("categories"))
    countries = _split_csv(payload.get("countries"))
    return OFFRecord(
        code=_flatten_field(payload.get("code")),
        product_name=name,
        brand=brand,
        generic_name=generic,
        image_url=image_url,
        categories=categories,
        countries=countries,
        provider=provider,
        raw={
            k: payload.get(k)
            for k in (
                "code",
                "product_name",
                "product_name_en",
                "brands",
                "generic_name",
                "image_front_url",
                "categories_tags",
                "countries_tags",
                "nutriscore_grade",
            )
            if payload.get(k) is not None
        },
    )


# ---------------------------------------------------------------------------
# Barcode reader (soft dep on pyzbar / libzbar)
# ---------------------------------------------------------------------------


class BarcodeReader:
    """Soft wrapper around pyzbar. Disabled gracefully when pyzbar or libzbar
    is not installed."""

    def __init__(self) -> None:
        self.enabled = not _is_disabled("OPEN_WORLD_BARCODE_ENABLED")
        self._decode = None
        if self.enabled:
            try:
                from pyzbar.pyzbar import decode  # type: ignore

                self._decode = decode
            except Exception as exc:  # ImportError or libzbar0 missing
                logger.info("Barcode reader unavailable: %s", exc)
                self.enabled = False

    def read(self, image_path: Path) -> str | None:
        if not self.enabled or self._decode is None:
            return None
        try:
            from PIL import Image as _PILImage  # local to avoid cold import

            with _PILImage.open(image_path) as img:
                rgb = img.convert("RGB")
            symbols = self._decode(rgb)
        except Exception as exc:
            logger.debug("Barcode decode failed for %s: %s", image_path, exc)
            return None
        for sym in symbols:
            data = getattr(sym, "data", None)
            if not data:
                continue
            try:
                text = data.decode("utf-8").strip()
            except Exception:
                continue
            if text.isdigit() and len(text) in (8, 12, 13, 14):
                return text
        return None


# ---------------------------------------------------------------------------
# Open*Facts HTTP client
# ---------------------------------------------------------------------------


class OpenFactsClient:
    """Minimal HTTP client for Open Food / Beauty / Products Facts with
    disk-backed cache and per-process rate limiting."""

    def __init__(
        self,
        cache_dir: Path,
        user_agent: str | None = None,
        providers: tuple[str, ...] = ("openfoodfacts",),
        timeout: float = 8.0,
        min_request_interval: float = 0.25,
    ) -> None:
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.user_agent = user_agent or _DEFAULT_USER_AGENT
        self.timeout = timeout
        self.min_request_interval = max(0.0, min_request_interval)
        self.providers: tuple[str, ...] = tuple(
            p for p in providers if p in _OFF_BASE_URLS
        ) or ("openfoodfacts",)
        self._lock = threading.Lock()
        self._last_request_ts: float = 0.0
        self._requests_module = None
        self._requests_unavailable = False

    # ------------------------------------------------------------------ http

    def _throttle(self) -> None:
        with self._lock:
            elapsed = time.monotonic() - self._last_request_ts
            if elapsed < self.min_request_interval:
                time.sleep(self.min_request_interval - elapsed)
            self._last_request_ts = time.monotonic()

    def _http_get_json(self, url: str) -> dict | None:
        self._throttle()
        headers = {"User-Agent": self.user_agent, "Accept": "application/json"}
        # Prefer requests for connection reuse + redirects; fall back to urllib.
        if not self._requests_unavailable and self._requests_module is None:
            try:
                import requests  # type: ignore

                self._requests_module = requests
            except ImportError:
                self._requests_unavailable = True
        # Use a (connect, read) tuple so a stalled host fails fast on connect
        # but still allows the configured read budget for slow JSON payloads.
        connect_timeout = min(3.0, self.timeout)
        timeout: tuple[float, float] | float = (connect_timeout, self.timeout)
        try:
            if self._requests_module is not None:
                try:
                    resp = self._requests_module.get(
                        url, headers=headers, timeout=timeout
                    )
                except (
                    self._requests_module.exceptions.Timeout,
                    self._requests_module.exceptions.ConnectionError,
                ) as exc:
                    logger.warning("OFF request timed out / failed: %s (%s)", url, exc)
                    return None
                if resp.status_code != 200:
                    logger.debug("OFF %s -> HTTP %s", url, resp.status_code)
                    return None
                try:
                    return resp.json()
                except ValueError as exc:
                    logger.debug("OFF %s returned non-JSON: %s", url, exc)
                    return None
            import urllib.request as _ur

            req = _ur.Request(url, headers=headers)
            with _ur.urlopen(req, timeout=self.timeout) as response:  # noqa: S310
                return json.loads(response.read().decode("utf-8"))
        except Exception as exc:
            logger.warning("OFF request failed for %s: %s", url, exc)
            return None

    # ------------------------------------------------------------------ cache

    def _cache_path(self, kind: str, key: str, provider: str) -> Path:
        digest = hashlib.sha256(f"{kind}|{provider}|{key}".encode("utf-8")).hexdigest()
        return self.cache_dir / f"{digest}.json"

    def _cache_load(self, kind: str, key: str, provider: str) -> dict | None:
        path = self._cache_path(kind, key, provider)
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return None

    def _cache_save(self, kind: str, key: str, provider: str, payload: dict) -> None:
        path = self._cache_path(kind, key, provider)
        try:
            path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        except OSError as exc:
            logger.debug("Could not write OFF cache %s: %s", path, exc)

    # ------------------------------------------------------------------ api

    def lookup_barcode(self, code: str) -> OFFRecord | None:
        code = re.sub(r"\D", "", code or "")
        if not code:
            return None
        for provider in self.providers:
            cached = self._cache_load("barcode", code, provider)
            payload = cached
            if cached is None:
                url = (
                    f"{_OFF_BASE_URLS[provider]}/api/v2/product/"
                    f"{urllib.parse.quote(code)}.json"
                )
                payload = self._http_get_json(url)
                # Only cache real responses. An empty/None payload means a
                # transient network failure; caching it would silently poison
                # every subsequent run with an empty barcode lookup.
                if isinstance(payload, dict) and payload.get("status") is not None:
                    self._cache_save("barcode", code, provider, payload)
            if not payload or not isinstance(payload, dict):
                continue
            if int(payload.get("status") or 0) != 1:
                continue
            product = payload.get("product") or {}
            if not product:
                continue
            record = _flatten_off_product(product, provider)
            record.code = code
            record.score = 1.0  # barcode lookup is authoritative when found
            return record
        return None

    def _build_search_request(
        self, provider: str, query: str, page_size: int
    ) -> tuple[str, str]:
        """Return ``(url, items_key)`` for a free-text search against ``provider``.

        For Open Food Facts we use Search-a-licious (fast, <1s); for the other
        Open*Facts providers we fall back to the legacy ``/cgi/search.pl``
        endpoint, which is slow but unavoidable."""
        if provider == "openfoodfacts":
            params = urllib.parse.urlencode(
                {
                    "q": query,
                    "page_size": page_size,
                    "fields": ",".join(
                        (
                            "code",
                            "product_name",
                            "product_name_en",
                            "brands",
                            "generic_name",
                            "generic_name_en",
                            "image_front_url",
                            "image_url",
                            "image_small_url",
                            "categories",
                            "categories_tags",
                            "countries",
                            "countries_tags",
                            "nutriscore_grade",
                        )
                    ),
                }
            )
            return (
                f"{_SEARCH_A_LICIOUS_BASE}/search?{params}",
                "hits",
            )
        params = urllib.parse.urlencode(
            {
                "search_terms": query,
                "search_simple": 1,
                "action": "process",
                "json": 1,
                "page_size": page_size,
                "fields": ",".join(
                    (
                        "code",
                        "product_name",
                        "product_name_en",
                        "brands",
                        "generic_name",
                        "generic_name_en",
                        "image_front_url",
                        "image_url",
                        "image_small_url",
                        "categories",
                        "categories_tags",
                        "countries",
                        "countries_tags",
                        "nutriscore_grade",
                    )
                ),
            }
        )
        return (f"{_OFF_BASE_URLS[provider]}/cgi/search.pl?{params}", "products")

    def search(self, query: str, *, page_size: int = 6) -> list[OFFRecord]:
        query = (query or "").strip()
        if not query:
            return []
        results: list[OFFRecord] = []
        for provider in self.providers:
            cache_key = f"{query}|{page_size}"
            payload = self._cache_load("search", cache_key, provider)
            url, items_key = self._build_search_request(provider, query, page_size)
            if payload is None:
                payload = self._http_get_json(url)
                # Only cache successful responses (those that returned a hits
                # or products container, even if empty). Network failures
                # return ``None`` and must not be persisted, otherwise the
                # next run hits the empty cache and silently skips search.
                if isinstance(payload, dict) and (
                    items_key in payload or "hits" in payload or "products" in payload
                ):
                    self._cache_save("search", cache_key, provider, payload)
            items: list | None = None
            if isinstance(payload, dict):
                # Search-a-licious returns hits; cgi/search.pl returns products.
                items = payload.get(items_key) or payload.get("hits") or payload.get("products")
            if not items:
                continue
            for product in items[:page_size]:
                if not isinstance(product, dict):
                    continue
                record = _flatten_off_product(product, provider)
                if record.product_name:
                    record.score = _fuzzy_ratio(query, record.display_name)
                    results.append(record)
        results.sort(key=lambda r: r.score, reverse=True)
        return results


# ---------------------------------------------------------------------------
# Free-form VLM identification call (delegates to vlm_identifier)
# ---------------------------------------------------------------------------


@dataclass
class OpenWorldVLMResult:
    product_name: str | None
    brand: str | None
    variant: str | None
    confidence: float
    reason: str
    raw_text: str | None = None

    @property
    def display_name(self) -> str | None:
        if self.product_name and self.brand:
            if self.brand.lower() not in self.product_name.lower():
                return f"{self.brand} {self.product_name}".strip()
        return self.product_name


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------


class OpenWorldIdentifier:
    """Identify any grocery product by chaining barcode lookup, free-form VLM,
    and Open*Facts verification."""

    def __init__(
        self,
        catalog,  # type: ProductCatalog
        vlm,  # type: VLMIdentifier
        knowledge_base_dir: Path | None = None,
    ) -> None:
        from product_catalog import DEFAULT_KB_DIR  # local to avoid cycle

        self._catalog = catalog
        self._vlm = vlm
        self.kb_dir = knowledge_base_dir or DEFAULT_KB_DIR
        self.enabled = (
            not _is_disabled("OPEN_WORLD_ENABLED")
            and getattr(vlm, "enabled", False)
        )
        self.min_side = int(os.getenv("OPEN_WORLD_MIN_SIDE", "96"))
        self.accept_confidence = float(
            os.getenv("OPEN_WORLD_ACCEPT_CONFIDENCE", "0.55")
        )
        self.match_threshold = float(
            os.getenv("OPEN_WORLD_MATCH_THRESHOLD", "0.55")
        )
        self.auto_catalog = not _is_disabled("OPEN_WORLD_AUTO_CATALOG", default="0")
        providers_env = os.getenv(
            "OPEN_WORLD_PROVIDERS", "openfoodfacts"
        ).strip().lower()
        providers = tuple(
            p.strip() for p in providers_env.split(",") if p.strip()
        ) or ("openfoodfacts",)
        cache_dir = self.kb_dir / "open_world_cache"
        self.client = OpenFactsClient(
            cache_dir=cache_dir,
            user_agent=os.getenv("OPEN_WORLD_USER_AGENT") or _DEFAULT_USER_AGENT,
            providers=providers,
            timeout=float(os.getenv("OPEN_WORLD_HTTP_TIMEOUT", "8")),
            min_request_interval=float(os.getenv("OPEN_WORLD_MIN_INTERVAL", "0.25")),
        )
        self.barcode_reader = BarcodeReader()
        self._catalog_lock = threading.Lock()

    # ------------------------------------------------------------------ utils

    def _crop_meets_min_side(self, crop_path: Path) -> bool:
        try:
            from PIL import Image as _PILImage

            with _PILImage.open(crop_path) as img:
                return min(img.size) >= self.min_side
        except Exception:
            return True  # be permissive on read failure

    @staticmethod
    def _hash_crop(crop_path: Path) -> str:
        with crop_path.open("rb") as fh:
            return hashlib.sha256(fh.read()).hexdigest()

    # ------------------------------------------------------------------ vlm

    def _call_open_vlm(
        self,
        crop_path: Path,
        ocr_text: str | None,
        catalog_candidates: list[str],
    ) -> OpenWorldVLMResult | None:
        if not getattr(self._vlm, "enabled", False):
            return None
        payload = self._vlm.identify_open(
            crop_path, ocr_text=ocr_text, candidates=catalog_candidates
        )
        if not payload:
            return None
        product_name = _coerce_str(payload.get("product_name"))
        brand = _coerce_str(payload.get("brand"))
        variant = _coerce_str(payload.get("variant"))
        try:
            confidence = float(payload.get("confidence") or 0.0)
        except (TypeError, ValueError):
            confidence = 0.0
        confidence = max(0.0, min(1.0, confidence))
        reason = _coerce_str(payload.get("reason")) or "VLM open-mode identification."
        raw_text = _coerce_str(payload.get("raw"))
        if not product_name and not brand:
            return None
        return OpenWorldVLMResult(
            product_name=product_name,
            brand=brand,
            variant=variant,
            confidence=confidence,
            reason=reason,
            raw_text=raw_text,
        )

    # ------------------------------------------------------------------ main

    def identify(
        self,
        crop_path: Path,
        raw_crop_path: Path | None = None,
        ocr_text: str | None = None,
        catalog_candidates: list[str] | None = None,
    ) -> dict | None:
        """Return a result dict compatible with `ProductIdentifier._apply_ocr_result`
        or ``None`` when the stage cannot run."""

        if not self.enabled:
            return None
        if not crop_path.exists():
            return None
        if not self._crop_meets_min_side(crop_path):
            return None

        catalog_candidates = catalog_candidates or []
        result: dict[str, Any] = {
            "source": OPEN_WORLD_SOURCE,
            "product_name": None,
            "exact_match": False,
            "confidence": 0.0,
            "reason": "open-world identification not attempted",
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }

        # 1. Barcode lookup — authoritative when present.
        barcode_paths: list[Path] = []
        if raw_crop_path is not None and raw_crop_path.exists():
            barcode_paths.append(raw_crop_path)
        barcode_paths.append(crop_path)
        barcode_value: str | None = None
        for candidate in barcode_paths:
            barcode_value = self.barcode_reader.read(candidate)
            if barcode_value:
                break
        off_record: OFFRecord | None = None
        if barcode_value:
            off_record = self.client.lookup_barcode(barcode_value)
            if off_record and off_record.display_name:
                self._populate_result(
                    result,
                    off_record=off_record,
                    confidence=0.97,
                    reason=f"barcode {barcode_value} → {off_record.provider} canonical name",
                    barcode=barcode_value,
                    match_score=1.0,
                )
                if self.auto_catalog:
                    self._auto_grow_catalog(off_record, crop_path)
                return result

        # 2. Free-form VLM identification.
        vlm_result = self._call_open_vlm(
            crop_path=crop_path,
            ocr_text=ocr_text,
            catalog_candidates=catalog_candidates,
        )
        if not vlm_result or not (vlm_result.product_name or vlm_result.brand):
            result["reason"] = "open-mode VLM returned no product guess"
            return result

        vlm_display = vlm_result.display_name or vlm_result.product_name
        if not vlm_display:
            result["reason"] = "open-mode VLM did not return a usable name"
            return result

        # 3. Open*Facts verification.
        query = vlm_display
        if vlm_result.variant and vlm_result.variant.lower() not in query.lower():
            query = f"{query} {vlm_result.variant}"
        off_candidates = self.client.search(query)
        best_off: OFFRecord | None = off_candidates[0] if off_candidates else None
        match_score = best_off.score if best_off else 0.0

        # 4. Choose canonical name + confidence.
        canonical_name: str
        canonical_brand: str | None = None
        verification_reason: str
        if best_off and match_score >= self.match_threshold and best_off.display_name:
            canonical_name = best_off.display_name
            canonical_brand = best_off.brand or vlm_result.brand
            verification_reason = (
                f"VLM '{vlm_display}' → {best_off.provider} '{canonical_name}' "
                f"(match {match_score:.2f})"
            )
            confidence = round(
                min(1.0, 0.6 * vlm_result.confidence + 0.4 * match_score + 0.05),
                3,
            )
        else:
            canonical_name = vlm_display
            canonical_brand = vlm_result.brand
            verification_reason = (
                f"VLM open-mode '{vlm_display}' (no Open*Facts match)"
                if not best_off
                else f"VLM open-mode '{vlm_display}'; best Open*Facts candidate "
                     f"'{best_off.display_name}' below match threshold "
                     f"({match_score:.2f} < {self.match_threshold:.2f})"
            )
            confidence = round(min(1.0, 0.75 * vlm_result.confidence), 3)

        self._populate_result(
            result,
            off_record=best_off,
            canonical_name=canonical_name,
            canonical_brand=canonical_brand,
            confidence=confidence,
            reason=verification_reason,
            barcode=None,
            match_score=match_score,
            vlm_result=vlm_result,
            other_off_candidates=off_candidates[1:4],
        )

        # 5. Optional knowledge-base growth.
        if (
            self.auto_catalog
            and result["exact_match"]
            and best_off is not None
            and match_score >= max(self.match_threshold, 0.7)
        ):
            self._auto_grow_catalog(best_off, crop_path, fallback_name=canonical_name)

        return result

    # ------------------------------------------------------------------ helpers

    def _populate_result(
        self,
        result: dict[str, Any],
        *,
        off_record: OFFRecord | None,
        canonical_name: str | None = None,
        canonical_brand: str | None = None,
        confidence: float,
        reason: str,
        barcode: str | None,
        match_score: float,
        vlm_result: OpenWorldVLMResult | None = None,
        other_off_candidates: list[OFFRecord] | None = None,
    ) -> None:
        if off_record is not None and not canonical_name:
            canonical_name = off_record.display_name
        if off_record is not None and not canonical_brand:
            canonical_brand = off_record.brand
        result["product_name"] = canonical_name
        result["display_name"] = canonical_name
        result["brand"] = canonical_brand
        result["confidence"] = float(round(max(0.0, min(1.0, confidence)), 4))
        result["exact_match"] = bool(canonical_name) and result["confidence"] >= self.accept_confidence
        result["reason"] = reason
        result["barcode"] = barcode
        result["match_score"] = round(float(match_score), 4)
        if off_record is not None:
            result["off_record"] = off_record.to_dict()
            result["off_code"] = off_record.code
            result["off_image_url"] = off_record.image_url
            result["off_categories"] = off_record.categories
            result["off_provider"] = off_record.provider
        if vlm_result is not None:
            result["vlm"] = {
                "product_name": vlm_result.product_name,
                "brand": vlm_result.brand,
                "variant": vlm_result.variant,
                "confidence": round(vlm_result.confidence, 4),
                "reason": vlm_result.reason,
                "raw": vlm_result.raw_text,
            }
        if other_off_candidates:
            result["off_alternatives"] = [r.to_dict() for r in other_off_candidates]

    # ------------------------------------------------------------------ auto-grow

    def _auto_grow_catalog(
        self,
        off_record: OFFRecord,
        crop_path: Path,
        fallback_name: str | None = None,
    ) -> None:
        canonical_name = off_record.display_name or fallback_name
        if not canonical_name:
            return
        try:
            with self._catalog_lock:
                product_dir = self.kb_dir / _safe_dirname(canonical_name) / "auto"
                product_dir.mkdir(parents=True, exist_ok=True)
                dest = product_dir / f"{self._hash_crop(crop_path)[:16]}{crop_path.suffix.lower()}"
                if not dest.exists():
                    shutil.copy2(crop_path, dest)
                self._update_catalog_manifest(canonical_name, off_record)
        except Exception as exc:
            logger.warning(
                "Open-world auto-grow failed for '%s': %s", canonical_name, exc
            )

    def _update_catalog_manifest(self, canonical_name: str, off_record: OFFRecord) -> None:
        manifest_path = self.kb_dir / "catalog.json"
        manifest: dict[str, Any] = {"products": {}}
        if manifest_path.exists():
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                if not isinstance(manifest, dict):
                    manifest = {"products": {}}
                manifest.setdefault("products", {})
            except json.JSONDecodeError:
                manifest = {"products": {}}
        products = manifest.setdefault("products", {})
        entry: dict[str, Any] = products.get(canonical_name) or {}
        aliases = set(entry.get("aliases") or [])
        if off_record.product_name:
            aliases.add(off_record.product_name)
        if off_record.brand:
            aliases.add(off_record.brand)
        if off_record.generic_name:
            aliases.add(off_record.generic_name)
        entry["aliases"] = sorted(a for a in aliases if a and a != canonical_name)
        if off_record.code:
            entry["off_code"] = off_record.code
        if off_record.provider:
            entry["source"] = off_record.provider
        if off_record.image_url:
            entry["reference_image_url"] = off_record.image_url
        entry["auto_added_at"] = datetime.now(timezone.utc).isoformat()
        products[canonical_name] = entry
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
