"""Quick recognition test: load all cached detections for a job, run the
ProductIdentifier (OCR + DINOv2 + CLIP + VLM), report named/unnamed counts and
which layer named each detection."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path

WORKSPACE_DIR = Path(__file__).resolve().parent / "workspace"


def _identity_confidence(detection: dict) -> float:
    value = detection.get("product_identity_confidence")
    if value is None:
        value = detection.get("confidence")
    try:
        return max(0.0, min(1.0, float(value or 0.0)))
    except (TypeError, ValueError):
        return 0.0


def load_detections(job_dir: Path) -> list[dict]:
    detections_dir = job_dir / "detections"
    if not detections_dir.exists():
        raise SystemExit(f"No detections cache at {detections_dir}")
    detections: list[dict] = []
    for path in sorted(detections_dir.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        items = payload if isinstance(payload, list) else payload.get("detections") or []
        detections.extend(items)
    return detections


def _live_off_smoke_test() -> int:
    """Hit Open Food Facts with a known barcode and a free-form search query
    to verify the open-world identifier's HTTP path works end-to-end.

    Returns 0 on success, non-zero when neither call returned a usable record."""

    import sys
    import time as _time

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from open_world_identifier import OpenFactsClient

    cache_dir = Path(__file__).resolve().parent / "knowledge_base" / "open_world_cache"
    client = OpenFactsClient(cache_dir=cache_dir)

    print("=" * 60)
    print("OPEN FOOD FACTS LIVE SMOKE TEST")
    print("=" * 60)
    print(f"Cache dir: {cache_dir}")
    print(f"Providers: {client.providers}")
    print(f"User-Agent: {client.user_agent}")
    print()

    failures: list[str] = []

    # Coca-Cola 1.5L PET (well-known reference barcode).
    barcode = "5449000000996"
    t0 = _time.time()
    print(f"[barcode] looking up {barcode} ...")
    record = client.lookup_barcode(barcode)
    elapsed = _time.time() - t0
    if record and record.product_name:
        print(
            f"  ok  ({elapsed:.2f}s)  {record.provider}  "
            f"{record.brand or '?'} :: {record.product_name}"
        )
        print(f"      image: {record.image_url}")
    else:
        print(f"  fail ({elapsed:.2f}s)  no record returned")
        failures.append("barcode lookup")
    print()

    query = "Lays Classic Potato Chips"
    t0 = _time.time()
    print(f"[search]  query={query!r} ...")
    candidates = client.search(query, page_size=5)
    elapsed = _time.time() - t0
    if candidates:
        print(f"  ok  ({elapsed:.2f}s)  {len(candidates)} candidates:")
        for record in candidates[:5]:
            print(
                f"    - score={record.score:.2f}  {record.brand or '?':<20s}  "
                f"{record.product_name}"
            )
    else:
        print(f"  fail ({elapsed:.2f}s)  no candidates returned")
        failures.append("search")
    print()

    if failures:
        print(f"FAILED: {', '.join(failures)}")
        return 1
    print("OK — Open Food Facts connectivity verified.")
    return 0


def _local_vlm_self_check(image_path: Path | None = None) -> int:
    """Load the local VLM and decode one image so we know the model + device
    pipeline works end-to-end before kicking off a full job."""
    import sys as _sys
    import time as _time

    _sys.path.insert(0, str(Path(__file__).resolve().parent))
    os.environ.setdefault("LOCAL_VLM_ENABLED", "1")
    from product_catalog import ProductCatalog
    from local_vlm import LocalVLM

    catalog = ProductCatalog()
    vlm = LocalVLM(catalog)
    print("=" * 60)
    print("LOCAL VLM SELF-CHECK")
    print("=" * 60)
    status = vlm.status()
    for key, value in status.items():
        print(f"  {key}: {value}")
    print()

    print(f"Loading {vlm.model_id} on {vlm._device} ... (first run downloads weights)")
    t0 = _time.time()
    ok = vlm._ensure_loaded()
    print(f"  load: {'OK' if ok else 'FAIL'} in {_time.time() - t0:.1f}s")
    if not ok:
        print(f"  load_error: {vlm._load_error}")
        return 1

    if image_path is None:
        # Pick any catalog image as a smoke test target.
        for candidate in (Path(__file__).resolve().parent / "knowledge_base").rglob("*.jpg"):
            image_path = candidate
            break
    if image_path is None or not image_path.exists():
        print("  no test image found; skipping decode.")
        return 0

    print(f"\nDecoding {image_path} ...")
    t0 = _time.time()
    result = vlm.identify_open(image_path)
    elapsed = _time.time() - t0
    if not result:
        print(f"  fail ({elapsed:.1f}s) — local VLM returned None.")
        return 1
    print(f"  ok ({elapsed:.1f}s):")
    for key in ("brand", "product_name", "variant", "confidence", "reason"):
        print(f"    {key}: {result.get(key)!r}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "job_id",
        nargs="?",
        default=None,
        help="Job id under backend/workspace/. Optional when --live-off / --local-vlm-check is used.",
    )
    parser.add_argument("--no-vlm", action="store_true", help="Disable Anthropic VLM fallback for this run.")
    parser.add_argument("--reset-vlm-cache", action="store_true", help="Delete the per-detection VLM cache before running.")
    parser.add_argument(
        "--live-off",
        action="store_true",
        help="Run an Open Food Facts connectivity smoke test (one barcode lookup + one search) and exit.",
    )
    parser.add_argument(
        "--local-vlm-check",
        action="store_true",
        help="Load the local Qwen2.5-VL model, decode one image, and exit.",
    )
    parser.add_argument(
        "--local-vlm-image",
        default=None,
        help="Path to an image for --local-vlm-check (default: a catalog image).",
    )
    args = parser.parse_args()

    if args.live_off:
        return _live_off_smoke_test()

    if args.local_vlm_check:
        image = Path(args.local_vlm_image) if args.local_vlm_image else None
        return _local_vlm_self_check(image)

    if not args.job_id:
        parser.error(
            "job_id is required unless --live-off or --local-vlm-check is supplied."
        )

    if args.no_vlm:
        os.environ["VLM_FALLBACK_ENABLED"] = "0"
    os.environ.setdefault("PRODUCT_NAME_EMBEDDING_FALLBACK", "1")

    job_dir = WORKSPACE_DIR / args.job_id
    if not job_dir.exists():
        raise SystemExit(f"Job '{args.job_id}' not found at {job_dir}")

    if args.reset_vlm_cache:
        cache_dir = job_dir / "vlm_names"
        if cache_dir.exists():
            for p in cache_dir.iterdir():
                p.unlink(missing_ok=True)
            print(f"Cleared VLM cache: {cache_dir}")

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import time as _time
    t0 = _time.time()
    print(f"[{_time.time()-t0:6.1f}s] importing ProductIdentifier...", flush=True)
    from product_identifier import ProductIdentifier
    print(f"[{_time.time()-t0:6.1f}s] imported.", flush=True)

    detections = load_detections(job_dir)
    print(f"[{_time.time()-t0:6.1f}s] Loaded {len(detections)} cached detections for job '{args.job_id}'.", flush=True)

    before_named = sum(1 for d in detections if d.get("product_identity_label"))
    before_labels = Counter(
        (d.get("display_label") or d.get("label") or "?")
        for d in detections
        if d.get("product_identity_label")
    )

    print(f"[{_time.time()-t0:6.1f}s] instantiating ProductIdentifier...", flush=True)
    identifier = ProductIdentifier()
    print(f"[{_time.time()-t0:6.1f}s] OCR engine: {identifier.engine}", flush=True)
    print(f"[{_time.time()-t0:6.1f}s] CLIP enabled: {identifier._clip_classifier.enabled} ({identifier._clip_classifier.model_name})", flush=True)
    local_backend = getattr(identifier._vlm, "_local_backend", None)
    if local_backend is not None and local_backend.enabled:
        vlm_name = f"local {local_backend.model_id}"
    else:
        vlm_name = identifier._vlm.model
    print(f"[{_time.time()-t0:6.1f}s] VLM enabled: {identifier._vlm.enabled} ({vlm_name})", flush=True)

    print(f"[{_time.time()-t0:6.1f}s] Pipeline stages:", flush=True)
    for stage in identifier.pipeline_status():
        state = "on " if stage["enabled"] else "OFF"
        detail = stage.get("detail") or stage.get("reason") or ""
        print(f"    [{state}] {stage['stage']:<11s} {detail}", flush=True)

    print(f"[{_time.time()-t0:6.1f}s] running enrich_detections...", flush=True)
    enriched, summary = identifier.enrich_detections(job_dir, detections)
    print(f"[{_time.time()-t0:6.1f}s] enrich_detections done.", flush=True)

    from product_classes import build_class_records

    class_records = build_class_records(enriched, identifier.class_registry)

    final_results_path = job_dir / "recognition_results.json"
    final_results = []
    for detection in enriched:
        label = detection.get("display_label") or detection.get("label")
        final_results.append(
            {
                "id": detection.get("id"),
                "image_name": detection.get("image_name"),
                "label": label,
                "recognized_product_name": detection.get("recognized_product_name") or label,
                "class_id": detection.get("class_id"),
                "class_name": detection.get("class_name"),
                "identity_confidence": _identity_confidence(detection),
                "detector_confidence": detection.get("confidence"),
                "source": detection.get("product_identity_source"),
                "label_source": detection.get("label_source"),
                "identity_stage": detection.get("identity_stage"),
                "exact_match": bool(detection.get("product_identity_label")),
                "reason": detection.get("product_identity_reason"),
                "crop_path": detection.get("crop_path"),
                "raw_crop_path": detection.get("raw_crop_path"),
                "open_world": detection.get("open_world"),
            }
        )
    final_results_path.write_text(
        json.dumps(
            {
                "job_id": args.job_id,
                "summary": summary,
                "class_records": class_records,
                "products": final_results,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    after_named = sum(1 for d in enriched if d.get("product_identity_label"))
    after_labels = Counter(
        (d.get("display_label") or d.get("label") or "?")
        for d in enriched
        if d.get("product_identity_label")
    )
    by_source = Counter(d.get("product_identity_source") or "<none>" for d in enriched)
    new_names = sorted(set(after_labels) - set(before_labels))

    print("=" * 60)
    print("RECOGNITION RESULTS")
    print("=" * 60)
    print(f"Total detections: {len(enriched)}")
    print(f"Named (before pipeline): {before_named}")
    print(f"Named (after pipeline):  {after_named}")
    print(f"Newly named this run:    {after_named - before_named}")
    print(f"Coverage: {after_named}/{len(enriched)} = {100.0 * after_named / max(1, len(enriched)):.1f}%")
    print()

    print("Source breakdown (after):")
    for src, n in by_source.most_common():
        print(f"  {n:4d}  {src}")
    print()

    named = [d for d in enriched if d.get("product_identity_label")]
    confidence_by_label: dict[str, list[float]] = {}
    for detection in named:
        label = detection.get("display_label") or detection.get("label") or "?"
        confidence_by_label.setdefault(label, []).append(_identity_confidence(detection))

    if named:
        confidences = [_identity_confidence(d) for d in named]
        print(
            "Identity confidence: "
            f"avg={sum(confidences) / len(confidences):.2f}, "
            f"min={min(confidences):.2f}, max={max(confidences):.2f}"
        )
        print()

    print("Top 15 named labels (after):")
    for lbl, n in after_labels.most_common(15):
        values = confidence_by_label.get(lbl, [])
        avg_conf = sum(values) / len(values) if values else 0.0
        min_conf = min(values) if values else 0.0
        print(f"  {n:3d}  conf avg={avg_conf:.2f} min={min_conf:.2f}  {lbl}")
    print()

    if named:
        print("Lowest-confidence named detections:")
        for detection in sorted(named, key=_identity_confidence)[:15]:
            label = detection.get("display_label") or detection.get("label") or "?"
            source = detection.get("product_identity_source") or "<none>"
            identity_conf = _identity_confidence(detection)
            detector_conf = detection.get("confidence")
            detector_conf_text = (
                f"{float(detector_conf):.2f}"
                if isinstance(detector_conf, (int, float))
                else "n/a"
            )
            print(
                f"  - {detection.get('id')}: identity_conf={identity_conf:.2f} "
                f"det_conf={detector_conf_text} source={source} label={label}"
            )
        print()

    if new_names:
        print(f"New label values introduced this run ({len(new_names)}):")
        for n in new_names:
            print(f"  + {n}")
        print()

    unnamed = [d for d in enriched if not d.get("product_identity_label")]
    if unnamed:
        print(f"Still unidentified ({len(unnamed)} detections):")
        for d in unnamed[:10]:
            cands = d.get("clip_candidates") or d.get("catalog_candidates") or []
            cand_str = ", ".join(
                f"{c.get('product_name')} ({c.get('confidence')})" for c in cands[:3] if isinstance(c, dict)
            )
            print(f"  - {d.get('id')}: top-clip-candidates=[{cand_str}]")
        if len(unnamed) > 10:
            print(f"  ... ({len(unnamed) - 10} more)")
        print()

    print("Class records:")
    for record in class_records["classes"]:
        cid = record["class_id"] if record["class_id"] is not None else "--"
        marker = "" if record["in_catalog"] else "  [not in catalog]"
        mean_conf = record["confidence"]["mean"]
        conf_text = f"{mean_conf:.2f}" if mean_conf is not None else "n/a"
        print(
            f"  id={cid!s:>3}  n={record['detection_count']:4d}  "
            f"imgs={record['image_count']:3d}  conf={conf_text}  "
            f"{record['class_name']}{marker}"
        )
    print(
        f"  ({class_records['unidentified_detection_count']} detections unidentified, "
        f"{class_records['registry_class_count']} classes registered)"
    )
    print()

    print("Summary metadata:")
    print(json.dumps(summary, indent=2))
    print()
    print(f"Final per-product results written to: {final_results_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
