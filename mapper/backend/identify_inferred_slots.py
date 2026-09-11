"""
Identify inferred_unknown_slot products by two complementary strategies:

1. Neighbor propagation: most inferred slots are between detected products of
   the same brand/SKU (dense facings), so the nearest recognized neighbor(s) on
   the same shelf are the most reliable signal.

2. Visual projection + embedding: project the slot's 3D Blender position back
   into the source image via linear regression on known products, crop the
   region, and run the visual embedding classifier against the product catalog.

Both signals are combined: propagation is the primary result; visual embedding
is used to set product_name when it agrees with propagation, or provides a
fallback when propagation has no clear winner.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--merged-json", required=True)
    parser.add_argument("--job-dir", required=True)
    parser.add_argument("--images-dir", required=True, help="Directory containing source images")
    parser.add_argument("--output-json", required=True)
    parser.add_argument(
        "--crop-padding",
        type=float,
        default=1.25,
        help="Multiply median crop size by this factor",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Project and crop but skip recognition, print projected positions",
    )
    parser.add_argument(
        "--min-crop-px",
        type=int,
        default=80,
        help="Minimum crop side in pixels (skip crops smaller than this)",
    )
    parser.add_argument(
        "--visual-threshold",
        type=float,
        default=0.65,
        help="Minimum embedding similarity confidence to accept visual identification",
    )
    parser.add_argument(
        "--no-visual",
        action="store_true",
        help="Skip visual embedding; use only neighbor propagation",
    )
    parser.add_argument(
        "--no-ocr",
        action="store_true",
        help="Skip OCR; use only neighbor propagation and visual embedding",
    )
    parser.add_argument(
        "--ocr-threshold",
        type=float,
        default=0.52,
        help="Minimum OCR KB-match score to accept (0-1)",
    )
    parser.add_argument(
        "--knowledge-base",
        default="knowledge_base",
        help="Path to knowledge base directory (for OCR KB matching)",
    )
    return parser.parse_args()


def horizontal_axis(unit: str) -> int:
    """Return Blender location axis index for horizontal shelf direction."""
    return 1 if unit in ("Unit2", "Unit4") else 0


def build_shelf_index(placements: list[dict]) -> dict[str, list[dict]]:
    """Index recognized/unknown placements by shelf_object, sorted by horizontal coord."""
    by_shelf: dict[str, list[dict]] = defaultdict(list)
    for p in placements:
        if p.get("product_identity_status") not in ("recognized", "unknown"):
            continue
        shelf = p.get("shelf_object", "")
        if not shelf:
            continue
        unit = shelf.split("_")[0]
        axis = horizontal_axis(unit)
        by_shelf[shelf].append(
            {
                "h": p["location"][axis],
                "name": p.get("product_name", "Unknown"),
                "status": p.get("product_identity_status", ""),
            }
        )
    for shelf in by_shelf:
        by_shelf[shelf].sort(key=lambda x: x["h"])
    return dict(by_shelf)


def propagate_neighbors(
    shelf: str,
    h_coord: float,
    shelf_index: dict[str, list[dict]],
) -> tuple[str | None, str]:
    """
    Infer a product name from shelf neighbors.

    Returns (product_name, confidence_label) where confidence_label is one of:
    "high"  — both neighbors are the same recognized product
    "medium" — only one side has a neighbor, or one neighbor is recognized
    "low"   — neighbors differ; use shelf mode
    None    — no data
    """
    known = shelf_index.get(shelf, [])
    if not known:
        return None, "none"

    recognized = [k for k in known if k["name"] not in ("Unknown", None)]
    if not recognized:
        return None, "none"

    left = [k for k in recognized if k["h"] < h_coord]
    right = [k for k in recognized if k["h"] > h_coord]

    left_name = left[-1]["name"] if left else None
    right_name = right[0]["name"] if right else None

    if left_name and right_name:
        if left_name == right_name:
            return left_name, "high"
        # Different neighbors — use closest
        left_dist = h_coord - left[-1]["h"]
        right_dist = right[0]["h"] - h_coord
        shelf_mode = Counter(k["name"] for k in recognized).most_common(1)[0][0]
        if left_name == shelf_mode or right_name == shelf_mode:
            return shelf_mode, "low"
        return (left_name if left_dist <= right_dist else right_name), "low"

    if left_name:
        return left_name, "medium"
    if right_name:
        return right_name, "medium"

    # Only Unknown neighbors
    shelf_mode = Counter(k["name"] for k in recognized).most_common(1)
    if shelf_mode:
        return shelf_mode[0][0], "low"
    return None, "none"


def fit_linear(xs: list[float], ys: list[float]) -> tuple[float, float] | None:
    """Fit y = a*x + b. Returns (a, b) or None if fewer than 2 points."""
    if len(xs) < 2:
        return None
    coeffs = np.polyfit(xs, ys, 1)
    return float(coeffs[0]), float(coeffs[1])


def apply_linear(mapping: tuple[float, float], x: float) -> float:
    return mapping[0] * x + mapping[1]


def build_unit_calibration(
    placements: list[dict],
) -> dict[str, dict]:
    """
    Build per-unit calibration from recognized/unknown placements that have image_bbox.

    Returns a dict: unit -> {
        "h_mapping": (a, b),  # blender_hcoord -> img_cx
        "z_mapping": (a, b),  # blender_z      -> img_cy
        "med_h": float,       # median image bbox height
        "med_w": float,       # median image bbox width
        "image_name": str,
    }
    """
    unit_data: dict[str, list] = defaultdict(list)

    for p in placements:
        if p.get("product_identity_status") not in ("recognized", "unknown"):
            continue
        if not p.get("image_bbox") or not p.get("shelf_object"):
            continue
        shelf = p["shelf_object"]
        unit = shelf.split("_")[0]
        axis = horizontal_axis(unit)
        loc = p["location"]
        bbox = p["image_bbox"]
        img_cx = (bbox[0] + bbox[2]) / 2
        img_cy = (bbox[1] + bbox[3]) / 2
        img_h = bbox[3] - bbox[1]
        img_w = bbox[2] - bbox[0]
        unit_data[unit].append(
            {
                "blender_h": loc[axis],
                "blender_z": loc[2],
                "img_cx": img_cx,
                "img_cy": img_cy,
                "img_h": img_h,
                "img_w": img_w,
                "image_name": p.get("image_name", ""),
            }
        )

    calibration: dict[str, dict] = {}
    for unit, pts in unit_data.items():
        h_coords = [p["blender_h"] for p in pts]
        img_cxs = [p["img_cx"] for p in pts]
        z_coords = [p["blender_z"] for p in pts]
        img_cys = [p["img_cy"] for p in pts]

        h_mapping = fit_linear(h_coords, img_cxs)
        z_mapping = fit_linear(z_coords, img_cys)

        med_h = statistics.median(p["img_h"] for p in pts)
        med_w = statistics.median(p["img_w"] for p in pts)

        # Most common image name
        names = [p["image_name"] for p in pts if p["image_name"]]
        image_name = max(set(names), key=names.count) if names else ""

        calibration[unit] = {
            "h_mapping": h_mapping,
            "z_mapping": z_mapping,
            "med_h": med_h,
            "med_w": med_w,
            "image_name": image_name,
            "n_points": len(pts),
        }
        print(
            f"  {unit}: {len(pts)} calibration points, "
            f"h_mapping=({h_mapping[0]:.1f}x+{h_mapping[1]:.0f}) "
            f"z_mapping=({z_mapping[0]:.1f}x+{z_mapping[1]:.0f}) "
            f"med_crop=({med_w:.0f}x{med_h:.0f}) img={image_name}"
            if h_mapping and z_mapping
            else f"  {unit}: {len(pts)} calibration points, insufficient data"
        )

    return calibration


def crop_and_identify(
    image: Image.Image,
    px: float,
    py: float,
    med_w: float,
    med_h: float,
    padding: float,
    crop_path: Path,
    identifier,  # ProductIdentifier
    min_crop_px: int,
    dry_run: bool,
) -> dict | None:
    """
    Crop the image at the projected position and run recognition.
    Returns the identification result, or None if the crop is too small.
    """
    iw, ih = image.size
    half_w = (med_w / 2) * padding
    half_h = (med_h / 2) * padding

    x1 = max(0, int(px - half_w))
    x2 = min(iw, int(px + half_w))
    y1 = max(0, int(py - half_h))
    y2 = min(ih, int(py + half_h))

    crop_w = x2 - x1
    crop_h = y2 - y1

    if crop_w < min_crop_px or crop_h < min_crop_px:
        return None

    crop = image.crop((x1, y1, x2, y2))
    crop.save(str(crop_path), quality=95)

    if dry_run:
        return {
            "product_name": None,
            "dry_run": True,
            "projected_px": (px, py),
            "crop_bbox": [x1, y1, x2, y2],
        }

    result = identifier.identify_crop(crop_path)
    result["crop_bbox"] = [x1, y1, x2, y2]
    result["projected_px"] = (px, py)
    return result


def _enrich_unknown_with_ocr(
    idx: int,
    p: dict,
    placements: list[dict],
    ocr_matcher,
    images_dir: Path,
    image_cache: dict,
) -> None:
    """Run OCR on raw-unknown products and store ocr_text even if no KB match."""
    bbox = p.get("image_bbox")
    img_name = p.get("image_name", "")
    if not bbox or not img_name:
        return
    path = images_dir / img_name
    if not path.exists():
        if img_name not in image_cache:
            return
    img = image_cache.get(img_name)
    if img is None:
        try:
            img = Image.open(path)
            image_cache[img_name] = img
        except Exception:
            return
    x1, y1, x2, y2 = [int(v) for v in bbox]
    crop = img.crop((x1, y1, x2, y2))
    try:
        result = ocr_matcher.match(crop)
        if result.get("ocr_text"):
            placements[idx]["ocr_text"] = result["ocr_text"]
        if result.get("product_name") and result.get("confidence", 0) >= ocr_matcher.threshold:
            placements[idx].update({
                "product_name": result["product_name"],
                "product_identity_status": "recognized",
                "identification_source": f"ocr_kb_match (conf={result['confidence']:.2f})",
                "ocr_confidence": result["confidence"],
                "ocr_text": result["ocr_text"],
            })
            print(f"  OCR_UNKNOWN  {p['object']} -> {result['product_name']!r} (conf={result['confidence']:.2f})")
    except Exception:
        pass


def main() -> None:
    args = parse_args()

    data = json.load(open(args.merged_json))
    placements = data["placements"]
    job_dir = Path(args.job_dir)
    images_dir = Path(args.images_dir)
    crops_dir = job_dir / "focus_crops"
    crops_dir.mkdir(exist_ok=True)

    print("Building calibration from recognized products...")
    calibration = build_unit_calibration(placements)

    # Override image_name from selected_units if available
    selected_units = data.get("selected_units", {})
    for unit, sel in selected_units.items():
        if unit in calibration:
            calibration[unit]["image_name"] = sel["image_name"]

    print("Building shelf neighbor index...")
    shelf_index = build_shelf_index(placements)

    if not args.dry_run and not args.no_visual:
        print("Loading visual embedding classifier...")
        os.environ.setdefault("PRODUCT_NAME_OCR", "0")
        os.environ.setdefault(
            "PRODUCT_NAME_VISUAL_ONLY_CONFIDENCE", str(args.visual_threshold)
        )
        from embedding_classifier import EmbeddingClassifier

        embedder = EmbeddingClassifier()
        print(f"  embedding_classifier enabled={embedder.enabled} products={embedder.product_count()}")
    else:
        embedder = None

    ocr_matcher = None
    if not args.dry_run and not args.no_ocr:
        print("Loading OCR product matcher...")
        from ocr_product_matcher import OCRProductMatcher
        ocr_matcher = OCRProductMatcher(
            kb_dir=Path(args.knowledge_base),
            kb_match_threshold=args.ocr_threshold,
        )
        print(f"  OCR KB products={len(ocr_matcher._product_names)}")

    # Cache loaded images
    image_cache: dict[str, Image.Image] = {}

    def load_image(name: str) -> Image.Image | None:
        if name in image_cache:
            return image_cache[name]
        path = images_dir / name
        if not path.exists():
            print(f"  WARNING: image not found: {path}")
            return None
        img = Image.open(path)
        image_cache[name] = img
        return img

    total = 0
    propagated = 0
    visual_identified = 0
    ocr_identified = 0
    still_unknown = 0
    skipped = 0

    for idx, p in enumerate(placements):
        # Also enrich raw unknowns with OCR text even if we don't change their status
        if p.get("product_identity_status") == "unknown" and ocr_matcher:
            _enrich_unknown_with_ocr(idx, p, placements, ocr_matcher, images_dir, image_cache)

        if p.get("product_identity_status") != "inferred_unknown_slot":
            continue

        shelf = p["shelf_object"]
        unit = shelf.split("_")[0]
        total += 1
        loc = p["location"]
        axis = horizontal_axis(unit)
        blender_h = loc[axis]
        blender_z = loc[2]

        # --- Strategy 1: neighbor propagation ---
        prop_name, prop_conf = propagate_neighbors(shelf, blender_h, shelf_index)

        # --- Strategy 2: visual embedding projection ---
        visual_name: str | None = None
        visual_conf: float = 0.0
        crop_bbox: list = []
        crop_path_str: str = ""
        img_name: str = ""

        calib = calibration.get(unit)
        if calib and calib.get("h_mapping") and calib.get("z_mapping"):
            px = apply_linear(calib["h_mapping"], blender_h)
            py = apply_linear(calib["z_mapping"], blender_z)
            img_name = calib["image_name"] or ""

            if img_name and 0 <= px and 0 <= py:
                image = load_image(img_name)
                if image is not None:
                    iw, ih = image.size
                    if px <= iw and py <= ih:
                        crop_name = f"inferred_{p['object']}.jpg"
                        crop_path = crops_dir / crop_name
                        half_w = (calib["med_w"] / 2) * args.crop_padding
                        half_h = (calib["med_h"] / 2) * args.crop_padding
                        x1 = max(0, int(px - half_w))
                        x2 = min(iw, int(px + half_w))
                        y1 = max(0, int(py - half_h))
                        y2 = min(ih, int(py + half_h))
                        crop_bbox = [x1, y1, x2, y2]

                        if x2 - x1 >= args.min_crop_px and y2 - y1 >= args.min_crop_px:
                            crop_img = image.crop((x1, y1, x2, y2))
                            crop_img.save(str(crop_path), quality=95)
                            crop_path_str = str(crop_path)

                            if not args.dry_run and embedder and embedder.enabled:
                                try:
                                    result = embedder.classify(crop_img.convert("RGB"))
                                    if result.get("exact_match") and result.get("product_name"):
                                        visual_name = result["product_name"]
                                        visual_conf = float(result.get("confidence") or 0.0)
                                    elif result.get("candidates"):
                                        top = result["candidates"][0]
                                        if float(top.get("confidence") or 0) >= args.visual_threshold:
                                            visual_name = top["product_name"]
                                            visual_conf = float(top["confidence"])
                                except Exception:
                                    pass

        # --- Strategy 3: OCR on the projected crop ---
        ocr_name: str | None = None
        ocr_conf: float = 0.0
        ocr_raw_text: str = ""
        if not args.dry_run and ocr_matcher and crop_path_str:
            try:
                crop_img_for_ocr = Image.open(crop_path_str).convert("RGB")
                ocr_result = ocr_matcher.match(crop_img_for_ocr)
                ocr_raw_text = ocr_result.get("ocr_text", "")
                if ocr_result.get("product_name"):
                    ocr_name = ocr_result["product_name"]
                    ocr_conf = float(ocr_result.get("confidence") or 0.0)
            except Exception:
                pass

        if args.dry_run:
            print(
                f"  DRY RUN {p['object']} [{shelf}]: "
                f"propagation={prop_name!r}({prop_conf}) "
                f"visual={visual_name!r}({visual_conf:.2f}) "
                f"ocr={ocr_name!r}({ocr_conf:.2f}) ocr_text={ocr_raw_text!r} "
                f"crop={crop_bbox}"
            )
            continue

        # --- Combine signals (priority: OCR KB match > visual+prop > propagation > OCR alone) ---
        final_name: str | None = None
        id_source: str = ""

        # All three agree
        if ocr_name and visual_name and prop_name and ocr_name == visual_name == prop_name:
            final_name = ocr_name
            id_source = f"ocr+visual+propagation ({prop_conf})"
        # OCR agrees with propagation
        elif ocr_name and prop_name and ocr_name == prop_name:
            final_name = ocr_name
            id_source = f"ocr+propagation ({prop_conf}, ocr_conf={ocr_conf:.2f})"
        # OCR agrees with visual
        elif ocr_name and visual_name and ocr_name == visual_name:
            final_name = ocr_name
            id_source = f"ocr+visual (ocr_conf={ocr_conf:.2f})"
        # OCR alone (strong match)
        elif ocr_name and ocr_conf >= args.ocr_threshold:
            final_name = ocr_name
            id_source = f"ocr (conf={ocr_conf:.2f})"
        # Visual + propagation agree
        elif visual_name and prop_name and visual_name == prop_name:
            final_name = visual_name
            id_source = f"propagation+visual ({prop_conf})"
        # Visual alone (strong)
        elif visual_name and visual_conf >= args.visual_threshold:
            final_name = visual_name
            id_source = f"visual (conf={visual_conf:.2f})"
        # Propagation alone
        elif prop_name and prop_conf in ("high", "medium"):
            final_name = prop_name
            id_source = f"propagation ({prop_conf})"
        elif prop_name:
            final_name = prop_name
            id_source = f"propagation ({prop_conf}/shelf_mode)"

        if final_name and final_name != "Unknown":
            new_status = "recognized"
            if "ocr" in id_source:
                new_status = "recognized"
            elif prop_conf in ("high", "medium") and not visual_name:
                new_status = "inferred_neighbor"
            elif prop_conf == "low" and not visual_name:
                new_status = "inferred_neighbor_low"

            placements[idx].update(
                {
                    "product_name": final_name,
                    "product_identity_status": new_status,
                    "identification_source": id_source,
                    "propagated_neighbor_confidence": prop_conf,
                    "visual_confidence": visual_conf,
                    "ocr_confidence": ocr_conf,
                }
            )
            if ocr_raw_text:
                placements[idx]["ocr_text"] = ocr_raw_text
            if crop_path_str:
                placements[idx]["crop_file"] = crop_path_str
                placements[idx]["image_name"] = img_name
                placements[idx]["image_bbox"] = crop_bbox

            if "ocr" in id_source:
                ocr_identified += 1
            elif "visual" in id_source:
                visual_identified += 1
            else:
                propagated += 1
            label = (
                "OCR+PROP" if "ocr" in id_source and "propagation" in id_source
                else "OCR+VIS" if "ocr" in id_source and "visual" in id_source
                else "OCR" if "ocr" in id_source
                else "VIS+PROP" if "+" in id_source
                else "VISUAL" if "visual" in id_source
                else "PROPAGATED"
            )
            print(f"  {label:12s} {p['object']} -> {final_name!r} [{id_source}]")
        else:
            still_unknown += 1
            update = {"identification_source": "projection_unknown"}
            if ocr_raw_text:
                update["ocr_text"] = ocr_raw_text
            if crop_path_str:
                update.update({"crop_file": crop_path_str, "image_name": img_name, "image_bbox": crop_bbox})
            placements[idx].update(update)
            ocr_hint = f" ocr_text={ocr_raw_text!r}" if ocr_raw_text else ""
            print(f"  STILL_UNK  {p['object']} [{shelf}] prop={prop_name!r}({prop_conf}){ocr_hint}")

    if not args.dry_run:
        # Update summary counts
        by_status: dict[str, int] = defaultdict(int)
        for p in placements:
            by_status[p["product_identity_status"]] += 1

        data["recognized_products"] = by_status.get("recognized", 0)
        data["unknown_products"] = by_status.get("unknown", 0)
        data["inferred_unknown_slots"] = by_status.get("inferred_unknown_slot", 0)
        data["inferred_neighbor_slots"] = by_status.get("inferred_neighbor", 0) + by_status.get("inferred_neighbor_low", 0)

        with open(args.output_json, "w") as f:
            json.dump(data, f, indent=2)
        print(f"\nSaved updated JSON to {args.output_json}")
        print(
            f"Results: total={total} propagated={propagated} visual={visual_identified} "
            f"ocr={ocr_identified} unknown={still_unknown} skipped={skipped}"
        )
        print("Status breakdown:", dict(by_status))
    else:
        print(f"\nDry run complete: {total} inferred slots processed")


if __name__ == "__main__":
    main()
