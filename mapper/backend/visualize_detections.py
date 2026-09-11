#!/usr/bin/env python3
"""
visualize_detections.py — draw labeled bounding boxes on a single frame.

Usage:
    python visualize_detections.py <job_id> [image_name]

Examples:
    python visualize_detections.py iphone16-2
    python visualize_detections.py iphone16-2 00010.jpg
"""

import argparse
import json
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

GREEN      = (34, 197, 94)
YELLOW     = (234, 179, 8)
TEXT_COLOR = (255, 255, 255)
MAX_CHARS  = 32

FONT_CANDIDATES = [
    "/System/Library/Fonts/Helvetica.ttc",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
]

BASE_DIR      = Path(__file__).resolve().parent
WORKSPACE_DIR = BASE_DIR / "workspace"


def load_font(size: int):
    for path in FONT_CANDIDATES:
        try:
            return ImageFont.truetype(path, size)
        except (OSError, IOError):
            continue
    return ImageFont.load_default()


def best_frame(job_dir: Path) -> str | None:
    det_dir = job_dir / "detections"
    pn_dir  = job_dir / "product_names"
    best_stem, best_score = None, -1
    for det_file in sorted(det_dir.glob("*.json")):
        data  = json.loads(det_file.read_text(encoding="utf-8"))
        dets  = data.get("detections", [])
        named = sum(
            1 for d in dets
            if (pn_dir / f"{d['id']}.json").exists()
            and json.loads((pn_dir / f"{d['id']}.json").read_text(encoding="utf-8")).get("exact_match")
        )
        score = len(dets) * 2 + named
        if score > best_score:
            best_score, best_stem = score, det_file.stem
    return best_stem


def clamp_bbox(bbox, W: int, H: int) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = [int(v) for v in bbox]
    return max(0, x1), max(0, y1), min(W - 1, x2), min(H - 1, y2)


def build_label(det: dict, pn: dict | None) -> tuple[str, str, tuple]:
    """Return (line1, line2, color)."""
    model_label = (det.get("model_label") or "unknown").strip()
    det_conf    = det.get("confidence", 0.0)

    if pn and pn.get("exact_match") and pn.get("product_name"):
        name    = pn["product_name"]
        id_conf = pn.get("confidence", 0.0)
        # Wrap long names onto two display lines
        if len(name) > MAX_CHARS:
            mid = name.rfind(" ", 0, MAX_CHARS)
            mid = mid if mid > 0 else MAX_CHARS
            line1 = name[:mid]
            line2 = f"{name[mid:].strip()}  {det_conf:.0%} / {id_conf:.0%}"
        else:
            line1 = name
            line2 = f"det {det_conf:.0%}  id {id_conf:.0%}"
        return line1, line2, GREEN

    candidate = (pn.get("candidate") or "") if pn else ""
    display   = candidate[:MAX_CHARS] if candidate else model_label[:MAX_CHARS]
    return display, f"det {det_conf:.0%}  (unconfirmed)", YELLOW


def annotate(img: Image.Image, detections: list[dict], pn_dir: Path, scale: float):
    W, H = img.size

    line_w    = max(2, int(4 * scale))
    font_size = max(20, int(28 * scale))
    pad       = max(6, int(10 * scale))
    font      = load_font(font_size)
    sm_font   = load_font(max(16, int(22 * scale)))

    # Build RGBA overlay for semi-transparent label backgrounds
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    ov_draw = ImageDraw.Draw(overlay)

    # Draw boxes on original (full opacity, crisp)
    box_draw = ImageDraw.Draw(img)

    items = []
    for det in detections:
        pn_path = pn_dir / f"{det['id']}.json"
        pn      = json.loads(pn_path.read_text(encoding="utf-8")) if pn_path.exists() else None
        line1, line2, color = build_label(det, pn)
        x1, y1, x2, y2 = clamp_bbox(det["bbox"], W, H)
        items.append((x1, y1, x2, y2, color, line1, line2, pn))

    # Pass 1 — draw all boxes
    for x1, y1, x2, y2, color, *_ in items:
        box_draw.rectangle([(x1, y1), (x2, y2)], outline=color, width=line_w)

    # Pass 2 — draw labels INSIDE each box at top-left (semi-transparent bg)
    for x1, y1, x2, y2, color, line1, line2, _ in items:
        box_w = x2 - x1
        box_h = y2 - y1

        # Measure text
        w1 = int(box_draw.textlength(line1, font=font))
        w2 = int(box_draw.textlength(line2, font=sm_font))
        label_w = min(max(w1, w2) + pad * 2, box_w)
        label_h = font_size + max(16, int(22 * scale)) + pad * 3

        # Always inside the box — top-left corner
        lx1 = x1
        ly1 = y1
        lx2 = min(x1 + label_w, x2)
        ly2 = min(y1 + label_h, y2)

        # Semi-transparent fill (alpha=180 out of 255)
        ov_draw.rectangle([(lx1, ly1), (lx2, ly2)], fill=(*color, 180))

        # Text on the overlay too (fully opaque white)
        ov_draw.text((lx1 + pad, ly1 + pad), line1,
                     fill=(255, 255, 255, 255), font=font)
        ov_draw.text((lx1 + pad, ly1 + pad + font_size + pad // 2), line2,
                     fill=(255, 255, 255, 230), font=sm_font)

    # Composite overlay onto image
    result = Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB")
    return result, items


def main():
    parser = argparse.ArgumentParser(description="Draw detection boxes on a frame.")
    parser.add_argument("job_id", help="Job ID, e.g. iphone16-2")
    parser.add_argument("image_name", nargs="?", help="Frame filename, e.g. 00010.jpg")
    args = parser.parse_args()

    job_dir = WORKSPACE_DIR / args.job_id
    det_dir = job_dir / "detections"
    pn_dir  = job_dir / "product_names"
    img_dir = job_dir / "images"

    if not job_dir.exists():
        sys.exit(f"ERROR: Job not found: {job_dir}")
    if not det_dir.exists() or not any(det_dir.glob("*.json")):
        sys.exit(f"ERROR: No detection files in {det_dir}")

    if args.image_name:
        frame_stem = Path(args.image_name).stem
    else:
        frame_stem = best_frame(job_dir)
        if not frame_stem:
            sys.exit("ERROR: No non-empty detection file found.")

    img_path = img_dir / f"{frame_stem}.jpg"
    det_path = det_dir / f"{frame_stem}.json"

    if not img_path.exists():
        sys.exit(f"ERROR: Image not found: {img_path}")
    if not det_path.exists():
        sys.exit(f"ERROR: No detections for frame {frame_stem}")

    data = json.loads(det_path.read_text(encoding="utf-8"))
    dets = data.get("detections", [])
    if not dets:
        print(f"0 detections in {frame_stem}. Nothing to draw.")
        sys.exit(0)

    img   = Image.open(img_path).convert("RGB")
    W, H  = img.size
    scale = min(W, H) / 2160

    result, items = annotate(img, dets, pn_dir, scale)

    out_path = job_dir / f"annotated_{frame_stem}.jpg"
    result.save(str(out_path), "JPEG", quality=92)

    named   = sum(1 for *_, color, __, ___, ____ in items if color == GREEN)
    unnamed = len(items) - named

    print(f"Frame:      {frame_stem}.jpg  ({W} x {H})")
    print(f"Detections: {len(dets)}")
    print(f"  Named   (GREEN):  {named}   — exact product match")
    print(f"  Unnamed (YELLOW): {unnamed}  — unconfirmed")
    print()
    for det in dets:
        pn_path = pn_dir / f"{det['id']}.json"
        pn      = json.loads(pn_path.read_text(encoding="utf-8")) if pn_path.exists() else None
        tag     = "GREEN " if (pn and pn.get("exact_match")) else "YELLOW"
        name    = (pn.get("product_name") if pn else None) or det.get("model_label", "?")
        conf    = det.get("confidence", 0.0)
        print(f"  [{det['id']}]  {tag}  {name}  (det:{conf:.0%})")
    print()
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
