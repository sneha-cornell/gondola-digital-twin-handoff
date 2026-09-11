"""
Visual check: do Blender shelf assignments match image shelf rows?

For each unit's best image, draws every detected product bbox color-coded
by which Blender shelf it was assigned to. Also draws the median Y band per
shelf. If the colours form clean horizontal stripes → alignment is correct.

Outputs one annotated image per unit to --output-dir.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


# One color per shelf index (0=bottom … 5=top)
SHELF_COLORS = [
    (230,  57,  70),   # 0 bottom – red
    (244, 162,  97),   # 1 – orange
    ( 42, 157, 143),   # 2 – teal
    ( 72, 149, 239),   # 3 – blue
    (131,  56, 236),   # 4 – purple
    ( 36, 213,  96),   # 5 top – green
]


def shelf_index(shelf_name: str) -> int:
    try:
        return int(shelf_name.split("_Shelf_")[-1])
    except ValueError:
        return 0


def load_detections(job_dir: Path) -> dict[str, dict]:
    """id → detection dict with bbox."""
    out: dict[str, dict] = {}
    for f in (job_dir / "detections").glob("*.json"):
        data = json.loads(f.read_text())
        for d in data.get("detections", []):
            out[d["id"]] = d
    return out


def annotate_unit(
    unit: str,
    image_name: str,
    placements: list[dict],
    images_dir: Path,
    output_path: Path,
    scale: float = 0.4,
) -> None:
    img_path = images_dir / image_name
    if not img_path.exists():
        print(f"  image not found: {img_path}")
        return

    img = Image.open(img_path).convert("RGB")
    iw, ih = img.size

    # Scale down for output
    sw, sh = int(iw * scale), int(ih * scale)
    img = img.resize((sw, sh), Image.LANCZOS)
    draw = ImageDraw.Draw(img, "RGBA")

    # Filter to this unit+image
    unit_p = [p for p in placements
              if str(p.get("shelf_object", "")).startswith(unit)
              and p.get("image_name") == image_name
              and p.get("image_bbox")]

    # Group by shelf
    by_shelf: dict[str, list[dict]] = defaultdict(list)
    for p in unit_p:
        by_shelf[str(p["shelf_object"])].append(p)

    # Draw filled bbox per product
    for p in unit_p:
        x1, y1, x2, y2 = [v * scale for v in p["image_bbox"]]
        shelf = str(p["shelf_object"])
        idx = shelf_index(shelf)
        color = SHELF_COLORS[idx % len(SHELF_COLORS)]
        # Semi-transparent fill
        draw.rectangle([x1, y1, x2, y2], fill=(*color, 80), outline=(*color, 230), width=2)

    # Draw median horizontal band per shelf
    for shelf, items in sorted(by_shelf.items()):
        ys = [(p["image_bbox"][1] + p["image_bbox"][3]) * 0.5 * scale for p in items]
        y_med = sorted(ys)[len(ys) // 2]
        idx = shelf_index(shelf)
        color = SHELF_COLORS[idx % len(SHELF_COLORS)]
        draw.line([(0, y_med), (sw, y_med)], fill=(*color, 180), width=2)
        # Label on the left
        label = shelf.split("_Shelf_")[-1]
        draw.rectangle([2, y_med - 11, 30, y_med + 11], fill=(*color, 200))
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 14)
        except Exception:
            font = ImageFont.load_default()
        draw.text((5, y_med - 8), label, fill=(255, 255, 255), font=font)

    # Legend
    legend_x, legend_y = sw - 130, 10
    draw.rectangle([legend_x - 4, legend_y - 4, sw - 4, legend_y + len(by_shelf) * 22 + 4],
                   fill=(0, 0, 0, 160))
    for i, shelf in enumerate(sorted(by_shelf.keys(), key=shelf_index)):
        idx = shelf_index(shelf)
        color = SHELF_COLORS[idx % len(SHELF_COLORS)]
        n = len(by_shelf[shelf])
        draw.rectangle([legend_x, legend_y + i*22, legend_x+14, legend_y + i*22+14],
                       fill=color)
        draw.text((legend_x + 18, legend_y + i*22), f"{shelf.split('_Shelf_')[-1]}: {n}",
                  fill=(255, 255, 255), font=font)

    # Title
    draw.text((10, 8), f"{unit}  ←  {image_name}  ({len(unit_p)} detections)",
              fill=(255, 255, 80), font=font)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(output_path, quality=88)
    print(f"  saved {sw}×{sh}  {output_path.name}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--layout-json", required=True)
    parser.add_argument("--images-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    layout = json.loads(Path(args.layout_json).read_text())
    placements = layout.get("placements", [])
    selected_units = layout.get("selected_units", {})
    images_dir = Path(args.images_dir)
    output_dir = Path(args.output_dir)

    print(f"Checking {len(selected_units)} units, {len(placements)} placements\n")

    for unit, info in sorted(selected_units.items()):
        image_name = info["image_name"]
        print(f"[{unit}]  image={image_name}  "
              f"rows={info['row_count']}  shelves={info['shelf_count']}")
        annotate_unit(
            unit, image_name, placements,
            images_dir,
            output_dir / f"{unit.lower()}_shelf_check.jpg",
        )
    print("\nDone.")


if __name__ == "__main__":
    main()
