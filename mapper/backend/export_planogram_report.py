"""
Export shelf map as a planogram HTML report + CSV.

Usage:
  python3 export_planogram_report.py \
    --identified-json  <path/blender_single_best_image_layout_identified.json> \
    --renders-dir      <dir with unit1_labeled.png … render_overview_final.png> \
    --knowledge-base   <path/knowledge_base> \
    --output-html      <path/planogram_report.html> \
    --output-csv       <path/planogram_data.csv>
"""
from __future__ import annotations

import argparse
import base64
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path


# ── helpers ──────────────────────────────────────────────────────────────────

UNIT_AXIS = {
    "Unit1": 0, "Unit3": 0,   # horizontal = X
    "Unit2": 1, "Unit4": 1,   # horizontal = Y
}

STATUS_COLOR = {
    "recognized":         "#4e8fb5",   # blue
    "inferred_neighbor":  "#d9952d",   # amber
    "inferred_neighbor_low": "#d4d44a", # yellow
    "unknown":            "#aaaaaa",   # grey
}
STATUS_LABEL = {
    "recognized":            "Recognized",
    "inferred_neighbor":     "Inferred (high)",
    "inferred_neighbor_low": "Inferred (low)",
    "unknown":               "Unknown",
}

SHELF_DISPLAY = {
    "Unit1_Shelf_04": "Top",    "Unit1_Shelf_03": "Upper",
    "Unit1_Shelf_02": "Middle", "Unit1_Shelf_01": "Lower",
    "Unit1_Shelf_00": "Bottom",
    "Unit2_Shelf_05": "Top",    "Unit2_Shelf_04": "Upper-2",
    "Unit2_Shelf_03": "Upper-1","Unit2_Shelf_02": "Middle",
    "Unit2_Shelf_01": "Lower",  "Unit2_Shelf_00": "Bottom",
    "Unit3_Shelf_04": "Top",    "Unit3_Shelf_03": "Upper",
    "Unit3_Shelf_02": "Middle", "Unit3_Shelf_01": "Lower",
    "Unit3_Shelf_00": "Bottom",
    "Unit4_Shelf_05": "Top",    "Unit4_Shelf_04": "Upper-2",
    "Unit4_Shelf_03": "Upper-1","Unit4_Shelf_02": "Middle",
    "Unit4_Shelf_01": "Lower",  "Unit4_Shelf_00": "Bottom",
}


def img_b64(path: Path) -> str:
    # Prefer thumbnail if it exists
    thumb = path.with_stem(path.stem + "_thumb")
    src = thumb if thumb.exists() else path
    if not src.exists():
        return ""
    data = base64.b64encode(src.read_bytes()).decode()
    return f"data:image/png;base64,{data}"


def img_b64_raw(path: Path, max_px: int = 120) -> str:
    """Read image, resize to max_px on longest side, return data URI."""
    if not path or not path.exists():
        return ""
    try:
        from PIL import Image as PILImage
        import io
        img = PILImage.open(path).convert("RGB")
        w, h = img.size
        if max(w, h) > max_px:
            scale = max_px / max(w, h)
            img = img.resize((max(1, int(w * scale)), max(1, int(h * scale))), PILImage.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=72, optimize=True)
        data = base64.b64encode(buf.getvalue()).decode()
        return f"data:image/jpeg;base64,{data}"
    except Exception:
        suffix = path.suffix.lower()
        mime = "image/jpeg" if suffix in (".jpg", ".jpeg") else "image/png"
        data = base64.b64encode(path.read_bytes()).decode()
        return f"data:{mime};base64,{data}"


def build_product_image_map(
    kb_dir: Path,
    placements: list[dict],
) -> dict[str, str]:
    """Return {product_name: data-uri} using KB front > KB auto > crop fallback."""
    # Collect best crop per product from placements (first recognized one)
    crop_fallback: dict[str, Path] = {}
    for p in placements:
        name = str(p.get("product_name") or "")
        if name and name not in crop_fallback:
            cf = p.get("crop_file") or ""
            if cf:
                cp = Path(cf)
                if cp.exists():
                    crop_fallback[name] = cp

    img_map: dict[str, str] = {}
    if kb_dir and kb_dir.exists():
        for prod_dir in sorted(kb_dir.iterdir()):
            if not prod_dir.is_dir() or prod_dir.name.startswith(("open_world", "_")):
                continue
            name = prod_dir.name
            # Priority: front → auto → crop
            candidates: list[Path] = []
            for sub in ("front", "auto"):
                sub_dir = prod_dir / sub
                if sub_dir.exists():
                    candidates.extend(sorted(sub_dir.glob("*.jpg")))
                    candidates.extend(sorted(sub_dir.glob("*.png")))
            if candidates:
                img_map[name] = img_b64_raw(candidates[0])

    # Fill gaps with crop fallbacks
    all_names = {str(p.get("product_name") or "") for p in placements}
    for name in all_names:
        if name and name not in img_map and name in crop_fallback:
            img_map[name] = img_b64_raw(crop_fallback[name])

    return img_map


def product_thumb(name: str, size: int = 48) -> str:
    """Return an <img> that loads its src from the JS IMGS map at runtime."""
    safe = name.replace("'", "\\'").replace('"', "&quot;")
    return (
        f'<img data-prod="{safe}" style="width:{size}px;height:{size}px;'
        f'object-fit:contain;border-radius:3px;background:#f5f5f5;flex-shrink:0">'
    )


def img_map_script(img_map: dict[str, str]) -> str:
    """Emit a <script> that maps product name → data URI, then sets all img src."""
    entries = ",\n".join(
        f'  {json.dumps(k)}: {json.dumps(v)}'
        for k, v in img_map.items()
    )
    return f"""<script>
(function(){{
  var IMGS = {{\n{entries}\n  }};
  document.addEventListener('DOMContentLoaded', function(){{
    document.querySelectorAll('img[data-prod]').forEach(function(el){{
      var src = IMGS[el.getAttribute('data-prod')];
      if (src) el.src = src;
      else el.style.background = '#e0e0e0';
    }});
  }});
}})();
</script>"""


def shelf_sort_key(shelf: str) -> tuple:
    parts = shelf.split("_Shelf_")
    unit = parts[0]
    idx = int(parts[1]) if len(parts) > 1 else 0
    unit_order = {"Unit1": 0, "Unit2": 1, "Unit3": 2, "Unit4": 3}
    return (unit_order.get(unit, 9), -idx)   # highest shelf index = top row first


# ── core data build ───────────────────────────────────────────────────────────

def build_shelf_rows(placements: list[dict]) -> dict[str, list[dict]]:
    """Return {shelf: [placements sorted left→right]}."""
    by_shelf: dict[str, list[dict]] = defaultdict(list)
    for p in placements:
        shelf = str(p.get("shelf_object") or "")
        if not shelf:
            continue
        by_shelf[shelf].append(p)

    result = {}
    for shelf, items in by_shelf.items():
        unit = shelf.split("_")[0]
        axis = UNIT_AXIS.get(unit, 0)
        result[shelf] = sorted(items, key=lambda x: float((x.get("location") or [0, 0, 0])[axis]))
    return result


def product_summary(placements: list[dict]) -> list[dict]:
    """Return products sorted by total facings desc."""
    counter: Counter = Counter()
    by_unit: dict[str, Counter] = defaultdict(Counter)
    for p in placements:
        name = str(p.get("product_name") or "Unknown")
        status = str(p.get("product_identity_status") or "")
        if status == "unknown":
            name = "Unknown"
        counter[name] += 1
        unit = str(p.get("shelf_object") or "").split("_")[0]
        by_unit[unit][name] += 1

    rows = []
    for name, total in counter.most_common():
        rows.append({
            "product": name,
            "total": total,
            "unit1": by_unit["Unit1"][name],
            "unit2": by_unit["Unit2"][name],
            "unit3": by_unit["Unit3"][name],
            "unit4": by_unit["Unit4"][name],
        })
    return rows


# ── HTML builder ──────────────────────────────────────────────────────────────

def render_shelf_table(shelf: str, items: list[dict]) -> str:
    label = SHELF_DISPLAY.get(shelf, shelf.split("_Shelf_")[-1])
    rows = []
    for i, p in enumerate(items, 1):
        name = str(p.get("product_name") or "Unknown")
        status = str(p.get("product_identity_status") or "")
        color = STATUS_COLOR.get(status, "#cccccc")
        conf = str(p.get("propagated_neighbor_confidence") or p.get("identification_source") or "direct")
        thumb = product_thumb(name, size=36)
        rows.append(
            f'<tr title="{status} | {conf}">'
            f'<td style="padding:3px 5px;color:#888;font-size:10px;vertical-align:middle">{i}</td>'
            f'<td style="padding:2px 4px;vertical-align:middle">{thumb}</td>'
            f'<td style="padding:3px 6px;vertical-align:middle">'
            f'<span style="background:{color};color:#fff;padding:2px 6px;border-radius:4px;'
            f'font-size:11px;white-space:nowrap">{name}</span>'
            f'</td></tr>'
        )
    rows_html = "\n".join(rows)
    return (
        f'<table style="border-collapse:collapse;width:100%">'
        f'<thead><tr><th colspan="3" style="background:#444;color:#fff;padding:4px 8px;'
        f'text-align:left;font-size:12px">{label}</th></tr></thead>'
        f'<tbody>{rows_html}</tbody></table>'
    )


def render_unit_section(unit: str, shelf_rows: dict[str, list[dict]], renders_dir: Path) -> str:
    img_src = img_b64(renders_dir / f"{unit.lower()}_labeled.png")
    img_tag = (
        f'<img src="{img_src}" style="width:100%;border-radius:6px;box-shadow:0 2px 8px #0003">'
        if img_src else '<p style="color:#999">render not found</p>'
    )

    unit_shelves = sorted(
        [s for s in shelf_rows if s.startswith(unit + "_")],
        key=shelf_sort_key,
    )
    shelf_html = ""
    for shelf in unit_shelves:
        items = shelf_rows[shelf]
        shelf_html += f'<div style="margin-bottom:8px">{render_shelf_table(shelf, items)}</div>'

    total = sum(len(shelf_rows[s]) for s in unit_shelves)
    recognized = sum(
        sum(1 for p in shelf_rows[s] if p.get("product_identity_status") == "recognized")
        for s in unit_shelves
    )

    return f"""
<div style="margin-bottom:40px;background:#fff;border-radius:8px;
            box-shadow:0 2px 12px #0002;padding:20px">
  <h2 style="margin:0 0 12px;color:#333;font-size:18px">
    {unit} &nbsp;<span style="font-weight:normal;color:#666;font-size:14px">
      {total} facings &nbsp;|&nbsp; {recognized} recognized
      &nbsp;({round(recognized/total*100) if total else 0}%)
    </span>
  </h2>
  <div style="display:grid;grid-template-columns:1fr 1fr;gap:20px;align-items:start">
    <div>{img_tag}</div>
    <div style="max-height:520px;overflow-y:auto">{shelf_html}</div>
  </div>
</div>"""


def render_summary_table(summary: list[dict]) -> str:
    rows = ""
    for i, r in enumerate(summary):
        bg = "#f9f9f9" if i % 2 == 0 else "#fff"
        thumb = product_thumb(r["product"], size=44)
        rows += (
            f'<tr style="background:{bg}">'
            f'<td style="padding:4px 8px;color:#888;font-size:12px">{i+1}</td>'
            f'<td style="padding:4px 6px">{thumb}</td>'
            f'<td style="padding:4px 10px;font-weight:500;font-size:13px">{r["product"]}</td>'
            f'<td style="padding:4px 10px;text-align:center;font-size:13px">{r["total"]}</td>'
            f'<td style="padding:4px 10px;text-align:center;font-size:13px">{r["unit1"] or ""}</td>'
            f'<td style="padding:4px 10px;text-align:center;font-size:13px">{r["unit2"] or ""}</td>'
            f'<td style="padding:4px 10px;text-align:center;font-size:13px">{r["unit3"] or ""}</td>'
            f'<td style="padding:4px 10px;text-align:center;font-size:13px">{r["unit4"] or ""}</td>'
            f'</tr>'
        )
    header = (
        '<tr style="background:#444;color:#fff">'
        '<th style="padding:6px 8px">#</th>'
        '<th style="padding:6px 8px"></th>'
        '<th style="padding:6px 10px;text-align:left">Product</th>'
        '<th style="padding:6px 10px">Total</th>'
        '<th style="padding:6px 10px">Unit1</th>'
        '<th style="padding:6px 10px">Unit2</th>'
        '<th style="padding:6px 10px">Unit3</th>'
        '<th style="padding:6px 10px">Unit4</th>'
        '</tr>'
    )
    return (
        f'<table style="border-collapse:collapse;width:100%;font-size:13px">'
        f'<thead>{header}</thead><tbody>{rows}</tbody></table>'
    )


def render_legend() -> str:
    items = "".join(
        f'<span style="background:{c};color:#fff;padding:3px 10px;border-radius:4px;'
        f'margin-right:8px;font-size:12px">{STATUS_LABEL[s]}</span>'
        for s, c in STATUS_COLOR.items()
    )
    return f'<div style="margin-bottom:24px">{items}</div>'


def build_html(
    placements: list[dict],
    shelf_rows: dict[str, list[dict]],
    summary: list[dict],
    renders_dir: Path,
    img_map: dict[str, str],
    meta: dict,
) -> str:
    imgs_script = img_map_script(img_map)
    overview_src = img_b64(renders_dir / "render_overview_final.png")
    overview_tag = (
        f'<img src="{overview_src}" style="width:100%;border-radius:6px;'
        f'box-shadow:0 2px 8px #0003;margin-bottom:32px">'
        if overview_src else ""
    )

    unit_sections = "".join(
        render_unit_section(u, shelf_rows, renders_dir)
        for u in ["Unit1", "Unit2", "Unit3", "Unit4"]
    )

    total = len(placements)
    recognized = sum(1 for p in placements if p.get("product_identity_status") == "recognized")
    inferred = sum(1 for p in placements if "inferred" in str(p.get("product_identity_status", "")))
    unique_products = len({p.get("product_name") for p in placements
                           if p.get("product_name") not in (None, "Unknown", "Blank")})

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Planogram Report — Shelf 22-2</title>
{imgs_script}
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
          background:#f0f2f5; margin:0; padding:32px; color:#222; }}
  h1 {{ font-size:26px; margin:0 0 6px; }}
  .meta {{ color:#666; font-size:13px; margin-bottom:28px; }}
  .kpi {{ display:flex; gap:16px; margin-bottom:32px; flex-wrap:wrap; }}
  .kpi-card {{ background:#fff; border-radius:8px; padding:16px 24px;
               box-shadow:0 2px 8px #0002; min-width:120px; text-align:center; }}
  .kpi-val {{ font-size:32px; font-weight:700; color:#4e8fb5; }}
  .kpi-lbl {{ font-size:12px; color:#888; margin-top:2px; }}
  h2.section {{ font-size:20px; margin:32px 0 12px; color:#333; border-bottom:2px solid #ddd; padding-bottom:6px; }}
</style>
</head>
<body>
<h1>Planogram Report — Shelf 22-2</h1>
<div class="meta">
  Generated from: blender_single_best_image_layout_identified.json
  &nbsp;|&nbsp; {total} total facings across 4 rack faces
</div>

<div class="kpi">
  <div class="kpi-card"><div class="kpi-val">{total}</div><div class="kpi-lbl">Total Facings</div></div>
  <div class="kpi-card"><div class="kpi-val">{recognized}</div><div class="kpi-lbl">Recognized</div></div>
  <div class="kpi-card"><div class="kpi-val">{inferred}</div><div class="kpi-lbl">Inferred</div></div>
  <div class="kpi-card"><div class="kpi-val">{unique_products}</div><div class="kpi-lbl">Unique SKUs</div></div>
  <div class="kpi-card"><div class="kpi-val">{round(recognized/total*100) if total else 0}%</div><div class="kpi-lbl">ID Confidence</div></div>
</div>

{overview_tag}

{render_legend()}

<h2 class="section">Shelf Layout — All Units</h2>
{unit_sections}

<h2 class="section">Product Summary (Facings by Unit)</h2>
<div style="background:#fff;border-radius:8px;box-shadow:0 2px 12px #0002;
            padding:20px;overflow-x:auto">
  {render_summary_table(summary)}
</div>
</body>
</html>"""


# ── CSV export ────────────────────────────────────────────────────────────────

def write_csv(shelf_rows: dict[str, list[dict]], path: Path) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "unit", "shelf", "shelf_label", "position",
            "product_name", "status", "confidence",
            "loc_x", "loc_y", "loc_z",
            "image_name", "detection_id",
        ])
        for shelf in sorted(shelf_rows, key=shelf_sort_key):
            unit = shelf.split("_")[0]
            label = SHELF_DISPLAY.get(shelf, shelf)
            for pos, p in enumerate(shelf_rows[shelf], 1):
                loc = p.get("location") or [0, 0, 0]
                conf = p.get("propagated_neighbor_confidence") or p.get("identification_source") or "direct"
                w.writerow([
                    unit, shelf, label, pos,
                    p.get("product_name", ""),
                    p.get("product_identity_status", ""),
                    conf,
                    round(float(loc[0]), 4),
                    round(float(loc[1]), 4),
                    round(float(loc[2]), 4),
                    p.get("image_name", ""),
                    p.get("detection_id", ""),
                ])


# ── main ─────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--identified-json", required=True)
    p.add_argument("--renders-dir", required=True)
    p.add_argument("--knowledge-base", default="")
    p.add_argument("--output-html", required=True)
    p.add_argument("--output-csv", required=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    payload = json.loads(Path(args.identified_json).read_text(encoding="utf-8"))
    placements = payload.get("placements") or []
    meta = {k: v for k, v in payload.items() if k != "placements"}

    shelf_rows = build_shelf_rows(placements)
    summary = product_summary(placements)
    renders_dir = Path(args.renders_dir)
    kb_dir = Path(args.knowledge_base) if args.knowledge_base else None

    print("Building product image map…")
    img_map = build_product_image_map(kb_dir, placements)
    print(f"  {len(img_map)} products with images")

    html = build_html(placements, shelf_rows, summary, renders_dir, img_map, meta)
    Path(args.output_html).write_text(html, encoding="utf-8")
    print(f"HTML → {args.output_html}  ({len(html)//1024} KB)")

    write_csv(shelf_rows, Path(args.output_csv))
    total_rows = sum(len(v) for v in shelf_rows.values())
    print(f"CSV  → {args.output_csv}  ({total_rows} rows)")


if __name__ == "__main__":
    main()
