"""
Render an aligned planogram (output of align_products_to_gondola.py) as an SVG
elevation -- one panel per side, scale-true, products drawn where they landed.

SVG rather than a raster: it needs no dependencies (the file is just a string),
it can carry real text labels and per-product tooltips, and it stays sharp when
you zoom in on a 4 mm shelf board. Open it in a browser or VS Code.

Reading it: products should sit in tidy rows a few centimeters above their shelf
lines, spread across the bay run. Rows that sag between two boards, products
bunched at one end, or a side that is suspiciously empty are all alignment
failures that a residual number alone will not show you.

    python3 render_planogram.py --aligned /tmp/demo_aligned.json --output /tmp/planogram.svg

VS Code opens .svg as XML text, not as a picture -- use "Open With... > Image
Preview", or a browser. For a PNG instead (no extra packages needed; read the
width/height off the <svg> tag):

    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" \\
      --headless --disable-gpu --hide-scrollbars --force-device-scale-factor=2 \\
      --window-size=940,1126 --screenshot=planogram.png file:///abs/path/planogram.svg

Colors come from the project data-viz palette. The mark pair was validated for
colour-vision deficiency: the semantically obvious green/red ("seated" = good,
"unseated" = critical) fails deuteranope separation at OKLab dE 4.1, so seated
products take the categorical blue and only the exceptions carry a status hue
(dE 23.8 light / 25.7 dark, both clear of the >= 8 target).
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from html import escape
from pathlib import Path

# Palette roles. Light and dark are both selected steps, not an auto-flip.
LIGHT = {
    "surface": "#fcfcfb", "ink": "#0b0b0b", "secondary": "#52514e", "muted": "#898781",
    "grid": "#e1e0d9", "baseline": "#c3c2b7", "board": "#898781",
    "seated": "#2a78d6", "unseated": "#d03b3b",
}
DARK = {
    "surface": "#1a1a19", "ink": "#ffffff", "secondary": "#c3c2b7", "muted": "#898781",
    "grid": "#2c2c2a", "baseline": "#383835", "board": "#898781",
    "seated": "#3987e5", "unseated": "#d03b3b",
}

PANEL_WIDTH = 940
PANEL_HEIGHT = 250
MARGIN_LEFT = 78
MARGIN_RIGHT = 26
MARGIN_TOP = 34
MARGIN_BOTTOM = 40


def theme_css() -> str:
    """Both modes declared under their own scopes, so a viewer toggle wins."""
    light = "\n".join(f"      --{role}: {value};" for role, value in LIGHT.items())
    dark = "\n".join(f"      --{role}: {value};" for role, value in DARK.items())
    return f"""  <style>
    svg {{
      color-scheme: light;
{light}
      font-family: system-ui, -apple-system, "Segoe UI", sans-serif;
    }}
    @media (prefers-color-scheme: dark) {{
      svg:where(:not([data-theme="light"])) {{
        color-scheme: dark;
{dark}
      }}
    }}
    svg[data-theme="dark"] {{
      color-scheme: dark;
{dark}
    }}
    .surface {{ fill: var(--surface); }}
    .title {{ fill: var(--ink); font-size: 15px; font-weight: 600; }}
    .caption {{ fill: var(--secondary); font-size: 11.5px; }}
    .axis {{ fill: var(--muted); font-size: 10.5px; font-variant-numeric: tabular-nums; }}
    .panel-label {{ fill: var(--ink); font-size: 12.5px; font-weight: 600; }}
    .legend {{ fill: var(--secondary); font-size: 11.5px; }}
    .grid {{ stroke: var(--grid); stroke-width: 1; }}
    .outline {{ stroke: var(--baseline); stroke-width: 2; fill: none; }}
    /* fill:none matters -- an SVG rect with no fill declared paints solid black. */
    .envelope {{ stroke: var(--board); stroke-width: 2; stroke-dasharray: 5 4; fill: none; }}
    .board {{ stroke: var(--board); stroke-width: 2; stroke-linecap: round; }}
    /* 2px surface ring: facings overlap along a shelf, and without it a dense
       row reads as one blob instead of countable products. */
    .mark {{ stroke: var(--surface); stroke-width: 2; }}
    .seated {{ fill: var(--seated); }}
    .unseated {{ fill: var(--unseated); }}
    .mark:hover {{ stroke: var(--ink); }}
  </style>"""


def panel(
    label: str,
    caption: str,
    shelves: list[dict],
    products: list[dict],
    *,
    offset_y: float,
    half_span: float,
    fixture_height: float,
    axis_label: str,
    envelope: tuple[float, float] | None = None,
) -> list[str]:
    """One side's elevation, drawn to scale in meters."""
    plot_w = PANEL_WIDTH - MARGIN_LEFT - MARGIN_RIGHT
    plot_h = PANEL_HEIGHT - MARGIN_TOP - MARGIN_BOTTOM
    x0, y0 = MARGIN_LEFT, offset_y + MARGIN_TOP

    def to_x(meters: float) -> float:
        return x0 + (meters + half_span) / (2 * half_span) * plot_w

    def to_y(meters: float) -> float:
        return y0 + plot_h - (meters / fixture_height) * plot_h

    out = [f'  <g class="panel">']
    out.append(f'    <text class="panel-label" x="{x0}" y="{offset_y + 20:.0f}">{escape(label)}</text>')
    out.append(f'    <text class="caption" x="{x0 + 132}" y="{offset_y + 20:.0f}">{escape(caption)}</text>')

    # Plot frame. On a side panel the frame IS the fixture envelope; when the
    # axes are wider than the fixture (the unplaced panel), the envelope is
    # drawn separately so "outside the fixture" stays literally visible instead
    # of being implied by a frame that is really just the data extent.
    out.append(f'    <rect class="outline" x="{to_x(-half_span):.1f}" y="{to_y(fixture_height):.1f}" '
               f'width="{plot_w:.1f}" height="{plot_h:.1f}" rx="2"/>')
    if envelope is not None:
        env_half, env_height = envelope
        out.append(f'    <rect class="envelope" x="{to_x(-env_half):.1f}" '
                   f'y="{to_y(env_height):.1f}" '
                   f'width="{to_x(env_half) - to_x(-env_half):.1f}" '
                   f'height="{to_y(0) - to_y(env_height):.1f}" rx="2">'
                   f'<title>fixture envelope — {2 * env_half:.2f} m run, '
                   f'{env_height:.2f} m high</title></rect>')

    # Vertical metre grid, recessive.
    tick = 1.0 if half_span > 1.5 else 0.25
    value = -half_span - (-half_span % tick)
    while value <= half_span:
        if abs(value) <= half_span:
            out.append(f'    <line class="grid" x1="{to_x(value):.1f}" y1="{to_y(0):.1f}" '
                       f'x2="{to_x(value):.1f}" y2="{to_y(fixture_height):.1f}"/>')
            out.append(f'    <text class="axis" x="{to_x(value):.1f}" y="{to_y(0) + 15:.1f}" '
                       f'text-anchor="middle">{value:+.2f}</text>')
        value += tick
    out.append(f'    <text class="axis" x="{x0 + plot_w / 2:.0f}" y="{to_y(0) + 30:.1f}" '
               f'text-anchor="middle">{escape(axis_label)}</text>')

    # Shelf boards at their real heights, spanning their real footprint.
    by_shelf: dict[str, list[dict]] = defaultdict(list)
    for product in products:
        if product.get("shelf_id"):
            by_shelf[product["shelf_id"]].append(product)
    # Every board is drawn, but heights are labelled selectively: real shelf
    # bands come as close as 11 cm, which at this scale overlaps the type. The
    # tooltip on each board carries the exact height regardless.
    labelled_y = [to_y(0)]
    for shelf in sorted(shelves, key=lambda s: float(s["height"])):
        bounds = shelf["bounds"]["u"]
        height = float(shelf["height"])
        y = to_y(height)
        out.append(f'    <line class="board" x1="{to_x(bounds[0]):.1f}" y1="{y:.1f}" '
                   f'x2="{to_x(bounds[1]):.1f}" y2="{y:.1f}">'
                   f'<title>{escape(shelf["id"])} — z {height:.3f} m, '
                   f'{len(by_shelf.get(shelf["id"], []))} facings</title></line>')
        if all(abs(y - other) >= 13 for other in labelled_y):
            labelled_y.append(y)
            out.append(f'    <text class="axis" x="{x0 - 10}" y="{y + 3.5:.1f}" '
                       f'text-anchor="end">{height:.3f}</text>')
    if not shelves:
        # No boards to hang heights off, so fall back to a regular scale --
        # otherwise the only label is the floor and the panel is unreadable.
        step = 0.5
        value = step
        while value < fixture_height:
            out.append(f'    <line class="grid" x1="{to_x(-half_span):.1f}" y1="{to_y(value):.1f}" '
                       f'x2="{to_x(half_span):.1f}" y2="{to_y(value):.1f}"/>')
            out.append(f'    <text class="axis" x="{x0 - 10}" y="{to_y(value) + 3.5:.1f}" '
                       f'text-anchor="end">{value:.3f}</text>')
            value += step
    out.append(f'    <text class="axis" x="{x0 - 10}" y="{to_y(0) + 3.5:.1f}" '
               f'text-anchor="end">0.000</text>')
    out.append(f'    <text class="axis" x="{x0 - 10}" y="{y0 - 6:.1f}" '
               f'text-anchor="end">height (m)</text>')

    # Products. 9px marks, above their board by their real clearance.
    for product in products:
        local = product.get("shelf_local") or {}
        z = product["position"][2]
        cx, cy = to_x(local.get("u", 0.0)), to_y(z)
        seated = bool(product.get("seated"))
        tooltip = (f'{product.get("product_name") or "?"} — {product.get("shelf_id") or "no shelf"}'
                   f' — z {z:.3f} m, clearance {(product.get("clearance_m") or 0) * 100:.1f} cm')
        if seated:
            out.append(f'    <circle class="mark seated" cx="{cx:.1f}" cy="{cy:.1f}" r="4.5">'
                       f'<title>{escape(tooltip)}</title></circle>')
        else:
            # Diamond, not just a different colour: status hue never carries
            # meaning on its own.
            out.append(f'    <path class="mark unseated" d="M {cx:.1f} {cy - 5.5:.1f} '
                       f'L {cx + 5.5:.1f} {cy:.1f} L {cx:.1f} {cy + 5.5:.1f} '
                       f'L {cx - 5.5:.1f} {cy:.1f} Z">'
                       f'<title>{escape(tooltip)}</title></path>')
    out.append("  </g>")
    return out


def render(aligned: dict) -> str:
    products = aligned.get("products", [])
    shelves = aligned.get("shelves", [])
    layout = aligned.get("layout", {})
    alignment = aligned.get("product_alignment", {})
    quality = alignment.get("quality", {})
    fit = alignment.get("shelf_height_fit", {})
    transform = alignment.get("transform", {})

    fixture_height = float(layout.get("gondola_height_m") or 1.8)
    half_run = 0.5 * float(layout.get("total_bay_run_m") or 2.0)
    half_depth = 0.5 * float(layout.get("gondola_depth_m") or 1.0)

    sides = []
    for side in ("Side A", "Side C", "Side B", "Side D"):
        side_shelves = [s for s in shelves if s.get("side_label") == side]
        seated_here = [p for p in products if p.get("side_label") == side]
        if side_shelves and seated_here:
            sides.append((side, side_shelves, seated_here))
    unseated = [p for p in products if not p.get("seated")]

    panels = len(sides) + (1 if unseated else 0)
    height = MARGIN_TOP + 62 + panels * PANEL_HEIGHT + 30

    body: list[str] = []
    body.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{PANEL_WIDTH}" height="{height}" '
                f'viewBox="0 0 {PANEL_WIDTH} {height}" role="img" '
                f'aria-label="Aligned planogram elevation">')
    body.append(theme_css())
    body.append(f'  <rect class="surface" x="0" y="0" width="{PANEL_WIDTH}" height="{height}"/>')
    body.append(f'  <text class="title" x="{MARGIN_LEFT}" y="30">'
                f'Products aligned onto the fitted gondola</text>')
    body.append(f'  <text class="caption" x="{MARGIN_LEFT}" y="48">'
                f'{layout.get("total_bay_run_m", 0):.2f} m run · '
                f'{layout.get("gondola_depth_m", 0):.2f} m deep · '
                f'{fixture_height:.2f} m high · shelf heights '
                f'{escape(str(layout.get("shelf_heights_source", "?")))} · '
                f'{transform.get("scale_meters_per_colmap_unit", 0):.4f} m per COLMAP unit '
                f'(shelf-match RMS {fit.get("rms_m", 0) * 100:.1f} cm)</text>')

    # Legend: two marks, always present, each with its own label.
    legend_y = 66
    body.append(f'  <circle class="mark seated" cx="{MARGIN_LEFT + 6}" cy="{legend_y - 4}" r="4.5"/>')
    body.append(f'  <text class="legend" x="{MARGIN_LEFT + 18}" y="{legend_y}">'
                f'seated on a board ({quality.get("snapped_to_a_shelf", 0)})</text>')
    body.append(f'  <path class="mark unseated" d="M {MARGIN_LEFT + 190} {legend_y - 9.5} '
                f'L {MARGIN_LEFT + 195.5} {legend_y - 4} L {MARGIN_LEFT + 190} {legend_y + 1.5} '
                f'L {MARGIN_LEFT + 184.5} {legend_y - 4} Z"/>')
    body.append(f'  <text class="legend" x="{MARGIN_LEFT + 202}" y="{legend_y}">'
                f'not on any board ({len(unseated)})</text>')

    offset = MARGIN_TOP + 62
    for side, side_shelves, seated_here in sides:
        endcap = side in ("Side B", "Side D")
        clearances = [p["clearance_m"] for p in seated_here if p.get("clearance_m") is not None]
        caption = (f'{len(seated_here)} facings on {len(side_shelves)} shelves'
                   + (f' · median clearance {sorted(clearances)[len(clearances) // 2] * 100:.1f} cm'
                      if clearances else ''))
        body.extend(panel(
            side, caption, side_shelves, seated_here,
            offset_y=offset,
            half_span=half_depth if endcap else half_run,
            fixture_height=fixture_height,
            axis_label="across the end-cap (m)" if endcap else "along the bay run (m)",
        ))
        offset += PANEL_HEIGHT

    if unseated:
        # Positioned by model X/Z, because they have no shelf to be local to --
        # that is the point: they fell outside every board.
        for product in unseated:
            product["shelf_local"] = {"u": product["position"][0]}
        span = max(half_run, max(abs(p["position"][0]) for p in unseated)) * 1.02
        top = max(fixture_height, max(p["position"][2] for p in unseated)) * 1.02
        body.extend(panel(
            "Unplaced", f"{len(unseated)} products outside every board (model X vs height)",
            [], unseated,
            offset_y=offset, half_span=span, fixture_height=top,
            axis_label="model X (m) — dashed box is the fixture envelope",
            envelope=(half_run, fixture_height),
        ))

    body.append("</svg>")
    return "\n".join(body)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--aligned", required=True, help="Output of align_products_to_gondola.py")
    parser.add_argument("--output", required=True, help="SVG path to write")
    args = parser.parse_args()

    aligned = json.loads(Path(args.aligned).read_text(encoding="utf-8"))
    svg = render(aligned)
    Path(args.output).write_text(svg, encoding="utf-8")
    print(f"PLANOGRAM_WRITTEN: {args.output} ({len(svg) / 1024:.1f} kB)")


if __name__ == "__main__":
    main()
