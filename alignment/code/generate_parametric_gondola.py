"""
Parametric 3D Model Generator for Retail Gondola Shelving Fixture.

Directly models physical retail gondola specifications:
1. Shared Vertical Frame: Black powder-coated steel, 4 slotted corner uprights,
   center upright posts, center pegboard divider, black plinth base, and kickplates.
2. Flat Shelf Generator (Side A & Side C - Long Faces):
   6 horizontal flat rectangular shelf boards with price-tag rail strips on front edges.
3. End-Cap Shelf Generator (Side B & Side D - Short End-Cap Faces):
   4-5 uniform flat tiers, same depth/width as each other, facing outward along the
   short axis instead of the long axis.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def create_box_part(
    part_id: str,
    kind: str,
    center: tuple[float, float, float],
    size: tuple[float, float, float],
    color: str = "#18181b",
    opacity: float = 0.98,
) -> dict:
    cx, cy, cz = center
    wx, wy, wz = size
    hx, hy, hz = wx * 0.5, wy * 0.5, wz * 0.5

    vertices = [
        [cx - hx, cy - hy, cz - hz],
        [cx + hx, cy - hy, cz - hz],
        [cx + hx, cy + hy, cz - hz],
        [cx - hx, cy + hy, cz - hz],
        [cx - hx, cy - hy, cz + hz],
        [cx + hx, cy - hy, cz + hz],
        [cx + hx, cy + hy, cz + hz],
        [cx - hx, cy + hy, cz + hz],
    ]
    triangles = [
        [0, 1, 2], [0, 2, 3],  # Bottom
        [4, 6, 5], [4, 7, 6],  # Top
        [0, 4, 5], [0, 5, 1],  # Front
        [1, 5, 6], [1, 6, 2],  # Right
        [2, 6, 7], [2, 7, 3],  # Back
        [3, 7, 4], [3, 4, 0],  # Left
    ]
    return {
        "id": part_id,
        "kind": kind,
        "vertices": vertices,
        "triangles": triangles,
        "style": {"color": color, "opacity": opacity},
    }


def create_shelf_board_mesh(
    shelf_id: str,
    unit: str,
    side_label: str,
    shelf_type: str,  # "flat" (used for every shelf, long faces and end-caps alike)
    center: tuple[float, float, float],
    size: tuple[float, float, float],
    facing: str,
    color: str = "#27272a",
    has_price_tag_rail: bool = False,
) -> tuple[dict, list[dict]]:
    cx, cy, cz = center
    width, depth, thickness = size
    hx, hy, hz = width * 0.5, depth * 0.5, thickness * 0.5

    if facing in {"-Y", "+Y"}:
        u_axis = [1.0, 0.0, 0.0]
        v_axis = [0.0, 1.0, 0.0]
        u_span = [-hx, hx]
        v_span = [-hy, hy]
    else:
        u_axis = [0.0, 1.0, 0.0]
        v_axis = [1.0, 0.0, 0.0]
        u_span = [-hy, hy]
        v_span = [-hx, hx]

    normal = [0.0, 0.0, 1.0]

    box = create_box_part(
        part_id=f"{shelf_id}-board",
        kind="shelf_board",
        center=center,
        size=size,
        color=color,
        opacity=0.98,
    )

    extra_parts = []

    # Price-tag rail strip running along front edge
    if has_price_tag_rail:
        rail_thickness = 0.012
        rail_height = 0.032
        if facing == "-Y":
            rail_center = (cx, cy - hy - rail_thickness * 0.5, cz)
            rail_size = (width, rail_thickness, rail_height)
        elif facing == "+Y":
            rail_center = (cx, cy + hy + rail_thickness * 0.5, cz)
            rail_size = (width, rail_thickness, rail_height)
        elif facing == "+X":
            rail_center = (cx + hx + rail_thickness * 0.5, cy, cz)
            # NOTE: for +/-X facing, `size = (depth, width-across-face, thickness)`,
            # so the rail's span across the shelf face is `depth`, not `width`
            # (that's the shelf's protrusion depth). Using `width` here made the
            # rail sized like the wrong axis entirely.
            rail_size = (rail_thickness, depth, rail_height)
        else:
            rail_center = (cx - hx - rail_thickness * 0.5, cy, cz)
            rail_size = (rail_thickness, depth, rail_height)

        extra_parts.append(
            create_box_part(
                part_id=f"{shelf_id}-price-tag-rail",
                kind="price_tag_rail",
                center=rail_center,
                size=rail_size,
                color="#38bdf8",  # Clear blue-tinted price rail
                opacity=0.90,
            )
        )

    shelf_dict = {
        "id": shelf_id,
        "unit": unit,
        "side_label": side_label,
        "shelf_type": shelf_type,
        "facing": facing,
        "height": cz,
        "centroid": [cx, cy, cz],
        "normal": normal,
        "axes": {"u": u_axis, "v": v_axis},
        "bounds": {"u": u_span, "v": v_span},
        "extents": {"width": width, "depth": depth, "thickness": thickness},
        "mesh": {
            "id": f"{shelf_id}-mesh",
            "type": "shelf_board",
            "vertices": box["vertices"],
            "triangles": box["triangles"],
            "volume": {
                "u": u_span,
                "v": v_span,
                "n": [cz - hz, cz + hz],
            },
        },
        "product_count": 0,
        "has_price_tag_rail": has_price_tag_rail,
    }

    return shelf_dict, extra_parts


def generate_flat_shelves(
    side_label: str,  # "Side A" or "Side C"
    unit: str,        # "Unit1" (Front, -Y) or "Unit3" (Back, +Y)
    facing: str,
    total_width: float,
    base_depth: float,
    count: int = 6,
    height_range: tuple[float, float] = (0.35, 1.65),
    shelf_thickness: float = 0.04,
    center_y: float = 0.0,
    has_price_tag_rail: bool = True,
    shelf_heights: list[float] | None = None,
) -> tuple[list[dict], list[dict]]:
    """Flat gondola shelving generator for long faces (Side A & C).

    Each shelf level is a single continuous board spanning `total_width`.
    There are no internal bay-boundary uprights for it to terminate against
    (a double-sided run supports internal seams via the corner uprights'
    slotted rails, not a post at every module), so one unbroken board per
    level is both simpler and structurally accurate.

    `shelf_heights` puts the boards at explicit Z values (meters above the
    floor) instead of spreading `count` of them evenly over `height_range`.
    Pass the scan's detected shelf bands here whenever they are available: real
    fixtures are not evenly spaced (the bottom shelf is usually deeper set and
    the top gap larger), so even spacing leaves every board a few centimeters
    off its true plane -- enough that products positioned from the same scan
    float above or sink through the shelf they are actually resting on.
    """
    shelves = []
    extra_parts = []
    if shelf_heights:
        heights = np.array(sorted(float(z) for z in shelf_heights), dtype=float)
    else:
        if count <= 0:
            return shelves, extra_parts
        z_start, z_end = height_range
        heights = np.linspace(z_start, z_end, count)

    for idx, z in enumerate(heights, 1):
        shelf_id = f"{unit}_Shelf_{idx:02d}"
        shelf, rails = create_shelf_board_mesh(
            shelf_id=shelf_id,
            unit=unit,
            side_label=side_label,
            shelf_type="flat",
            center=(0.0, center_y, float(z)),
            size=(total_width, base_depth, shelf_thickness),
            facing=facing,
            color="#27272a",
            has_price_tag_rail=has_price_tag_rail,
        )
        shelves.append(shelf)
        extra_parts.extend(rails)

    return shelves, extra_parts


def generate_endcap_shelves(
    side_label: str,  # "Side B" or "Side D"
    unit: str,        # "Unit2" (Right, +X) or "Unit4" (Left, -X)
    facing: str,
    endcap_face_width: float,
    tier_depth: float = 0.35,
    count: int = 5,
    height_range: tuple[float, float] = (0.30, 1.60),
    shelf_thickness: float = 0.04,
    frame_edge_x: float = 0.0,
    has_price_tag_rail: bool = True,
    shelf_heights: list[float] | None = None,
) -> tuple[list[dict], list[dict]]:
    """Flat shelving generator for short end-cap faces (Side B & D).

    Every tier is anchored flush against `frame_edge_x` (the main frame's
    outer face, where the corner uprights sit) and shares the same depth and
    width as every other tier -- uniform shelves, same as Side A/C, just
    facing outward along X instead of Y. `endcap_face_width` should match
    the opaque end-panel behind the shelves (== gondola_depth) so the
    shelves and their price rails span exactly as wide as the panel backing
    them, instead of an unrelated, independently-guessed width.
    """
    shelves = []
    extra_parts = []
    if shelf_heights:
        # Tier 01 is the top shelf on the end-caps, so descend like linspace does.
        heights = np.array(sorted((float(z) for z in shelf_heights), reverse=True), dtype=float)
    else:
        if count <= 0:
            return shelves, extra_parts
        z_start, z_end = height_range
        heights = np.linspace(z_end, z_start, count)
    sign = 1.0 if facing == "+X" else -1.0
    cx = frame_edge_x + sign * (tier_depth * 0.5)

    for tier_idx, z in enumerate(heights, 0):
        shelf_id = f"{unit}_EndcapShelf_{tier_idx+1:02d}"
        shelf, rails = create_shelf_board_mesh(
            shelf_id=shelf_id,
            unit=unit,
            side_label=side_label,
            shelf_type="flat",
            center=(cx, 0.0, float(z)),
            size=(tier_depth, endcap_face_width, shelf_thickness),
            facing=facing,
            color="#18181b",
            has_price_tag_rail=has_price_tag_rail,
        )
        shelf["tier_index"] = tier_idx
        shelves.append(shelf)
        extra_parts.extend(rails)

    return shelves, extra_parts


def generate_parametric_gondola_fixture(
    bay_width: float = 1.2,
    num_bays: int = 1,
    gondola_depth: float = 0.8,
    gondola_height: float = 1.8,
    plinth_height: float = 0.15,
    shelf_thickness: float = 0.04,
    side_a_shelves: int = 6,      # Side A (Front, Unit1): 6 flat shelves
    side_c_shelves: int = 6,      # Side C (Back, Unit3): 6 flat shelves
    side_b_shelves: int = 5,      # Side B (Right End-cap, Unit2): 5 flat shelves
    side_d_shelves: int = 5,      # Side D (Left End-cap, Unit4): 5 flat shelves
    endcap_shelf_depth: float = 0.35,  # Uniform depth of every end-cap shelf (m)
    # Measured shelf Z values (m above the floor), per side. When given, these
    # replace the even spacing derived from the side's shelf *count* -- pass the
    # scan's detected shelf bands so the boards land on their real planes.
    side_a_shelf_heights: list[float] | None = None,
    side_c_shelf_heights: list[float] | None = None,
    side_b_shelf_heights: list[float] | None = None,
    side_d_shelf_heights: list[float] | None = None,
    has_price_tag_rails: bool = True,
    has_top_canopy: bool = False,  # Solid cap plate over the whole top -- off by
                                    # default: real open-top gondolas photographed
                                    # for this project show no such panel (top
                                    # shelf sits open to the ceiling in every
                                    # reference photo). Only enable if your
                                    # source photos actually show one.
    color_theme: str = "black",
) -> dict:

    # Black powder-coated steel finishes. The uprights get a visibly lighter
    # tone than the spine/plinth on purpose: with everything near-black,
    # the individual rods have no contrast against the spine panel behind
    # them and disappear into one flat silhouette in render.
    frame_color = "#18181b"       # Black powder-coated steel
    upright_color = "#6b7076"     # Slotted upright posts (lighter so they read as separate rods)
    plinth_color = "#090d16"      # Deep black plinth base
    spine_color = "#18181b"       # Center pegboard divider

    num_bays = max(1, int(num_bays))
    total_width = bay_width * num_bays

    depth_half = gondola_depth * 0.5
    width_half = total_width * 0.5
    # Flat shelves sit at 0.95 * depth_half from center (see generate_flat_shelves:
    # base_depth = depth_half * 0.9, center offset = depth_half * 0.5), so their
    # outer edge is at depth_half * 0.95, not depth_half. Uprights placed at the
    # raw depth_half stick out past that edge and read as rods floating in front
    # of the merchandise face instead of a frame member behind/flush with it.
    shelf_edge_depth = depth_half * 0.95

    # 1. Shared Vertical Frame Architecture
    frame_parts = [
        # Center Pegboard Spine Divider (divides Side A and C back-to-back)
        create_box_part(
            part_id="frame-center-pegboard-spine",
            kind="back_panel",
            center=(0.0, 0.0, plinth_height + (gondola_height - plinth_height) * 0.5),
            size=(total_width, 0.04, gondola_height - plinth_height),
            color=spine_color,
        ),
        # Center Upright Support Posts
        create_box_part(
            part_id="frame-center-upright-left",
            kind="upright_post",
            center=(-width_half, 0.0, gondola_height * 0.5),
            size=(0.05, 0.06, gondola_height),
            color=upright_color,
        ),
        create_box_part(
            part_id="frame-center-upright-right",
            kind="upright_post",
            center=(width_half, 0.0, gondola_height * 0.5),
            size=(0.05, 0.06, gondola_height),
            color=upright_color,
        ),
        # 4 Slotted Corner Uprights -- true physical ends of the run, flush
        # with the shelf edge rather than proud of it.
        create_box_part(
            part_id="frame-corner-upright-fl",
            kind="upright_post",
            center=(-width_half, -shelf_edge_depth, gondola_height * 0.5),
            size=(0.045, 0.045, gondola_height),
            color=upright_color,
        ),
        create_box_part(
            part_id="frame-corner-upright-fr",
            kind="upright_post",
            center=(width_half, -shelf_edge_depth, gondola_height * 0.5),
            size=(0.045, 0.045, gondola_height),
            color=upright_color,
        ),
        create_box_part(
            part_id="frame-corner-upright-bl",
            kind="upright_post",
            center=(-width_half, shelf_edge_depth, gondola_height * 0.5),
            size=(0.045, 0.045, gondola_height),
            color=upright_color,
        ),
        create_box_part(
            part_id="frame-corner-upright-br",
            kind="upright_post",
            center=(width_half, shelf_edge_depth, gondola_height * 0.5),
            size=(0.045, 0.045, gondola_height),
            color=upright_color,
        ),
        # Freestanding 4-Sided Base Plinth & Kickplate. Extends past the main
        # frame by the end-cap shelves' actual protrusion depth (plus a small
        # overhang margin), not an unrelated width parameter -- otherwise the
        # base reads as its own oddly-proportioned slab instead of a plinth
        # sized to what's actually sitting on it.
        create_box_part(
            part_id="frame-base-plinth",
            kind="plinth",
            center=(0.0, 0.0, plinth_height * 0.5),
            size=(total_width + (endcap_shelf_depth + 0.06) * 2.0, gondola_depth + 0.12, plinth_height),
            color=plinth_color,
        ),
        # Opaque pegboard end-panels: without these, the ends of the run are
        # open frame -- looking at an end-cap from an angle, you see straight
        # through the interior to the opposite face's shelves poking through.
        # These close off each end the same way the center spine closes off
        # the middle, sized to the frame's depth/height, sitting right at the
        # end-cap plane.
        create_box_part(
            part_id="frame-end-panel-left",
            kind="back_panel",
            center=(-width_half, 0.0, plinth_height + (gondola_height - plinth_height) * 0.5),
            size=(0.04, gondola_depth, gondola_height - plinth_height),
            color=spine_color,
        ),
        create_box_part(
            part_id="frame-end-panel-right",
            kind="back_panel",
            center=(width_half, 0.0, plinth_height + (gondola_height - plinth_height) * 0.5),
            size=(0.04, gondola_depth, gondola_height - plinth_height),
            color=spine_color,
        ),
    ]

    if has_top_canopy:
        frame_parts.append(
            create_box_part(
                part_id="frame-top-canopy-cap",
                kind="top_cap",
                center=(0.0, 0.0, gondola_height - 0.025),
                size=(total_width + (endcap_shelf_depth + 0.06) * 2.0, gondola_depth, 0.05),
                color=frame_color,
            )
        )

    # No posts at internal bay boundaries: this is a double-sided unit, so
    # "the back" of one bay seam is the customer-facing side of Side C --
    # there's no face to put a full-height post on that isn't someone's
    # merchandise front. Real double-sided runs support internal seams via
    # the corner uprights' slotted rails and the continuous shelf boards, not
    # a separate post at every module. Only the true ends of the run (the
    # corner uprights above) show a visible post.

    all_shelves = []
    all_extra_parts = []

    # 2. Side A (Front Long Face, Unit1): Flat Shelves + Price-Tag Rails
    shelves_a, parts_a = generate_flat_shelves(
        side_label="Side A",
        unit="Unit1",
        facing="-Y",
        total_width=total_width,
        base_depth=depth_half * 0.9,
        count=side_a_shelves,
        height_range=(plinth_height + 0.18, gondola_height - 0.18),
        shelf_thickness=shelf_thickness,
        center_y=-depth_half * 0.5,
        has_price_tag_rail=has_price_tag_rails,
        shelf_heights=side_a_shelf_heights,
    )
    all_shelves.extend(shelves_a)
    all_extra_parts.extend(parts_a)

    # 3. Side C (Back Long Face, Unit3): Flat Shelves + Price-Tag Rails
    shelves_c, parts_c = generate_flat_shelves(
        side_label="Side C",
        unit="Unit3",
        facing="+Y",
        total_width=total_width,
        base_depth=depth_half * 0.9,
        count=side_c_shelves,
        height_range=(plinth_height + 0.18, gondola_height - 0.18),
        shelf_thickness=shelf_thickness,
        center_y=depth_half * 0.5,
        has_price_tag_rail=has_price_tag_rails,
        shelf_heights=side_c_shelf_heights,
    )
    all_shelves.extend(shelves_c)
    all_extra_parts.extend(parts_c)

    # 4. Side B (Right Short End-Cap Face, Unit2): Flat Shelves + Price-Tag Rails.
    # endcap_face_width == gondola_depth so the shelves (and their rails) span
    # exactly as wide as the opaque end-panel/pegboard sitting behind them.
    shelves_b, parts_b = generate_endcap_shelves(
        side_label="Side B",
        unit="Unit2",
        facing="+X",
        endcap_face_width=gondola_depth,
        tier_depth=endcap_shelf_depth,
        count=side_b_shelves,
        height_range=(plinth_height + 0.16, gondola_height - 0.22),
        shelf_thickness=shelf_thickness,
        frame_edge_x=width_half,
        has_price_tag_rail=has_price_tag_rails,
        shelf_heights=side_b_shelf_heights,
    )
    all_shelves.extend(shelves_b)
    all_extra_parts.extend(parts_b)

    # 5. Side D (Left Short End-Cap Face, Unit4): Flat Shelves + Price-Tag Rails
    shelves_d, parts_d = generate_endcap_shelves(
        side_label="Side D",
        unit="Unit4",
        facing="-X",
        endcap_face_width=gondola_depth,
        tier_depth=endcap_shelf_depth,
        count=side_d_shelves,
        height_range=(plinth_height + 0.16, gondola_height - 0.22),
        shelf_thickness=shelf_thickness,
        frame_edge_x=-width_half,
        has_price_tag_rail=has_price_tag_rails,
        shelf_heights=side_d_shelf_heights,
    )
    all_shelves.extend(shelves_d)
    all_extra_parts.extend(parts_d)

    return {
        "job_id": "freestanding-gondola-spec",
        "rack": {
            "id": "rack-black-steel-freestanding",
            "kind": "gondola",
            "material": "black_powder_coated_steel",
            "parts": frame_parts + all_extra_parts,
        },
        "shelves": all_shelves,
        "products": [],
        "layout": {
            "fixture_template": "gondola",
            "bay_width_m": bay_width,
            "num_bays": num_bays,
            "total_bay_run_m": total_width,
            "gondola_depth_m": gondola_depth,
            "gondola_height_m": gondola_height,
            "side_a_type": "flat",
            "side_c_type": "flat",
            "side_b_type": "flat",
            "side_d_type": "flat",
            "side_a_shelves": len(shelves_a),
            "side_c_shelves": len(shelves_c),
            "side_b_shelves": len(shelves_b),
            "side_d_shelves": len(shelves_d),
            "shelf_heights_source": (
                "measured" if any(
                    (side_a_shelf_heights, side_c_shelf_heights,
                     side_b_shelf_heights, side_d_shelf_heights)
                ) else "even_spacing"
            ),
            "endcap_shelf_depth_m": endcap_shelf_depth,
            "has_price_tag_rails": has_price_tag_rails,
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Construct a parametric 3D model of a retail gondola shelving fixture from physical specs."
    )
    parser.add_argument("--output", default="parametric_gondola_results.json", help="Output JSON path")
    parser.add_argument("--bay-width", type=float, default=1.2, help="Width of a single bay module (m)")
    parser.add_argument("--num-bays", type=int, default=1, help="Number of bay modules run end to end (default: 1)")
    parser.add_argument("--depth", type=float, default=0.8, help="Gondola total depth (m)")
    parser.add_argument("--height", type=float, default=1.8, help="Gondola height (m)")
    parser.add_argument("--side-a-shelves", type=int, default=6, help="Side A flat shelf count (default: 6)")
    parser.add_argument("--side-c-shelves", type=int, default=6, help="Side C flat shelf count (default: 6)")
    parser.add_argument("--side-b-shelves", type=int, default=5, help="Side B end-cap shelf count (default: 5)")
    parser.add_argument("--side-d-shelves", type=int, default=5, help="Side D end-cap shelf count (default: 5)")
    parser.add_argument("--endcap-shelf-depth", type=float, default=0.35, help="Uniform depth of every end-cap shelf (m)")
    parser.add_argument("--no-price-rails", action="store_true", help="Disable price-tag rail strips")
    parser.add_argument("--top-canopy", action="store_true",
                         help="Add a solid cap plate over the whole top (default: off -- "
                              "only enable if your reference photos actually show one)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model = generate_parametric_gondola_fixture(
        bay_width=args.bay_width,
        num_bays=args.num_bays,
        gondola_depth=args.depth,
        gondola_height=args.height,
        side_a_shelves=args.side_a_shelves,
        side_c_shelves=args.side_c_shelves,
        side_b_shelves=args.side_b_shelves,
        side_d_shelves=args.side_d_shelves,
        endcap_shelf_depth=args.endcap_shelf_depth,
        has_price_tag_rails=not args.no_price_rails,
        has_top_canopy=args.top_canopy,
    )
    output_path = Path(args.output)
    output_path.write_text(json.dumps(model, indent=2), encoding="utf-8")
    print(f"PARAMETRIC_GONDOLA_SPEC_GENERATED: {output_path}")
    print(f"  Bay run: {args.num_bays} x {args.bay_width}m = {model['layout']['total_bay_run_m']:.3f}m total")
    print(f"  Total Shelves: {len(model['shelves'])} (Side A: {args.side_a_shelves} flat, Side C: {args.side_c_shelves} flat, Side B: {args.side_b_shelves} flat, Side D: {args.side_d_shelves} flat)")
    print(f"  Total Frame & Rail Parts: {len(model['rack']['parts'])}")


if __name__ == "__main__":
    main()
