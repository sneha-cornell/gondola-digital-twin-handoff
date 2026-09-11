"""
Fill product gaps and empty shelves in the identified layout.

Works on the output of blender_apply_identified_slots.py:
  - Shelves with >= 2 products: fill gaps between detected products
  - Shelves with 0-1 products: fill entire shelf span with blank placeholders
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import bpy
from mathutils import Vector


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--identified-json", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--collection", default="Blank_Filled_Slots")
    parser.add_argument("--product-height", type=float, default=0.22)
    parser.add_argument("--product-width", type=float, default=0.14)
    parser.add_argument("--product-depth", type=float, default=0.055)
    parser.add_argument("--min-gap-multiplier", type=float, default=1.5)
    parser.add_argument("--replace", action="store_true")
    return parser.parse_args(sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else [])


def world_bounds(obj: bpy.types.Object):
    corners = [obj.matrix_world @ Vector(corner) for corner in obj.bound_box]
    xs = [corner.x for corner in corners]
    ys = [corner.y for corner in corners]
    zs = [corner.z for corner in corners]
    return (min(xs), max(xs)), (min(ys), max(ys)), (min(zs), max(zs))


def collection_for(name: str, replace: bool) -> bpy.types.Collection:
    existing = bpy.data.collections.get(name)
    if existing and replace:
        for obj in list(existing.objects):
            bpy.data.objects.remove(obj, do_unlink=True)
        bpy.data.collections.remove(existing)
        existing = None
    if existing:
        return existing
    collection = bpy.data.collections.new(name)
    bpy.context.scene.collection.children.link(collection)
    return collection


def link_to_collection(obj: bpy.types.Object, collection: bpy.types.Collection) -> None:
    for coll in list(obj.users_collection):
        coll.objects.unlink(obj)
    collection.objects.link(obj)


def make_blank_material() -> bpy.types.Material:
    mat = bpy.data.materials.get("InferredBlankSlot") or bpy.data.materials.new("InferredBlankSlot")
    mat.use_nodes = True
    mat.blend_method = "OPAQUE"
    mat.use_backface_culling = False
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()
    output = nodes.new("ShaderNodeOutputMaterial")
    bsdf = nodes.new("ShaderNodeBsdfPrincipled")
    bsdf.inputs["Base Color"].default_value = (0.78, 0.82, 0.78, 1.0)
    bsdf.inputs["Alpha"].default_value = 1.0
    output.location = (300, 0)
    bsdf.location = (0, 0)
    links.new(bsdf.outputs["BSDF"], output.inputs["Surface"])
    return mat


def unit_for(shelf_name: str) -> str:
    return shelf_name.split("_", 1)[0]


def horizontal_axis(shelf_name: str) -> str:
    return "x" if unit_for(shelf_name) in {"Unit1", "Unit3"} else "y"


def facing_for(shelf_name: str) -> str:
    unit = unit_for(shelf_name)
    if unit == "Unit1":
        return "-Y"
    if unit == "Unit3":
        return "+Y"
    if unit == "Unit2":
        return "+X"
    return "-X"


def placement_location(placement: dict) -> Vector:
    x, y, z = placement["location"]
    return Vector((float(x), float(y), float(z)))


def estimate_spacing(coords: list[float], default: float = 0.15) -> float:
    if len(coords) < 2:
        return default
    deltas = [b - a for a, b in zip(sorted(coords), sorted(coords)[1:]) if 0.045 <= b - a <= 0.34]
    if not deltas:
        return default
    return max(0.08, min(0.22, statistics.median(deltas)))


def gap_positions(coords: list[float], spacing: float, min_gap_multiplier: float) -> list[float]:
    slots: list[float] = []
    for left, right in zip(sorted(coords), sorted(coords)[1:]):
        gap = right - left
        if gap < spacing * min_gap_multiplier:
            continue
        missing = max(0, round(gap / spacing) - 1)
        if missing <= 0:
            continue
        step = gap / float(missing + 1)
        for i in range(1, missing + 1):
            slots.append(left + step * i)
    return slots


def full_shelf_positions(span_min: float, span_max: float, spacing: float) -> list[float]:
    """Evenly fill the entire shelf span."""
    count = max(1, round((span_max - span_min) / spacing))
    step = (span_max - span_min) / count
    return [span_min + step * (i + 0.5) for i in range(count)]


def create_blank_box(
    name: str,
    shelf_name: str,
    coord: float,
    depth_coord: float,
    shelf_top_z: float,
    collection: bpy.types.Collection,
    material: bpy.types.Material,
    product_width: float,
    product_depth: float,
    product_height: float,
) -> bpy.types.Object:
    facing = facing_for(shelf_name)
    axis = horizontal_axis(shelf_name)
    base = Vector((coord, depth_coord, shelf_top_z)) if axis == "x" else Vector((depth_coord, coord, shelf_top_z))

    if facing in {"-Y", "+Y"}:
        dims = (product_width, product_depth, product_height)
        offset = Vector((0, -product_depth * 0.5 if facing == "-Y" else product_depth * 0.5, product_height * 0.5 + 0.012))
    else:
        dims = (product_depth, product_width, product_height)
        offset = Vector((product_depth * 0.5 if facing == "+X" else -product_depth * 0.5, 0, product_height * 0.5 + 0.012))

    bpy.ops.mesh.primitive_cube_add(size=1.0, location=base + offset)
    obj = bpy.context.object
    obj.name = name
    obj.dimensions = dims
    bpy.context.view_layer.update()

    # Add UV map so textures can be applied later
    if obj.data.uv_layers:
        uv = obj.data.uv_layers.active
    else:
        uv = obj.data.uv_layers.new(name="UVMap")
    for loop, uvcoord in zip(uv.data, [(0, 0), (1, 0), (1, 1), (0, 1)] * (len(uv.data) // 4 + 1)):
        loop.uv = uvcoord

    obj.data.materials.append(material)
    obj["product_name"] = ""
    obj["product_identity_status"] = "inferred_blank_slot"
    obj["shelf_object"] = shelf_name
    link_to_collection(obj, collection)
    return obj


def main() -> None:
    args = parse_args()
    payload = json.loads(Path(args.identified_json).read_text(encoding="utf-8"))
    placements = payload["placements"] if isinstance(payload, dict) else payload

    # All shelf objects from the blend scene
    shelves = {
        obj.name: obj
        for obj in bpy.data.objects
        if obj.type == "MESH" and re.match(r"^Unit\d+_Shelf_\d+$", obj.name)
    }

    # Group placements by shelf
    by_shelf: dict[str, list[dict]] = defaultdict(list)
    for p in placements:
        sn = p.get("shelf_object")
        if sn and sn in shelves:
            by_shelf[sn].append(p)

    # Compute per-unit default spacing from shelves that have enough data
    unit_spacing: dict[str, float] = {}
    for shelf_name, shelf_placements in by_shelf.items():
        if len(shelf_placements) < 3:
            continue
        unit = unit_for(shelf_name)
        axis = horizontal_axis(shelf_name)
        coords = [placement_location(p).x if axis == "x" else placement_location(p).y for p in shelf_placements]
        spacing = estimate_spacing(coords)
        if unit not in unit_spacing:
            unit_spacing[unit] = spacing

    default_spacing = statistics.median(unit_spacing.values()) if unit_spacing else args.product_width

    collection = collection_for(args.collection, args.replace)
    blank_mat = make_blank_material()
    created = 0
    summary = []

    for shelf_name in sorted(shelves):
        shelf_obj = shelves[shelf_name]
        shelf_placements = by_shelf.get(shelf_name, [])
        axis = horizontal_axis(shelf_name)
        xb, yb, zb = world_bounds(shelf_obj)
        shelf_top_z = zb[1]

        if len(shelf_placements) >= 2:
            # Fill gaps between existing products
            points = [placement_location(p) for p in shelf_placements]
            coords = [p.x if axis == "x" else p.y for p in points]
            depth_coords = [p.y if axis == "x" else p.x for p in points]
            spacing = estimate_spacing(coords, default_spacing)
            slots = gap_positions(coords, spacing, args.min_gap_multiplier)
            depth_coord = statistics.median(depth_coords)
            mode = "gap_fill"
        elif len(shelf_placements) == 0:
            # Fill entire empty shelf
            span_min = xb[0] if axis == "x" else yb[0]
            span_max = xb[1] if axis == "x" else yb[1]
            span = span_max - span_min
            if span < 0.05:
                continue
            unit = unit_for(shelf_name)
            spacing = unit_spacing.get(unit, default_spacing)
            slots = full_shelf_positions(span_min, span_max, spacing)
            # Depth: midpoint of the opposite axis
            depth_coord = statistics.median([yb[0], yb[1]]) if axis == "x" else statistics.median([xb[0], xb[1]])
            mode = "empty_fill"
        else:
            continue  # 1 placement — not enough to determine spacing reliably

        for i, coord in enumerate(slots):
            create_blank_box(
                f"BlankSlot_{created:04d}_{shelf_name}_{i:02d}",
                shelf_name, coord, depth_coord, shelf_top_z,
                collection, blank_mat,
                args.product_width, args.product_depth, args.product_height,
            )
            created += 1
        summary.append({"shelf_object": shelf_name, "mode": mode, "slots": len(slots)})

    bpy.ops.wm.save_as_mainfile(filepath=args.output)
    print("BLANK_SLOTS_CREATED", created)
    print("SAVED", args.output)
    for item in summary:
        print("FILL", item["shelf_object"], item["mode"], item["slots"])


if __name__ == "__main__":
    main()
