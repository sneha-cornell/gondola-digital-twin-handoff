"""Add explicit unknown slots to the corrected detected-product layout.

This keeps real detections as product image planes and marks likely missing
facings as unknown slots in a separate collection. It does not clone product
identities, so the result distinguishes accurate detections from inferred gaps.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path

import bpy
from mathutils import Vector


SHELF_RE = re.compile(r"^(Unit\d+)_Shelf_(\d+)$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--placements-json", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--collection", default="Accuracy_Inferred_Unknown_Slots")
    parser.add_argument("--slot-spacing", type=float, default=0.115)
    parser.add_argument("--edge-margin", type=float, default=0.08)
    parser.add_argument("--occupied-threshold", type=float, default=0.075)
    parser.add_argument("--slot-width", type=float, default=0.085)
    parser.add_argument("--slot-height", type=float, default=0.20)
    parser.add_argument("--slot-depth", type=float, default=0.045)
    parser.add_argument(
        "--fill-empty-shelves",
        action="store_true",
        help="Mark shelves with no detections as unknown slots if the unit has detections elsewhere.",
    )
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


def make_material() -> bpy.types.Material:
    mat = bpy.data.materials.get("AccuracyUnknownSlot_Amber") or bpy.data.materials.new("AccuracyUnknownSlot_Amber")
    mat.diffuse_color = (1.0, 0.62, 0.08, 0.72)
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    if bsdf:
        bsdf.inputs["Base Color"].default_value = (1.0, 0.62, 0.08, 0.72)
        bsdf.inputs["Alpha"].default_value = 0.72
    mat.blend_method = "BLEND"
    return mat


def unit_for(shelf_name: str) -> str:
    match = SHELF_RE.match(shelf_name)
    return match.group(1) if match else shelf_name.split("_", 1)[0]


def horizontal_axis(unit: str) -> str:
    return "x" if unit in {"Unit1", "Unit3"} else "y"


def facing_for_unit(unit: str) -> str:
    if unit == "Unit1":
        return "-Y"
    if unit == "Unit3":
        return "+Y"
    if unit == "Unit2":
        return "+X"
    if unit == "Unit4":
        return "-X"
    return "-Y"


def shelf_depth_coord(unit: str, shelf_obj: bpy.types.Object) -> float:
    xb, yb, _ = world_bounds(shelf_obj)
    facing = facing_for_unit(unit)
    if facing == "-Y":
        return yb[0]
    if facing == "+Y":
        return yb[1]
    if facing == "+X":
        return xb[1]
    return xb[0]


def shelf_top_z(shelf_obj: bpy.types.Object) -> float:
    _xb, _yb, zb = world_bounds(shelf_obj)
    return zb[1]


def shelf_horizontal_span(unit: str, shelf_obj: bpy.types.Object, margin: float) -> tuple[float, float]:
    xb, yb, _ = world_bounds(shelf_obj)
    lo, hi = xb if horizontal_axis(unit) == "x" else yb
    return lo + margin, hi - margin


def slot_coords(lo: float, hi: float, spacing: float) -> list[float]:
    if hi <= lo:
        return []
    count = max(1, int(math.floor((hi - lo) / spacing)) + 1)
    if count == 1:
        return [(lo + hi) * 0.5]
    actual = (hi - lo) / float(count - 1)
    return [lo + idx * actual for idx in range(count)]


def product_coord_from_location(unit: str, location: list[float]) -> float:
    return float(location[0] if horizontal_axis(unit) == "x" else location[1])


def create_unknown_slot(index: int, shelf_name: str, shelf_obj: bpy.types.Object, coord: float, args, mat, collection):
    unit = unit_for(shelf_name)
    facing = facing_for_unit(unit)
    depth = shelf_depth_coord(unit, shelf_obj)
    top_z = shelf_top_z(shelf_obj)
    if horizontal_axis(unit) == "x":
        base = Vector((coord, depth, top_z))
    else:
        base = Vector((depth, coord, top_z))

    if facing in {"-Y", "+Y"}:
        dims = (args.slot_width, args.slot_depth, args.slot_height)
        offset = Vector((0, -args.slot_depth * 0.5 if facing == "-Y" else args.slot_depth * 0.5, args.slot_height * 0.5 + 0.012))
    else:
        dims = (args.slot_depth, args.slot_width, args.slot_height)
        offset = Vector((args.slot_depth * 0.5 if facing == "+X" else -args.slot_depth * 0.5, 0, args.slot_height * 0.5 + 0.012))

    bpy.ops.mesh.primitive_cube_add(size=1.0, location=base + offset)
    obj = bpy.context.object
    obj.name = f"AccuracyUnknownSlot_{index:04d}_{shelf_name}"
    obj.dimensions = dims
    bpy.context.view_layer.update()
    obj.data.materials.append(mat)
    obj["product_name"] = "Unknown slot"
    obj["product_identity_status"] = "accuracy_inferred_unknown_slot"
    obj["shelf_object"] = shelf_name
    obj["accuracy_layer"] = "inferred_unknown"
    link_to_collection(obj, collection)
    return obj


def main() -> None:
    args = parse_args()
    payload = json.loads(Path(args.placements_json).read_text(encoding="utf-8"))
    placements = payload.get("placements", [])

    shelves = {
        obj.name: obj
        for obj in bpy.data.objects
        if obj.type == "MESH" and SHELF_RE.match(obj.name)
    }

    occupied_by_shelf: dict[str, list[float]] = defaultdict(list)
    units_with_detections = set()
    real_detections = 0
    for placement in placements:
        shelf_name = str(placement.get("shelf_object") or "")
        if shelf_name not in shelves:
            continue
        status = str(placement.get("product_identity_status") or "")
        if status not in {"recognized", "unknown"}:
            continue
        location = placement.get("location")
        if not location:
            continue
        unit = unit_for(shelf_name)
        occupied_by_shelf[shelf_name].append(product_coord_from_location(unit, location))
        units_with_detections.add(unit)
        real_detections += 1

    shelves_to_mark = set(occupied_by_shelf)
    if args.fill_empty_shelves:
        for shelf_name in shelves:
            if unit_for(shelf_name) in units_with_detections:
                shelves_to_mark.add(shelf_name)

    collection = collection_for(args.collection, args.replace)
    mat = make_material()

    created = 0
    summary = {}
    for shelf_name in sorted(shelves_to_mark):
        shelf_obj = shelves[shelf_name]
        unit = unit_for(shelf_name)
        lo, hi = shelf_horizontal_span(unit, shelf_obj, args.edge_margin)
        occupied = list(occupied_by_shelf.get(shelf_name, []))
        created_here = 0
        for coord in slot_coords(lo, hi, args.slot_spacing):
            if any(abs(coord - seen) <= args.occupied_threshold for seen in occupied):
                continue
            create_unknown_slot(created, shelf_name, shelf_obj, coord, args, mat, collection)
            occupied.append(coord)
            created += 1
            created_here += 1
        summary[shelf_name] = {
            "real_detected_on_shelf": len(occupied_by_shelf.get(shelf_name, [])),
            "unknown_slots_added": created_here,
        }

    bpy.ops.wm.save_as_mainfile(filepath=args.output)
    print("REAL_DETECTIONS", real_detections)
    print("UNKNOWN_SLOTS_CREATED", created)
    print("SUMMARY", json.dumps(summary, sort_keys=True))
    print("SAVED", args.output)


if __name__ == "__main__":
    main()
