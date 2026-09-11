"""Fill corrected shelf gaps by cloning products from the same shelf row."""

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
    parser.add_argument("--collection", default="Row_Local_Product_Fill")
    parser.add_argument("--slot-spacing", type=float, default=0.115)
    parser.add_argument("--edge-margin", type=float, default=0.08)
    parser.add_argument("--occupied-threshold", type=float, default=0.075)
    parser.add_argument("--clone-width", type=float, default=0.085)
    parser.add_argument(
        "--fill-empty-shelves",
        action="store_true",
        help="Fill shelves with no detections using nearest detected shelf from the same unit.",
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


def base_location(unit: str, shelf_obj: bpy.types.Object, coord: float) -> Vector:
    depth = shelf_depth_coord(unit, shelf_obj)
    top_z = shelf_top_z(shelf_obj)
    if horizontal_axis(unit) == "x":
        return Vector((coord, depth, top_z))
    return Vector((depth, coord, top_z))


def plane_offset(unit: str) -> Vector:
    facing = facing_for_unit(unit)
    if facing == "-Y":
        return Vector((0, -0.018, 0.012))
    if facing == "+Y":
        return Vector((0, 0.018, 0.012))
    if facing == "+X":
        return Vector((0.018, 0, 0.012))
    return Vector((-0.018, 0, 0.012))


def horizontal_coord(unit: str, obj: bpy.types.Object) -> float:
    loc = obj.matrix_world.translation
    return loc.x if horizontal_axis(unit) == "x" else loc.y


def plane_for_detection_id(detection_id: str) -> bpy.types.Object | None:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", detection_id)
    candidates = [
        obj for obj in bpy.data.objects
        if obj.type == "MESH" and obj.name.startswith("LayoutProduct_") and obj.name.endswith("_" + safe)
    ]
    return candidates[0] if candidates else None


def clone_plane(source: bpy.types.Object, name: str, location: Vector, shelf_name: str, clone_width: float, collection) -> bpy.types.Object:
    clone = source.copy()
    clone.data = source.data.copy()
    clone.name = name
    clone.data.name = f"{name}_mesh"
    clone.location = location
    clone["product_identity_status"] = "row_local_stocked_clone"
    clone["shelf_object"] = shelf_name
    clone["cloned_from"] = source.name
    clone["image_layout_replica"] = True
    collection.objects.link(clone)
    bpy.context.view_layer.update()

    unit = unit_for(shelf_name)
    axis = horizontal_axis(unit)
    xb, yb, _ = world_bounds(clone)
    width = xb[1] - xb[0] if axis == "x" else yb[1] - yb[0]
    if width > clone_width and width > 1e-6:
        factor = clone_width / width
        if axis == "x":
            clone.scale.x *= factor
        else:
            clone.scale.y *= factor
    return clone


def main() -> None:
    args = parse_args()
    payload = json.loads(Path(args.placements_json).read_text(encoding="utf-8"))
    placements = payload.get("placements", [])

    shelves = {
        obj.name: obj
        for obj in bpy.data.objects
        if obj.type == "MESH" and SHELF_RE.match(obj.name)
    }

    source_by_shelf: dict[str, list[bpy.types.Object]] = defaultdict(list)
    source_by_unit: dict[str, list[bpy.types.Object]] = defaultdict(list)
    for placement in placements:
        shelf_name = str(placement.get("shelf_object") or "")
        detection_id = str(placement.get("detection_id") or "")
        if shelf_name not in shelves or not detection_id:
            continue
        source = plane_for_detection_id(detection_id)
        if source is not None and not source.hide_render:
            source_by_shelf[shelf_name].append(source)
            source_by_unit[unit_for(shelf_name)].append(source)

    collection = collection_for(args.collection, args.replace)
    created = 0
    summary = {}

    shelves_to_fill = set(source_by_shelf)
    if args.fill_empty_shelves:
        for shelf_name in shelves:
            if source_by_unit.get(unit_for(shelf_name)):
                shelves_to_fill.add(shelf_name)

    for shelf_name in sorted(shelves_to_fill):
        shelf_sources = source_by_shelf.get(shelf_name, [])
        source_pool = shelf_sources or source_by_unit.get(unit_for(shelf_name), [])
        if not source_pool:
            continue
        shelf_obj = shelves[shelf_name]
        unit = unit_for(shelf_name)
        lo, hi = shelf_horizontal_span(unit, shelf_obj, args.edge_margin)
        slots = slot_coords(lo, hi, args.slot_spacing)
        occupied = [horizontal_coord(unit, source) for source in shelf_sources]
        created_here = 0
        for idx, slot in enumerate(slots):
            if any(abs(slot - coord) <= args.occupied_threshold for coord in occupied):
                continue
            source = min(source_pool, key=lambda obj: abs(horizontal_coord(unit, obj) - slot))
            clone_plane(
                source,
                f"RowLocalClone_{created:04d}_{shelf_name}_{idx:02d}",
                base_location(unit, shelf_obj, slot) + plane_offset(unit),
                shelf_name,
                args.clone_width,
                collection,
            )
            occupied.append(slot)
            created += 1
            created_here += 1
        summary[shelf_name] = {
            "detected_on_shelf": len(shelf_sources),
            "source_pool": len(source_pool),
            "cloned": created_here,
            "final": len(shelf_sources) + created_here,
        }

    bpy.ops.wm.save_as_mainfile(filepath=args.output)
    print("SHELVES_WITH_DETECTIONS", len(source_by_shelf))
    print("ROW_LOCAL_CLONES_CREATED", created)
    print("SUMMARY", json.dumps(summary, sort_keys=True))
    print("SAVED", args.output)


if __name__ == "__main__":
    main()
