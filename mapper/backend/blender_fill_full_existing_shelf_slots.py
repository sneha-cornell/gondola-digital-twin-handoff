"""
Fill every existing shelf row with product slots.

Detected products from the strict placement JSON remain untouched. This pass
adds blank placeholder products for empty shelf slots, including shelves or shelf
sections where no product was detected.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import bpy
from mathutils import Vector


SHELF_RE = re.compile(r"^Unit\d+_Shelf_\d+$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--placements-json", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--collection", default="Full_Shelf_Blank_Slots")
    parser.add_argument("--slot-spacing", type=float, default=0.14)
    parser.add_argument("--edge-margin", type=float, default=0.07)
    parser.add_argument("--product-height", type=float, default=0.22)
    parser.add_argument("--product-width", type=float, default=0.12)
    parser.add_argument("--product-depth", type=float, default=0.055)
    parser.add_argument("--occupied-threshold", type=float, default=0.075)
    parser.add_argument("--replace", action="store_true")
    return parser.parse_args(sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else [])


def world_bounds(obj: bpy.types.Object):
    corners = [obj.matrix_world @ Vector(corner) for corner in obj.bound_box]
    xs = [corner.x for corner in corners]
    ys = [corner.y for corner in corners]
    zs = [corner.z for corner in corners]
    return (min(xs), max(xs)), (min(ys), max(ys)), (min(zs), max(zs))


def make_material(name: str, color: tuple[float, float, float, float]) -> bpy.types.Material:
    mat = bpy.data.materials.get(name) or bpy.data.materials.new(name)
    mat.diffuse_color = color
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    if bsdf:
        bsdf.inputs["Base Color"].default_value = color
        bsdf.inputs["Alpha"].default_value = color[3]
    mat.blend_method = "BLEND"
    return mat


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
    if unit == "Unit4":
        return "-X"
    return "-Y"


def placement_location(placement: dict) -> Vector:
    x, y, z = placement["location"]
    return Vector((float(x), float(y), float(z)))


def slot_coords(bounds: tuple[float, float], spacing: float, margin: float) -> list[float]:
    start = bounds[0] + margin
    end = bounds[1] - margin
    if end <= start:
        return []
    count = max(1, int(math.floor((end - start) / spacing)) + 1)
    if count == 1:
        return [(start + end) * 0.5]
    actual_spacing = (end - start) / float(count - 1)
    return [start + actual_spacing * index for index in range(count)]


def shelf_depth_coord(shelf_name: str, shelf_obj: bpy.types.Object) -> float:
    xb, yb, _ = world_bounds(shelf_obj)
    facing = facing_for(shelf_name)
    if facing == "-Y":
        return yb[0]
    if facing == "+Y":
        return yb[1]
    if facing == "+X":
        return xb[1]
    return xb[0]


def is_slot_occupied(slot: float, occupied_coords: list[float], threshold: float) -> bool:
    return any(abs(slot - coord) <= threshold for coord in occupied_coords)


def create_blank_box(
    name: str,
    shelf_name: str,
    slot: float,
    depth_coord: float,
    shelf_top_z: float,
    collection: bpy.types.Collection,
    material: bpy.types.Material,
    product_width: float,
    product_depth: float,
    product_height: float,
) -> bpy.types.Object:
    axis = horizontal_axis(shelf_name)
    facing = facing_for(shelf_name)
    if axis == "x":
        base = Vector((slot, depth_coord, shelf_top_z))
    else:
        base = Vector((depth_coord, slot, shelf_top_z))

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
    obj.data.materials.append(material)
    obj["product_name"] = ""
    obj["product_identity_status"] = "inferred_blank_slot"
    obj["shelf_object"] = shelf_name
    obj["image_layout_replica"] = True
    link_to_collection(obj, collection)
    return obj


def main() -> None:
    args = parse_args()
    payload = json.loads(Path(args.placements_json).read_text(encoding="utf-8"))
    placements = payload["placements"] if isinstance(payload, dict) else payload
    shelves = {
        obj.name: obj
        for obj in bpy.data.objects
        if obj.type == "MESH" and SHELF_RE.match(obj.name)
    }

    placements_by_shelf: dict[str, list[dict]] = defaultdict(list)
    for placement in placements:
        shelf_name = str(placement.get("shelf_object") or "")
        if shelf_name in shelves:
            placements_by_shelf[shelf_name].append(placement)

    collection = collection_for(args.collection, args.replace)
    blank_mat = make_material("FullShelfBlankProduct_NeutralGray", (0.70, 0.74, 0.70, 0.68))

    created = 0
    shelves_filled = 0
    for shelf_name, shelf_obj in sorted(shelves.items()):
        xb, yb, zb = world_bounds(shelf_obj)
        axis = horizontal_axis(shelf_name)
        bounds = xb if axis == "x" else yb
        slots = slot_coords(bounds, args.slot_spacing, args.edge_margin)
        if not slots:
            continue

        occupied = []
        for placement in placements_by_shelf.get(shelf_name, []):
            point = placement_location(placement)
            occupied.append(point.x if axis == "x" else point.y)

        depth_coord = shelf_depth_coord(shelf_name, shelf_obj)
        for slot_index, slot in enumerate(slots):
            if is_slot_occupied(slot, occupied, args.occupied_threshold):
                continue
            create_blank_box(
                f"FullBlankSlot_{created:04d}_{shelf_name}_{slot_index:02d}",
                shelf_name,
                slot,
                depth_coord,
                zb[1],
                collection,
                blank_mat,
                args.product_width,
                args.product_depth,
                args.product_height,
            )
            created += 1
        shelves_filled += 1

    bpy.ops.wm.save_as_mainfile(filepath=args.output)
    print("SHELVES_SCANNED", len(shelves))
    print("SHELVES_FILLED", shelves_filled)
    print("BLANK_SLOTS_CREATED", created)
    print("SAVED", args.output)


if __name__ == "__main__":
    main()
