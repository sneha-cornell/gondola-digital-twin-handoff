"""
Fill missing product slots between detected products on existing Blender shelves.

This is a second pass after blender_project_from_images_colmap.py. It preserves
all detected products and adds blank placeholder facings only for likely missing
slots between detected products on the same shelf.
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
    parser.add_argument("--placements-json", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--collection", default="Blank_Filled_Missing_Slots")
    parser.add_argument("--product-height", type=float, default=0.22)
    parser.add_argument("--product-width", type=float, default=0.14)
    parser.add_argument("--product-depth", type=float, default=0.055)
    parser.add_argument("--min-gap-multiplier", type=float, default=1.65)
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


def estimate_spacing(coords: list[float], default: float = 0.16) -> float:
    if len(coords) < 2:
        return default
    deltas = [
        b - a
        for a, b in zip(sorted(coords), sorted(coords)[1:])
        if 0.045 <= b - a <= 0.34
    ]
    if not deltas:
        return default
    return max(0.08, min(0.22, statistics.median(deltas)))


def slot_positions(coords: list[float], spacing: float, min_gap_multiplier: float) -> list[float]:
    slots: list[float] = []
    ordered = sorted(coords)
    for left, right in zip(ordered, ordered[1:]):
        gap = right - left
        if gap < spacing * min_gap_multiplier:
            continue
        missing_count = max(0, round(gap / spacing) - 1)
        if missing_count <= 0:
            continue
        step = gap / float(missing_count + 1)
        for idx in range(1, missing_count + 1):
            slots.append(left + step * idx)
    return slots


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
    if axis == "x":
        base = Vector((coord, depth_coord, shelf_top_z))
    else:
        base = Vector((depth_coord, coord, shelf_top_z))

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
        if obj.type == "MESH" and re.match(r"^Unit\d+_Shelf_\d+$", obj.name)
    }

    by_shelf: dict[str, list[dict]] = defaultdict(list)
    for placement in placements:
        shelf_name = placement.get("shelf_object")
        if shelf_name in shelves:
            by_shelf[str(shelf_name)].append(placement)

    collection = collection_for(args.collection, args.replace)
    blank_mat = make_material("InferredBlankProduct_NeutralGray", (0.72, 0.76, 0.72, 0.72))

    created = 0
    summary = []
    for shelf_name, shelf_placements in sorted(by_shelf.items()):
        if len(shelf_placements) < 2:
            continue
        axis = horizontal_axis(shelf_name)
        points = [placement_location(item) for item in shelf_placements]
        coords = [point.x if axis == "x" else point.y for point in points]
        depth_coords = [point.y if axis == "x" else point.x for point in points]
        spacing = estimate_spacing(coords)
        slots = slot_positions(coords, spacing, args.min_gap_multiplier)
        if not slots:
            continue
        _, _, zb = world_bounds(shelves[shelf_name])
        depth_coord = statistics.median(depth_coords)
        for slot_index, coord in enumerate(slots):
            create_blank_box(
                f"BlankSlot_{created:04d}_{shelf_name}_{slot_index:02d}",
                shelf_name,
                coord,
                depth_coord,
                zb[1],
                collection,
                blank_mat,
                args.product_width,
                args.product_depth,
                args.product_height,
            )
            created += 1
        summary.append({"shelf_object": shelf_name, "spacing": spacing, "blank_slots": len(slots)})

    bpy.ops.wm.save_as_mainfile(filepath=args.output)
    print("SHELVES_WITH_PLACEMENTS", len(by_shelf))
    print("BLANK_SLOTS_CREATED", created)
    print("SAVED", args.output)
    for item in summary[:30]:
        print("FILL_SUMMARY", item["shelf_object"], item["blank_slots"], f"spacing={item['spacing']:.3f}")


if __name__ == "__main__":
    main()
