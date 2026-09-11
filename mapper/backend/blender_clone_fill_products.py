"""Fill shelf slots by cloning nearby detected product facings.

This is for visual shelf stock, not detection scoring. It preserves real
detected FreshProduct planes, removes helper boxes, and populates empty slots
with duplicates of nearby product planes so the Blender rack looks stocked
instead of filled with neutral placeholder boxes.
"""

from __future__ import annotations

import argparse
import math
import re
import statistics
import sys
from collections import defaultdict

import bpy
from mathutils import Vector


SHELF_RE = re.compile(r"^Unit\d+_Shelf_\d+$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--collection", default="Stocked_Product_Clones")
    parser.add_argument("--slot-spacing", type=float, default=0.105)
    parser.add_argument("--clone-width", type=float, default=0.085)
    parser.add_argument("--edge-margin", type=float, default=0.06)
    parser.add_argument("--occupied-threshold", type=float, default=0.075)
    parser.add_argument("--hide-helper-boxes", action="store_true")
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


def plane_offset(shelf_name: str) -> Vector:
    facing = facing_for(shelf_name)
    if facing == "-Y":
        return Vector((0.0, -0.018, 0.012))
    if facing == "+Y":
        return Vector((0.0, 0.018, 0.012))
    if facing == "+X":
        return Vector((0.018, 0.0, 0.012))
    return Vector((-0.018, 0.0, 0.012))


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


def horizontal_coord(shelf_name: str, obj: bpy.types.Object) -> float:
    loc = obj.matrix_world.translation
    return loc.x if horizontal_axis(shelf_name) == "x" else loc.y


def base_location(shelf_name: str, shelf_obj: bpy.types.Object, slot: float) -> Vector:
    _, _, zb = world_bounds(shelf_obj)
    depth_coord = shelf_depth_coord(shelf_name, shelf_obj)
    if horizontal_axis(shelf_name) == "x":
        return Vector((slot, depth_coord, zb[1]))
    return Vector((depth_coord, slot, zb[1]))


def is_detected_product(obj: bpy.types.Object) -> bool:
    return obj.type == "MESH" and obj.name.startswith("FreshProduct_") and not obj.name.startswith("FreshProductBox_")


def clone_product(
    source: bpy.types.Object,
    name: str,
    location: Vector,
    shelf_name: str,
    collection: bpy.types.Collection,
    clone_width: float,
) -> bpy.types.Object:
    clone = source.copy()
    clone.data = source.data.copy()
    clone.name = name
    clone.data.name = f"{name}_mesh"
    clone.location = location
    clone["product_identity_status"] = "inferred_stocked_clone"
    clone["shelf_object"] = shelf_name
    clone["cloned_from"] = source.name
    clone["image_layout_replica"] = True
    collection.objects.link(clone)
    bpy.context.view_layer.update()
    axis = horizontal_axis(shelf_name)
    xb, yb, _ = world_bounds(clone)
    current_width = (xb[1] - xb[0]) if axis == "x" else (yb[1] - yb[0])
    if current_width > clone_width and current_width > 1e-6:
        factor = clone_width / current_width
        if axis == "x":
            clone.scale.x *= factor
        else:
            clone.scale.y *= factor
    return clone


def nearest_product(products: list[bpy.types.Object], shelf_name: str, slot: float) -> bpy.types.Object:
    return min(products, key=lambda obj: abs(horizontal_coord(shelf_name, obj) - slot))


def main() -> None:
    args = parse_args()

    if args.hide_helper_boxes:
        for obj in bpy.data.objects:
            if obj.type == "MESH" and obj.name.startswith("FreshProductBox_"):
                obj.hide_viewport = True
                obj.hide_render = True

    shelves = {
        obj.name: obj
        for obj in bpy.data.objects
        if obj.type == "MESH" and SHELF_RE.match(obj.name)
    }
    products = [obj for obj in bpy.data.objects if is_detected_product(obj)]

    products_by_shelf: dict[str, list[bpy.types.Object]] = defaultdict(list)
    products_by_unit: dict[str, list[bpy.types.Object]] = defaultdict(list)
    for obj in products:
        shelf_name = str(obj.get("shelf_object") or "")
        if shelf_name in shelves:
            products_by_shelf[shelf_name].append(obj)
            products_by_unit[unit_for(shelf_name)].append(obj)

    collection = collection_for(args.collection, args.replace)
    created = 0
    shelves_filled = 0
    per_shelf = {}

    for shelf_name, shelf_obj in sorted(shelves.items()):
        axis = horizontal_axis(shelf_name)
        xb, yb, _ = world_bounds(shelf_obj)
        bounds = xb if axis == "x" else yb
        slots = slot_coords(bounds, args.slot_spacing, args.edge_margin)
        if not slots:
            continue

        shelf_products = products_by_shelf.get(shelf_name, [])
        source_pool = shelf_products or products_by_unit.get(unit_for(shelf_name), []) or products
        if not source_pool:
            continue

        occupied = [horizontal_coord(shelf_name, obj) for obj in shelf_products]
        created_here = 0
        for slot_index, slot in enumerate(slots):
            if any(abs(slot - coord) <= args.occupied_threshold for coord in occupied):
                continue
            source = nearest_product(source_pool, shelf_name, slot)
            location = base_location(shelf_name, shelf_obj, slot) + plane_offset(shelf_name)
            clone_product(
                source,
                f"StockedClone_{created:04d}_{shelf_name}_{slot_index:02d}",
                location,
                shelf_name,
                collection,
                args.clone_width,
            )
            occupied.append(slot)
            created += 1
            created_here += 1
        shelves_filled += 1
        per_shelf[shelf_name] = len(shelf_products) + created_here

    bpy.ops.wm.save_as_mainfile(filepath=args.output)
    print("SHELVES_SCANNED", len(shelves))
    print("SHELVES_FILLED", shelves_filled)
    print("DETECTED_PRODUCTS_KEPT", len(products))
    print("STOCKED_CLONES_CREATED", created)
    print("FINAL_PRODUCTS_BY_SHELF", dict(sorted(per_shelf.items())))
    print("SAVED", args.output)


if __name__ == "__main__":
    main()
