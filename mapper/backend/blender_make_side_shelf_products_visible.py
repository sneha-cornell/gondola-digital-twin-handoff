"""Duplicate side-rack product planes onto the visible inner side face."""

from __future__ import annotations

import argparse
import re
import sys
from collections import defaultdict

import bpy
from mathutils import Vector


SHELF_RE = re.compile(r"^(Unit\d+)_Shelf_(\d+)$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--unit", default="Unit4")
    parser.add_argument("--collection", default="Visible_Side_Shelf_Product_Duplicates")
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


def shelf_objects(unit: str) -> dict[str, bpy.types.Object]:
    return {
        obj.name: obj
        for obj in bpy.data.objects
        if obj.type == "MESH" and obj.name.startswith(unit + "_Shelf_")
    }


def is_visible_product(obj: bpy.types.Object, unit: str) -> bool:
    if obj.type != "MESH" or obj.hide_render:
        return False
    if not obj.name.startswith(("LayoutProduct_", "RowLocalClone_")):
        return False
    shelf_name = str(obj.get("shelf_object") or "")
    return shelf_name.startswith(unit + "_Shelf_")


def main() -> None:
    args = parse_args()
    shelves = shelf_objects(args.unit)
    collection = collection_for(args.collection, args.replace)

    products_by_shelf: dict[str, list[bpy.types.Object]] = defaultdict(list)
    for obj in bpy.data.objects:
        if is_visible_product(obj, args.unit):
            products_by_shelf[str(obj.get("shelf_object"))].append(obj)

    created = 0
    for shelf_name, products in sorted(products_by_shelf.items()):
        shelf = shelves.get(shelf_name)
        if not shelf:
            continue
        xb, _yb, _zb = world_bounds(shelf)
        inner_x = xb[1] + 0.018 if args.unit == "Unit4" else xb[0] - 0.018
        for source in products:
            clone = source.copy()
            clone.data = source.data.copy()
            clone.name = f"VisibleSide_{created:04d}_{source.name}"
            clone.data.name = f"{clone.name}_mesh"
            clone.location = source.location.copy()
            clone.location.x = inner_x
            clone["product_identity_status"] = "visible_side_duplicate"
            clone["shelf_object"] = shelf_name
            clone["duplicated_from"] = source.name
            collection.objects.link(clone)
            created += 1

    bpy.ops.wm.save_as_mainfile(filepath=args.output)
    print("UNIT", args.unit)
    print("VISIBLE_SIDE_DUPLICATES_CREATED", created)
    print("SAVED", args.output)


if __name__ == "__main__":
    main()
