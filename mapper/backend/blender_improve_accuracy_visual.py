"""Polish the accuracy-layer scene for review.

Keeps the semantic layers intact:
- LayoutProduct_* are real detected product crops.
- AccuracyUnknownSlot_* are inferred unknown slots.

This pass makes unknown slots less visually dominant and duplicates side-rack
product/unknown faces onto the visible inner side so angled previews are easier
to inspect.
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict

import bpy
from mathutils import Vector


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--side-unit", default="Unit4")
    parser.add_argument("--collection", default="Accuracy_Visible_Side_Duplicates")
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


def make_unknown_material() -> bpy.types.Material:
    mat = bpy.data.materials.get("AccuracyUnknownSlot_TranslucentAmber") or bpy.data.materials.new("AccuracyUnknownSlot_TranslucentAmber")
    color = (1.0, 0.58, 0.05, 0.34)
    mat.diffuse_color = color
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    if bsdf:
        bsdf.inputs["Base Color"].default_value = color
        bsdf.inputs["Alpha"].default_value = color[3]
        if "Roughness" in bsdf.inputs:
            bsdf.inputs["Roughness"].default_value = 0.68
    mat.blend_method = "BLEND"
    mat.show_transparent_back = True
    return mat


def restyle_unknown_slots() -> int:
    mat = make_unknown_material()
    changed = 0
    for obj in bpy.data.objects:
        if obj.type != "MESH" or not obj.name.startswith("AccuracyUnknownSlot_"):
            continue
        obj.data.materials.clear()
        obj.data.materials.append(mat)
        obj.color = mat.diffuse_color
        # Make unknowns read as markers, not product packages.
        obj.scale.z *= 0.72
        obj["review_note"] = "Inferred unknown slot, not a product identity."
        changed += 1
    return changed


def shelves_for_unit(unit: str) -> dict[str, bpy.types.Object]:
    return {
        obj.name: obj
        for obj in bpy.data.objects
        if obj.type == "MESH" and obj.name.startswith(unit + "_Shelf_")
    }


def is_side_object(obj: bpy.types.Object, unit: str) -> bool:
    if obj.type != "MESH" or obj.hide_render:
        return False
    shelf_name = str(obj.get("shelf_object") or "")
    if not shelf_name.startswith(unit + "_Shelf_"):
        return False
    return obj.name.startswith(("LayoutProduct_", "AccuracyUnknownSlot_"))


def duplicate_side_faces(unit: str, collection: bpy.types.Collection) -> int:
    shelves = shelves_for_unit(unit)
    products_by_shelf: dict[str, list[bpy.types.Object]] = defaultdict(list)
    for obj in bpy.data.objects:
        if is_side_object(obj, unit):
            products_by_shelf[str(obj.get("shelf_object"))].append(obj)

    created = 0
    for shelf_name, objects in sorted(products_by_shelf.items()):
        shelf = shelves.get(shelf_name)
        if not shelf:
            continue
        xb, _yb, _zb = world_bounds(shelf)
        inner_x = xb[1] + 0.018 if unit == "Unit4" else xb[0] - 0.018
        for source in objects:
            clone = source.copy()
            clone.data = source.data.copy()
            clone.name = f"AccuracyVisibleSide_{created:04d}_{source.name}"
            clone.data.name = f"{clone.name}_mesh"
            clone.location = source.location.copy()
            clone.location.x = inner_x
            clone["visible_side_duplicate"] = True
            clone["duplicated_from"] = source.name
            collection.objects.link(clone)
            created += 1
    return created


def main() -> None:
    args = parse_args()
    collection = collection_for(args.collection, args.replace)
    unknowns = restyle_unknown_slots()
    side_dupes = duplicate_side_faces(args.side_unit, collection)
    bpy.ops.wm.save_as_mainfile(filepath=args.output)
    print("UNKNOWN_SLOTS_RESTYLED", unknowns)
    print("SIDE_DUPLICATES_CREATED", side_dupes)
    print("SAVED", args.output)


if __name__ == "__main__":
    main()
