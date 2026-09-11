"""Report shelf rows and product-like objects in a Blender scene."""

from __future__ import annotations

import json
import re
from collections import Counter

import bpy


SHELF_RE = re.compile(r"^Unit\d+_Shelf_\d+$")


def is_helper_box(obj: bpy.types.Object) -> bool:
    return obj.name.startswith("FreshProductBox")


def is_product_like(obj: bpy.types.Object) -> bool:
    if obj.type != "MESH":
        return False
    if is_helper_box(obj):
        return False
    if obj.name.startswith(("LayoutProduct", "FullBlankSlot", "ImageBlankSlot", "GapFill", "InferredSlot")):
        return True
    if obj.get("shelf_object"):
        return True
    if obj.get("product_name") is not None or obj.get("product_identity_status") is not None:
        return True
    return False


shelves = sorted(obj.name for obj in bpy.data.objects if obj.type == "MESH" and SHELF_RE.match(obj.name))
products = [obj for obj in bpy.data.objects if is_product_like(obj)]
helper_boxes = [obj for obj in bpy.data.objects if obj.type == "MESH" and is_helper_box(obj)]

by_shelf = Counter(str(obj.get("shelf_object", "none")) for obj in products)
by_status = Counter(str(obj.get("product_identity_status", "none")) for obj in products)
by_prefix = Counter(obj.name.split("_", 1)[0] for obj in products)
objects_by_collection = {
    coll.name: len([obj for obj in coll.objects if is_product_like(obj)])
    for coll in bpy.data.collections
}
mesh_prefix_by_collection = {
    coll.name: dict(sorted(Counter(obj.name.split("_", 1)[0] for obj in coll.objects if obj.type == "MESH").items()))
    for coll in bpy.data.collections
}

print("SCENE", bpy.data.filepath)
print("SHELF_ROWS", len(shelves))
print("SHELVES", json.dumps(shelves))
print("PRODUCT_LIKE_COUNT", len(products))
print("HELPER_BOX_COUNT", len(helper_boxes))
print("PRODUCTS_BY_SHELF", json.dumps(dict(sorted(by_shelf.items()))))
print("PRODUCTS_BY_STATUS", json.dumps(dict(sorted(by_status.items()))))
print("PRODUCTS_BY_PREFIX", json.dumps(dict(sorted(by_prefix.items()))))
print("PRODUCTS_BY_COLLECTION", json.dumps(dict(sorted(objects_by_collection.items()))))
print("MESH_PREFIX_BY_COLLECTION", json.dumps(dict(sorted(mesh_prefix_by_collection.items()))))
