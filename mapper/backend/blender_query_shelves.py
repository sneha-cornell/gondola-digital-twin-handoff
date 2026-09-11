"""Query shelf objects and product counts in the blend file."""
import bpy
import json

shelves = sorted(obj.name for obj in bpy.data.objects if "_Shelf_" in obj.name and obj.type == "MESH")
products = sorted(obj.name for obj in bpy.data.objects if obj.name.startswith("LayoutProduct"))

print("SHELVES", json.dumps(shelves))
print("PRODUCT_COUNT", len(products))

# Count products per shelf
from collections import Counter
by_shelf = Counter(obj.get("shelf_object", "none") for obj in bpy.data.objects if obj.name.startswith("LayoutProduct"))
print("BY_SHELF", json.dumps(dict(sorted(by_shelf.items()))))
