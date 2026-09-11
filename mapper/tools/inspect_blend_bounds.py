import bpy
from mathutils import Vector

print("BLEND_FILE", bpy.data.filepath)

for obj in sorted(bpy.data.objects, key=lambda item: item.name):
    if obj.type != "MESH":
        continue
    corners = [obj.matrix_world @ Vector(corner) for corner in obj.bound_box]
    xs = [corner.x for corner in corners]
    ys = [corner.y for corner in corners]
    zs = [corner.z for corner in corners]
    print(
        "BOUNDS "
        f"{obj.name} "
        f"x=({min(xs):.4f},{max(xs):.4f}) "
        f"y=({min(ys):.4f},{max(ys):.4f}) "
        f"z=({min(zs):.4f},{max(zs):.4f}) "
        f"loc=({obj.location.x:.4f},{obj.location.y:.4f},{obj.location.z:.4f}) "
        f"scale=({obj.scale.x:.4f},{obj.scale.y:.4f},{obj.scale.z:.4f})"
    )
