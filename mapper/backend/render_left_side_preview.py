"""Render the left side rack face (Unit4) straight-on."""

import sys

import bpy
from mathutils import Vector


def point_camera_at(cam_obj, target: Vector):
    direction = target - cam_obj.location
    cam_obj.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()


def bounds(objs):
    pts = []
    for obj in objs:
        pts.extend(obj.matrix_world @ Vector(corner) for corner in obj.bound_box)
    return (
        (min(p.x for p in pts), max(p.x for p in pts)),
        (min(p.y for p in pts), max(p.y for p in pts)),
        (min(p.z for p in pts), max(p.z for p in pts)),
    )


def main():
    scene = bpy.context.scene
    scene.render.engine = "BLENDER_WORKBENCH"
    scene.display.shading.light = "STUDIO"
    scene.display.shading.color_type = "TEXTURE"
    scene.render.resolution_x = 960
    scene.render.resolution_y = 1280

    objs = [
        obj for obj in bpy.data.objects
        if obj.type == "MESH" and not obj.hide_render and (
            obj.name.startswith("Unit4_")
            or str(obj.get("shelf_object") or "").startswith("Unit4_")
        )
    ]
    if not objs:
        print("No Unit4 visible objects found")
        return

    xb, yb, zb = bounds(objs)
    target = Vector(((xb[0] + xb[1]) * 0.5, (yb[0] + yb[1]) * 0.5, (zb[0] + zb[1]) * 0.5))
    span = max(yb[1] - yb[0], zb[1] - zb[0], 1.0)

    for obj in list(bpy.data.objects):
        if obj.type in ("CAMERA", "LIGHT"):
            bpy.data.objects.remove(obj, do_unlink=True)

    cam_loc = Vector((xb[0] - span * 1.35, target.y, target.z + span * 0.05))
    bpy.ops.object.camera_add(location=cam_loc)
    cam = bpy.context.active_object
    point_camera_at(cam, target)
    cam.data.lens = 38
    scene.camera = cam

    bpy.ops.object.light_add(type="SUN", location=(xb[0] - span, target.y - span, target.z + span))
    sun = bpy.context.active_object
    point_camera_at(sun, target)
    sun.data.energy = 3.0

    out = sys.argv[sys.argv.index("--output") + 1] if "--output" in sys.argv else "/tmp/left_side_preview.png"
    scene.render.filepath = out
    bpy.ops.render.render(write_still=True)
    print("RENDERED", out)


main()
