"""Render a preview using texture colors where available."""

import sys

import bpy
from mathutils import Vector


def point_camera_at(cam_obj, target: Vector):
    direction = target - cam_obj.location
    cam_obj.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()


def main():
    scene = bpy.context.scene
    scene.render.engine = "BLENDER_WORKBENCH"
    scene.display.shading.light = "STUDIO"
    scene.display.shading.color_type = "TEXTURE"
    scene.render.resolution_x = 1280
    scene.render.resolution_y = 960

    visible_meshes = [o for o in bpy.data.objects if o.type == "MESH" and not o.hide_render]
    if not visible_meshes:
        print("No visible mesh objects found")
        return

    xs = [o.matrix_world.translation.x for o in visible_meshes]
    ys = [o.matrix_world.translation.y for o in visible_meshes]
    zs = [o.matrix_world.translation.z for o in visible_meshes]
    cx, cy, cz = sum(xs) / len(xs), sum(ys) / len(ys), sum(zs) / len(zs)
    span = max(max(xs) - min(xs), max(ys) - min(ys), max(zs) - min(zs))

    for obj in list(bpy.data.objects):
        if obj.type in ("CAMERA", "LIGHT"):
            bpy.data.objects.remove(obj, do_unlink=True)

    cam_loc = Vector((cx - span * 0.45, cy - span * 1.35, cz + span * 0.35))
    bpy.ops.object.camera_add(location=cam_loc)
    cam = bpy.context.active_object
    point_camera_at(cam, Vector((cx, cy, cz)))
    cam.data.lens = 30
    scene.camera = cam

    bpy.ops.object.light_add(type="SUN", location=(cx - span, cy - span, cz + span * 1.6))
    sun = bpy.context.active_object
    point_camera_at(sun, Vector((cx, cy, cz)))
    sun.data.energy = 3.0

    out = sys.argv[sys.argv.index("--output") + 1] if "--output" in sys.argv else "/tmp/texture_preview.png"
    scene.render.filepath = out
    bpy.ops.render.render(write_still=True)
    print("RENDERED", out)


main()
