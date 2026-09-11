"""Render a front-facing preview of the shelf blend file."""
import bpy
import math
import sys
from mathutils import Vector

def point_camera_at(cam_obj, target: Vector):
    direction = target - cam_obj.location
    rot_quat = direction.to_track_quat('-Z', 'Y')
    cam_obj.rotation_euler = rot_quat.to_euler()

def main():
    scene = bpy.context.scene
    scene.render.engine = 'BLENDER_WORKBENCH'
    scene.display.shading.color_type = 'MATERIAL'
    scene.display.shading.light = 'STUDIO'
    scene.render.resolution_x = 1280
    scene.render.resolution_y = 960

    all_objs = [o for o in bpy.data.objects if o.type == 'MESH']
    if not all_objs:
        print("No mesh objects found"); return

    xs = [o.matrix_world.translation.x for o in all_objs]
    ys = [o.matrix_world.translation.y for o in all_objs]
    zs = [o.matrix_world.translation.z for o in all_objs]
    cx, cy, cz = sum(xs)/len(xs), sum(ys)/len(ys), sum(zs)/len(zs)
    span_x = max(xs) - min(xs)
    span_z = max(zs) - min(zs)
    span = max(span_x, span_z, max(ys) - min(ys))

    print(f"Center: ({cx:.2f}, {cy:.2f}, {cz:.2f})  span={span:.2f}")
    print(f"x=[{min(xs):.2f},{max(xs):.2f}] y=[{min(ys):.2f},{max(ys):.2f}] z=[{min(zs):.2f},{max(zs):.2f}]")

    # Remove old cameras/lights
    for obj in list(bpy.data.objects):
        if obj.type in ('CAMERA', 'LIGHT'):
            bpy.data.objects.remove(obj, do_unlink=True)

    # Camera: isometric-ish front view
    cam_loc = Vector((cx - span * 0.5, cy - span * 1.4, cz + span * 0.3))
    bpy.ops.object.camera_add(location=cam_loc)
    cam = bpy.context.active_object
    point_camera_at(cam, Vector((cx, cy, cz)))
    cam.data.lens = 28
    scene.camera = cam

    # Sun from upper-front
    bpy.ops.object.light_add(type='SUN', location=(cx - span, cy - span, cz + span * 1.5))
    sun = bpy.context.active_object
    point_camera_at(sun, Vector((cx, cy, cz)))
    sun.data.energy = 4.0

    # Bright world ambient
    world = bpy.data.worlds.get('World') or bpy.data.worlds.new('World')
    scene.world = world
    world.use_nodes = True
    bg = world.node_tree.nodes.get('Background')
    if bg:
        bg.inputs[0].default_value = (0.9, 0.9, 0.9, 1.0)
        bg.inputs['Strength'].default_value = 1.5

    out = sys.argv[sys.argv.index('--output') + 1] if '--output' in sys.argv else '/tmp/shelf_preview.png'
    scene.render.filepath = out
    bpy.ops.render.render(write_still=True)
    print("RENDERED", out)

main()
