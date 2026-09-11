"""Render each unit face at 4K showing product labels."""
import bpy, sys
from mathutils import Vector

def pt(obj, tgt):
    obj.rotation_euler = (tgt - obj.location).to_track_quat('-Z', 'Y').to_euler()

scene = bpy.context.scene
scene.render.engine = 'BLENDER_WORKBENCH'
scene.render.resolution_x = 3840
scene.render.resolution_y = 2160

w = bpy.data.worlds.get('World') or bpy.data.worlds.new('World')
scene.world = w
w.use_nodes = True
bg = w.node_tree.nodes.get('Background')
if bg:
    bg.inputs[0].default_value = (0.92, 0.92, 0.92, 1)
    bg.inputs['Strength'].default_value = 2.0

out_dir = sys.argv[sys.argv.index('--out-dir') + 1]

unit_configs = {
    'Unit1': ('y', -1),
    'Unit2': ('x', +1),
    'Unit3': ('y', +1),
    'Unit4': ('x', -1),
}

for unit_name, (axis, sign) in unit_configs.items():
    # Include MESH and FONT objects for bounding box
    objs = [o for o in bpy.data.objects
            if str(o.get('shelf_object', '')).startswith(unit_name) and o.type == 'MESH']
    if not objs:
        print(f"No objects for {unit_name}")
        continue

    xs = [o.matrix_world.translation.x for o in objs]
    ys = [o.matrix_world.translation.y for o in objs]
    zs = [o.matrix_world.translation.z for o in objs]
    cx = (min(xs)+max(xs))/2
    cy = (min(ys)+max(ys))/2
    cz = (min(zs)+max(zs))/2
    span_h = (max(xs)-min(xs)) if axis == 'y' else (max(ys)-min(ys))
    span_v = max(zs)-min(zs)
    span = max(span_h, span_v, 0.5)

    # Tight distance — just enough to fit the face
    aspect = scene.render.resolution_x / scene.render.resolution_y  # 16/9
    lens = 85  # longer focal = less distortion, tighter crop
    # FOV from lens on 36mm sensor: 2*atan(18/lens) rad
    import math
    fov_v = 2 * math.atan(12 / lens)   # vertical FOV (24mm sensor height)
    fov_h = 2 * math.atan(18 / lens)   # horizontal FOV
    dist_v = (span_v / 2) / math.tan(fov_v / 2) * 1.08
    dist_h = (span_h / 2) / math.tan(fov_h / 2) * 1.08
    dist = max(dist_v, dist_h)

    for o in list(bpy.data.objects):
        if o.type in ('CAMERA', 'LIGHT'):
            bpy.data.objects.remove(o, do_unlink=True)

    if axis == 'y':
        cam_loc = Vector((cx, cy - sign * dist, cz))
    else:
        cam_loc = Vector((cx + sign * dist, cy, cz))

    bpy.ops.object.camera_add(location=cam_loc)
    cam = bpy.context.active_object
    pt(cam, Vector((cx, cy, cz)))
    cam.data.lens = lens
    scene.camera = cam

    if axis == 'y':
        sun_loc = Vector((cx - 1, cy - sign * (dist + 2), cz + span * 0.8))
    else:
        sun_loc = Vector((cx + sign * (dist + 2), cy - 1, cz + span * 0.8))
    bpy.ops.object.light_add(type='SUN', location=sun_loc)
    sun = bpy.context.active_object
    pt(sun, Vector((cx, cy, cz)))
    sun.data.energy = 5.0

    out = f"{out_dir}/{unit_name.lower()}_labeled.png"
    scene.render.filepath = out
    bpy.ops.render.render(write_still=True)
    print(f"RENDERED {unit_name} → {out}  span_h={span_h:.2f} span_v={span_v:.2f} dist={dist:.2f}")
