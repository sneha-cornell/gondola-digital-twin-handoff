import bpy, sys
from mathutils import Vector

def pt(obj, tgt):
    obj.rotation_euler = (tgt - obj.location).to_track_quat('-Z', 'Y').to_euler()

scene = bpy.context.scene
scene.render.engine = 'BLENDER_WORKBENCH'
scene.render.resolution_x = 1920
scene.render.resolution_y = 1080

# Remove existing cameras/lights
for o in list(bpy.data.objects):
    if o.type in ('CAMERA', 'LIGHT'):
        bpy.data.objects.remove(o, do_unlink=True)

# Find Unit2 objects bounding box
unit2_objs = [o for o in bpy.data.objects if str(o.get('shelf_object', '')).startswith('Unit2') and o.type == 'MESH']
if not unit2_objs:
    unit2_objs = [o for o in bpy.data.objects if 'Unit2' in o.name and o.type == 'MESH']

print(f"Unit2 objects: {len(unit2_objs)}")
if unit2_objs:
    xs = [o.matrix_world.translation.x for o in unit2_objs]
    ys = [o.matrix_world.translation.y for o in unit2_objs]
    zs = [o.matrix_world.translation.z for o in unit2_objs]
    cx = (min(xs)+max(xs))/2
    cy = (min(ys)+max(ys))/2
    cz = (min(zs)+max(zs))/2
    span_z = max(zs)-min(zs)
    span_x = max(xs)-min(xs)
    span = max(span_z, span_x, 0.5)
    print(f"Unit2 center: ({cx:.2f},{cy:.2f},{cz:.2f}) span_x={span_x:.2f} span_z={span_z:.2f}")
    # Unit2 faces +X direction — camera on high-X side
    cam_loc = Vector((cx + span * 2.0, cy, cz))
else:
    cx, cy, cz, span = 4.0, 0.4, 1.6, 2.0
    cam_loc = Vector((cx + 4, cy, cz))

bpy.ops.object.camera_add(location=cam_loc)
cam = bpy.context.active_object
pt(cam, Vector((cx, cy, cz)))
cam.data.lens = 50
scene.camera = cam

bpy.ops.object.light_add(type='SUN', location=(cx + 6, cy - 2, cz + 3))
sun = bpy.context.active_object
pt(sun, Vector((cx, cy, cz)))
sun.data.energy = 5.0

w = bpy.data.worlds.get('World') or bpy.data.worlds.new('World')
scene.world = w
w.use_nodes = True
bg = w.node_tree.nodes.get('Background')
if bg:
    bg.inputs[0].default_value = (0.85, 0.85, 0.85, 1)
    bg.inputs['Strength'].default_value = 1.5

out = sys.argv[sys.argv.index('--output') + 1]
scene.render.filepath = out
bpy.ops.render.render(write_still=True)
print("RENDERED", out)
