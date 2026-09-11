import bpy, sys
from mathutils import Vector

def pt(obj, tgt):
    obj.rotation_euler = (tgt - obj.location).to_track_quat('-Z', 'Y').to_euler()

scene = bpy.context.scene
scene.render.engine = 'BLENDER_WORKBENCH'
scene.render.resolution_x = 1280
scene.render.resolution_y = 960

for o in list(bpy.data.objects):
    if o.type in ('CAMERA', 'LIGHT'):
        bpy.data.objects.remove(o, do_unlink=True)

cx, cy, cz = 2.0, 0.4, 1.6

# Unit2 +X face view (camera at high X)
bpy.ops.object.camera_add(location=(cx + 9, cy, cz))
cam = bpy.context.active_object
pt(cam, Vector((cx, cy, cz)))
cam.data.lens = 35
scene.camera = cam

bpy.ops.object.light_add(type='SUN', location=(cx + 8, cy - 3, cz + 4))
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
