import bpy, mathutils

bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.wm.ply_import(filepath="/Users/manaswi/Downloads/gondolaaa11/mesh_output/sparse_points.ply")
obj = bpy.context.selected_objects[0]

bbox = [obj.matrix_world @ mathutils.Vector(c) for c in obj.bound_box]
xs = [v.x for v in bbox]; ys = [v.y for v in bbox]; zs = [v.z for v in bbox]
print("PT BOUNDS X:", min(xs), max(xs))
print("PT BOUNDS Y:", min(ys), max(ys))
print("PT BOUNDS Z:", min(zs), max(zs))
print("PT COUNT:", len(obj.data.vertices))

# use geometry nodes point cloud -> to make points visible, convert verts to small spheres via particle-less approach:
# simplest: switch object display as point cloud using 'VERTEX' display, and render in solid/wireframe won't show points in eevee.
# Instead create small icospheres at each vertex via instancing (limit count for speed)
import random
mesh = obj.data
n = len(mesh.vertices)
step = max(1, n // 8000)  # cap displayed points for render speed

bpy.ops.mesh.primitive_ico_sphere_add(subdivisions=1, radius=0.03)
proto = bpy.context.active_object
proto.name = "proto"

coll = bpy.data.collections.new("pts")
bpy.context.scene.collection.children.link(coll)

mat = bpy.data.materials.new("m")
mat.diffuse_color = (1,0.3,0.3,1)

for i in range(0, n, step):
    co = obj.matrix_world @ mesh.vertices[i].co
    inst = proto.copy()
    inst.data = proto.data
    inst.location = co
    coll.objects.link(inst)

bpy.data.objects.remove(proto, do_unlink=True)
bpy.data.objects.remove(obj, do_unlink=True)

scene = bpy.context.scene
scene.render.engine = 'BLENDER_WORKBENCH'
scene.render.resolution_x = 900
scene.render.resolution_y = 700

cam_data = bpy.data.cameras.new("cam")
cam_obj = bpy.data.objects.new("cam", cam_data)
scene.collection.objects.link(cam_obj)
scene.camera = cam_obj

center = mathutils.Vector(((min(xs)+max(xs))/2, (min(ys)+max(ys))/2, (min(zs)+max(zs))/2))
size = max(max(xs)-min(xs), max(ys)-min(ys), max(zs)-min(zs), 0.1)
cam_obj.location = center + mathutils.Vector((size*1.6, -size*1.6, size*1.1))
direction = center - cam_obj.location
rot_quat = direction.to_track_quat('-Z', 'Y')
cam_obj.rotation_euler = rot_quat.to_euler()
cam_data.clip_end = size*10

scene.render.filepath = "/Users/manaswi/Downloads/gondolaaa11/mesh_output/points_preview.png"
bpy.ops.render.render(write_still=True)
