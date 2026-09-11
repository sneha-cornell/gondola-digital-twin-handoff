import bpy, math, mathutils

bpy.ops.wm.open_mainfile(filepath="/Users/manaswi/Downloads/gondolaaa11/mesh_output/gondola.blend")

obj = bpy.data.objects["gondola_mesh"]

# bounding box info
bbox = [obj.matrix_world @ mathutils.Vector(c) for c in obj.bound_box]
xs = [v.x for v in bbox]; ys = [v.y for v in bbox]; zs = [v.z for v in bbox]
print("BOUNDS X:", min(xs), max(xs))
print("BOUNDS Y:", min(ys), max(ys))
print("BOUNDS Z:", min(zs), max(zs))
print("VERTS:", len(obj.data.vertices), "FACES:", len(obj.data.polygons))

# set up simple render
scene = bpy.context.scene
scene.render.engine = 'BLENDER_EEVEE'
scene.render.resolution_x = 800
scene.render.resolution_y = 600

# camera
cam_data = bpy.data.cameras.new("cam")
cam_obj = bpy.data.objects.new("cam", cam_data)
scene.collection.objects.link(cam_obj)
scene.camera = cam_obj

center = mathutils.Vector(((min(xs)+max(xs))/2, (min(ys)+max(ys))/2, (min(zs)+max(zs))/2))
size = max(max(xs)-min(xs), max(ys)-min(ys), max(zs)-min(zs))
cam_obj.location = center + mathutils.Vector((size*1.5, -size*1.5, size*1.0))
direction = center - cam_obj.location
rot_quat = direction.to_track_quat('-Z', 'Y')
cam_obj.rotation_euler = rot_quat.to_euler()

# basic light
light_data = bpy.data.lights.new("sun", type='SUN')
light_obj = bpy.data.objects.new("sun", light_data)
scene.collection.objects.link(light_obj)
light_obj.rotation_euler = (0.7, 0, 0.7)

# shade flat, add solid material for visibility
mat = bpy.data.materials.new("m")
mat.diffuse_color = (0.6,0.6,0.9,1)
obj.data.materials.append(mat)

scene.render.filepath = "/Users/manaswi/Downloads/gondolaaa11/mesh_output/mesh_preview.png"
bpy.ops.render.render(write_still=True)
