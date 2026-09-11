import bpy

# Clear default scene objects
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete(use_global=False)

bpy.ops.wm.ply_import(filepath="/Users/manaswi/Downloads/gondolaaa11/mesh_output/gondola_sparse_mesh.ply")

obj = bpy.context.selected_objects[0]
obj.name = "gondola_mesh"

bpy.ops.wm.save_as_mainfile(filepath="/Users/manaswi/Downloads/gondolaaa11/mesh_output/gondola.blend")
