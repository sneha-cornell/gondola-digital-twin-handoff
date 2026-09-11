import bpy

print("BLEND_FILE", bpy.data.filepath)
print("OBJECT_COUNT", len(bpy.data.objects))
print("MESH_COUNT", len(bpy.data.meshes))

for obj in bpy.data.objects:
    mesh = obj.data if obj.type == "MESH" else None
    if mesh is None:
        print(f"OBJECT {obj.name} type={obj.type}")
        continue

    materials = [slot.material.name for slot in obj.material_slots if slot.material]
    print(
        "MESH_OBJECT "
        f"name={obj.name} "
        f"mesh={mesh.name} "
        f"vertices={len(mesh.vertices)} "
        f"edges={len(mesh.edges)} "
        f"polygons={len(mesh.polygons)} "
        f"materials={materials}"
    )
