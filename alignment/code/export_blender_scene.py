from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

import bpy
from mathutils import Vector


CUBE_TRIANGLES = [
    [0, 1, 2],
    [0, 2, 3],
    [4, 6, 5],
    [4, 7, 6],
    [0, 4, 5],
    [0, 5, 1],
    [1, 5, 6],
    [1, 6, 2],
    [2, 6, 7],
    [2, 7, 3],
    [3, 7, 4],
    [3, 4, 0],
]


def parse_args() -> argparse.Namespace:
    env_results = os.getenv("SHELF_MAPPER_RESULTS_JSON")
    env_output = os.getenv("SHELF_MAPPER_BLEND_OUTPUT")
    if env_results and env_output:
        return argparse.Namespace(
            results=env_results,
            output=env_output,
            dense_mesh=os.getenv("SHELF_MAPPER_DENSE_MESH", ""),
        )

    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else sys.argv[1:]
    parser = argparse.ArgumentParser(description="Export Shelf Mapper results to a Blender file.")
    parser.add_argument("--results", required=True, help="Path to Shelf Mapper results.json")
    parser.add_argument("--output", required=True, help="Path to the .blend file to write")
    parser.add_argument("--dense-mesh", default="", help="Optional dense mesh .ply to import")
    return parser.parse_args(argv)


def safe_name(value: str, fallback: str = "Object") -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9_. -]+", "_", str(value or "")).strip()
    return cleaned[:63] or fallback


def hex_to_rgba(value: str | None, alpha: float = 1.0) -> tuple[float, float, float, float]:
    value = (value or "#ffffff").lstrip("#")
    if len(value) != 6:
        return (1.0, 1.0, 1.0, alpha)
    return (
        int(value[0:2], 16) / 255.0,
        int(value[2:4], 16) / 255.0,
        int(value[4:6], 16) / 255.0,
        alpha,
    )


def make_material(name: str, color: str, alpha: float = 1.0):
    mat = bpy.data.materials.new(safe_name(name, "Material"))
    rgba = hex_to_rgba(color, alpha)
    mat.diffuse_color = rgba
    mat.blend_method = "BLEND" if alpha < 0.99 else "OPAQUE"
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    if bsdf:
        if "Base Color" in bsdf.inputs:
            bsdf.inputs["Base Color"].default_value = rgba
        if "Alpha" in bsdf.inputs:
            bsdf.inputs["Alpha"].default_value = alpha
    return mat


def get_collection(name: str):
    collection = bpy.data.collections.new(name)
    bpy.context.scene.collection.children.link(collection)
    return collection


def add_mesh_object(name: str, vertices: list, triangles: list, material, collection, props: dict | None = None):
    if not vertices or not triangles:
        return None
    mesh = bpy.data.meshes.new(f"{safe_name(name)}Mesh")
    mesh.from_pydata(vertices, [], triangles)
    mesh.update(calc_edges=True)
    obj = bpy.data.objects.new(safe_name(name), mesh)
    collection.objects.link(obj)
    obj.data.materials.append(material)
    for key, value in (props or {}).items():
        obj[key] = value
    return obj


def cube_vertices(center: list[float], size: float) -> list[list[float]]:
    cx, cy, cz = center
    r = size * 0.5
    return [
        [cx - r, cy - r, cz - r],
        [cx + r, cy - r, cz - r],
        [cx + r, cy + r, cz - r],
        [cx - r, cy + r, cz - r],
        [cx - r, cy - r, cz + r],
        [cx + r, cy - r, cz + r],
        [cx + r, cy + r, cz + r],
        [cx - r, cy + r, cz + r],
    ]


def import_dense_mesh(mesh_path: Path, collection, material, scale: float = 1.0) -> None:
    if not mesh_path.exists():
        return

    before = set(bpy.data.objects)
    try:
        if hasattr(bpy.ops.wm, "ply_import"):
            bpy.ops.wm.ply_import(filepath=str(mesh_path))
        else:
            bpy.ops.preferences.addon_enable(module="io_mesh_ply")
            bpy.ops.import_mesh.ply(filepath=str(mesh_path))
    except Exception as exc:
        print(f"WARNING: Could not import dense mesh {mesh_path}: {exc}")
        return

    imported = [obj for obj in bpy.data.objects if obj not in before]
    for obj in imported:
        obj.name = safe_name(f"Dense scan - {obj.name}", "Dense scan")
        if collection.name not in {item.name for item in obj.users_collection}:
            try:
                collection.objects.link(obj)
            except RuntimeError:
                pass
        if material and hasattr(obj.data, "materials"):
            obj.data.materials.append(material)
        obj.scale = (scale, scale, scale)


def add_scene_geometry(results: dict, dense_mesh_path: Path | None) -> list[list[float]]:
    rack_material = make_material("Rack material", "#d1d5db", 0.96)
    shelf_material = make_material("Shelf material", "#9ca3af", 1.0)
    product_material = make_material("Product marker material", "#f6c453", 0.96)
    dense_material = make_material("Dense scan material", "#8ac3e5", 0.38)

    rack_collection = get_collection("Rack")
    shelf_collection = get_collection("Shelves")
    product_collection = get_collection("Product markers")
    dense_collection = get_collection("Dense scan")

    all_points: list[list[float]] = []

    for part in (results.get("rack") or {}).get("parts", []):
        style = part.get("style") or {}
        material = make_material(
            f"{part.get('id', 'rack')} material",
            style.get("color", "#d1d5db"),
            float(style.get("opacity", 0.96)),
        )
        obj = add_mesh_object(
            part.get("id", "rack-part"),
            part.get("vertices", []),
            part.get("triangles", []),
            material,
            rack_collection,
            {"shelf_mapper_kind": part.get("kind", "rack_part")},
        )
        if obj:
            all_points.extend(part.get("vertices", []))

    for shelf in results.get("shelves", []):
        mesh = shelf.get("mesh") or {}
        obj = add_mesh_object(
            shelf.get("id", "shelf"),
            mesh.get("vertices", []),
            mesh.get("triangles", []),
            shelf_material,
            shelf_collection,
            {
                "shelf_mapper_kind": "shelf",
                "shelf_id": shelf.get("id", ""),
                "product_count": int(shelf.get("product_count", 0) or 0),
            },
        )
        if obj:
            all_points.extend(mesh.get("vertices", []))

    for product in results.get("products", []):
        point = product.get("projection_point") or product.get("p3d")
        if not point:
            continue
        label = product.get("display_label") or product.get("label") or product.get("id") or "product"
        obj = add_mesh_object(
            f"Product - {label}",
            cube_vertices([float(point[0]), float(point[1]), float(point[2])], 0.025),
            CUBE_TRIANGLES,
            product_material,
            product_collection,
            {
                "shelf_mapper_kind": "product_marker",
                "product_id": product.get("id", ""),
                "product_label": label,
                "shelf_id": product.get("shelf_id", ""),
                "source_image": product.get("image_name", ""),
            },
        )
        if obj:
            all_points.append(point)

    if dense_mesh_path:
        scale = float((results.get("layout") or {}).get("scale_factor") or 1.0)
        import_dense_mesh(dense_mesh_path, dense_collection, dense_material, scale)

    return all_points


def add_camera_and_light(points: list[list[float]]) -> None:
    if points:
        xs = [float(point[0]) for point in points]
        ys = [float(point[1]) for point in points]
        zs = [float(point[2]) for point in points]
        center = Vector((
            (min(xs) + max(xs)) * 0.5,
            (min(ys) + max(ys)) * 0.5,
            (min(zs) + max(zs)) * 0.5,
        ))
        radius = max(
            max(xs) - min(xs),
            max(ys) - min(ys),
            max(zs) - min(zs),
            1.0,
        )
    else:
        center = Vector((0.0, 0.0, 0.0))
        radius = 1.0

    camera_data = bpy.data.cameras.new("Shelf Mapper Camera")
    camera = bpy.data.objects.new("Shelf Mapper Camera", camera_data)
    bpy.context.scene.collection.objects.link(camera)
    camera.location = center + Vector((radius * 1.35, -radius * 1.9, radius * 1.1))
    direction = center - camera.location
    camera.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()
    camera_data.lens = 28
    bpy.context.scene.camera = camera

    light_data = bpy.data.lights.new("Shelf Mapper Key Light", type="AREA")
    light = bpy.data.objects.new("Shelf Mapper Key Light", light_data)
    bpy.context.scene.collection.objects.link(light)
    light.location = center + Vector((radius * 0.4, -radius * 0.7, radius * 1.6))
    light_data.energy = 500
    light_data.size = max(radius * 0.7, 1.0)


def main() -> None:
    args = parse_args()
    results_path = Path(args.results)
    output_path = Path(args.output)
    dense_mesh_path = Path(args.dense_mesh) if args.dense_mesh else None

    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()

    results = json.loads(results_path.read_text(encoding="utf-8"))
    points = add_scene_geometry(results, dense_mesh_path)
    add_camera_and_light(points)

    try:
        bpy.context.scene.render.engine = "BLENDER_EEVEE_NEXT"
    except Exception:
        try:
            bpy.context.scene.render.engine = "BLENDER_EEVEE"
        except Exception:
            pass
    output_path.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=str(output_path))


if __name__ == "__main__":
    main()
