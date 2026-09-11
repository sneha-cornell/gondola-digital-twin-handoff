"""
Populate an existing Blender rack from image-derived product placements.

Run from Blender, for example:

blender ../shelf_22-2.blend --background --python blender_replicate_image_shelf.py -- \
  --placements-json workspace/iphone16-2_named_high_recall_v4_job/blender_colmap_exact_placements_v2.json \
  --output /tmp/shelf_22-2_image_replica.blend \
  --product-boxes

The script preserves the existing rack. If a placement references a shelf object
that is missing from the .blend, it creates a matching shelf board before placing
the products.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path

import bpy
from mathutils import Vector


SHELF_RE = re.compile(r"^(?P<unit>.+)_Shelf_(?P<index>\d+)$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--placements-json", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--collection", default="Image_Shelf_Replica")
    parser.add_argument("--product-height", type=float, default=0.22)
    parser.add_argument("--product-depth", type=float, default=0.055)
    parser.add_argument("--shelf-thickness", type=float, default=0.045)
    parser.add_argument("--shelf-padding", type=float, default=0.18)
    parser.add_argument("--product-boxes", action="store_true")
    parser.add_argument("--replace", action="store_true")
    return parser.parse_args(sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else [])


def world_bounds(obj: bpy.types.Object) -> tuple[tuple[float, float], tuple[float, float], tuple[float, float]]:
    corners = [obj.matrix_world @ Vector(corner) for corner in obj.bound_box]
    xs = [corner.x for corner in corners]
    ys = [corner.y for corner in corners]
    zs = [corner.z for corner in corners]
    return (min(xs), max(xs)), (min(ys), max(ys)), (min(zs), max(zs))


def shelf_unit(shelf_name: str) -> str:
    match = SHELF_RE.match(shelf_name)
    return match.group("unit") if match else shelf_name.split("_", 1)[0]


def unit_facing(unit: str) -> str:
    # Existing tools use Unit1/Unit3 as front/back and Unit2/Unit4 as side racks.
    if unit == "Unit1":
        return "-Y"
    if unit == "Unit3":
        return "+Y"
    if unit == "Unit2":
        return "+X"
    if unit == "Unit4":
        return "-X"
    return "-Y"


def make_material(name: str, color: tuple[float, float, float, float]) -> bpy.types.Material:
    existing = bpy.data.materials.get(name)
    if existing:
        return existing
    mat = bpy.data.materials.new(name)
    mat.diffuse_color = color
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    if bsdf:
        bsdf.inputs["Base Color"].default_value = color
        bsdf.inputs["Alpha"].default_value = color[3]
    mat.blend_method = "BLEND"
    return mat


def image_material(name: str, image_path: Path) -> tuple[bpy.types.Material, bpy.types.Image | None]:
    safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", name)
    mat = bpy.data.materials.get(safe_name) or bpy.data.materials.new(safe_name)
    mat.use_nodes = True
    mat.blend_method = "BLEND"
    nodes = mat.node_tree.nodes
    bsdf = nodes.get("Principled BSDF")
    image = None
    if image_path.exists():
        image = bpy.data.images.load(str(image_path), check_existing=True)
        tex = nodes.new("ShaderNodeTexImage")
        tex.image = image
        if bsdf:
            mat.node_tree.links.new(tex.outputs["Color"], bsdf.inputs["Base Color"])
    elif bsdf:
        bsdf.inputs["Base Color"].default_value = (0.75, 0.75, 0.75, 1.0)
    return mat, image


def unknown_material() -> bpy.types.Material:
    return make_material("UnknownProduct_BlankLightGray", (0.88, 0.88, 0.84, 1.0))


def collection_for(name: str, replace: bool) -> bpy.types.Collection:
    existing = bpy.data.collections.get(name)
    if existing and replace:
        for obj in list(existing.objects):
            bpy.data.objects.remove(obj, do_unlink=True)
        bpy.data.collections.remove(existing)
        existing = None
    if existing:
        return existing
    collection = bpy.data.collections.new(name)
    bpy.context.scene.collection.children.link(collection)
    return collection


def link_to_collection(obj: bpy.types.Object, collection: bpy.types.Collection) -> None:
    for coll in list(obj.users_collection):
        coll.objects.unlink(obj)
    collection.objects.link(obj)


def existing_shelves_by_name() -> dict[str, bpy.types.Object]:
    return {
        obj.name: obj
        for obj in bpy.data.objects
        if obj.type == "MESH" and "_Shelf_" in obj.name
    }


def placement_location(placement: dict) -> Vector:
    x, y, z = placement["location"]
    return Vector((float(x), float(y), float(z)))


def infer_missing_shelf_bounds(
    shelf_name: str,
    placements: list[dict],
    shelves: dict[str, bpy.types.Object],
    shelf_padding: float,
    shelf_thickness: float,
) -> tuple[tuple[float, float], tuple[float, float], tuple[float, float]]:
    unit = shelf_unit(shelf_name)
    sibling_bounds = [
        world_bounds(obj)
        for name, obj in shelves.items()
        if shelf_unit(name) == unit
    ]
    product_points = [placement_location(item) for item in placements]
    if product_points:
        xs = [point.x for point in product_points]
        ys = [point.y for point in product_points]
        zs = [point.z for point in product_points]
    else:
        xs = ys = [0.0]
        zs = [0.0]

    if sibling_bounds:
        widths = [bounds[0][1] - bounds[0][0] for bounds in sibling_bounds]
        depths = [bounds[1][1] - bounds[1][0] for bounds in sibling_bounds]
        width = max(widths)
        depth = max(depths)
        center_x = sum((bounds[0][0] + bounds[0][1]) * 0.5 for bounds in sibling_bounds) / len(sibling_bounds)
        center_y = sum((bounds[1][0] + bounds[1][1]) * 0.5 for bounds in sibling_bounds) / len(sibling_bounds)
    else:
        width = max(max(xs) - min(xs) + shelf_padding * 2.0, 0.8)
        depth = max(max(ys) - min(ys) + shelf_padding * 2.0, 0.28)
        center_x = (min(xs) + max(xs)) * 0.5
        center_y = (min(ys) + max(ys)) * 0.5

    facing = unit_facing(unit)
    if facing in {"-Y", "+Y"}:
        x_bounds = (center_x - width * 0.5, center_x + width * 0.5)
        y_center = center_y if sibling_bounds else (min(ys) + max(ys)) * 0.5
        y_bounds = (y_center - depth * 0.5, y_center + depth * 0.5)
    else:
        # Side faces use Y as the horizontal product direction. Reuse sibling
        # X depth when available, otherwise create a narrow side shelf.
        y_width = max(depth, max(ys) - min(ys) + shelf_padding * 2.0)
        x_depth = max(min(width, 0.5), 0.28)
        x_center = center_x if sibling_bounds else (min(xs) + max(xs)) * 0.5
        x_bounds = (x_center - x_depth * 0.5, x_center + x_depth * 0.5)
        y_bounds = (center_y - y_width * 0.5, center_y + y_width * 0.5)

    z = min(zs) if product_points else 0.0
    z_bounds = (z - shelf_thickness, z)
    return x_bounds, y_bounds, z_bounds


def create_shelf_board(
    shelf_name: str,
    bounds: tuple[tuple[float, float], tuple[float, float], tuple[float, float]],
    collection: bpy.types.Collection,
    material: bpy.types.Material,
) -> bpy.types.Object:
    (x0, x1), (y0, y1), (z0, z1) = bounds
    center = ((x0 + x1) * 0.5, (y0 + y1) * 0.5, (z0 + z1) * 0.5)
    bpy.ops.mesh.primitive_cube_add(size=1.0, location=center)
    shelf = bpy.context.object
    shelf.name = shelf_name
    shelf.dimensions = (max(x1 - x0, 0.01), max(y1 - y0, 0.01), max(z1 - z0, 0.01))
    bpy.context.view_layer.update()
    shelf.data.materials.append(material)
    shelf["generated_from_image_layout"] = True
    link_to_collection(shelf, collection)
    return shelf


def ensure_missing_shelves(
    placements: list[dict],
    collection: bpy.types.Collection,
    shelf_padding: float,
    shelf_thickness: float,
) -> dict[str, bpy.types.Object]:
    shelves = existing_shelves_by_name()
    by_shelf: dict[str, list[dict]] = defaultdict(list)
    for placement in placements:
        shelf_name = placement.get("shelf_object")
        if shelf_name:
            by_shelf[str(shelf_name)].append(placement)

    shelf_mat = make_material("GeneratedMissingShelf_LightGray", (0.82, 0.86, 0.90, 1.0))
    created = 0
    for shelf_name, shelf_placements in sorted(by_shelf.items()):
        if shelf_name in shelves:
            continue
        bounds = infer_missing_shelf_bounds(
            shelf_name,
            shelf_placements,
            shelves,
            shelf_padding=shelf_padding,
            shelf_thickness=shelf_thickness,
        )
        shelves[shelf_name] = create_shelf_board(shelf_name, bounds, collection, shelf_mat)
        created += 1
    print("MISSING_SHELVES_CREATED", created)
    return shelves


def product_width_from_image(image: bpy.types.Image | None, height: float) -> float:
    if image and image.size[1] > 0:
        return max(0.055, min(0.42, height * image.size[0] / image.size[1]))
    return 0.14


def product_plane_vertices(facing: str, width: float, height: float) -> tuple[list[tuple[float, float, float]], Vector]:
    if facing in {"-Y", "+Y"}:
        verts = [(-width / 2, 0, 0), (width / 2, 0, 0), (width / 2, 0, height), (-width / 2, 0, height)]
        offset = Vector((0.0, -0.018 if facing == "-Y" else 0.018, 0.012))
    else:
        verts = [(0, -width / 2, 0), (0, width / 2, 0), (0, width / 2, height), (0, -width / 2, height)]
        offset = Vector((0.018 if facing == "+X" else -0.018, 0.0, 0.012))
    return verts, offset


def create_product_plane(
    index: int,
    placement: dict,
    collection: bpy.types.Collection,
    product_height: float,
) -> bpy.types.Object:
    detection_id = str(placement.get("detection_id") or f"{index:04d}")
    safe_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", detection_id)
    product_name = str(placement.get("product_name") or "").strip()
    is_unknown = not product_name
    crop_file = Path(str(placement.get("crop_file") or ""))
    if is_unknown:
        material = unknown_material()
        image = None
    else:
        material, image = image_material(f"ReplicaCropMat_{safe_id}", crop_file)
    width = product_width_from_image(image, product_height)
    shelf_name = str(placement.get("shelf_object") or "")
    facing = unit_facing(shelf_unit(shelf_name))
    verts, offset = product_plane_vertices(facing, width, product_height)

    mesh = bpy.data.meshes.new(f"ReplicaProduct_{index:04d}_{safe_id}_mesh")
    mesh.from_pydata(verts, [], [(0, 1, 2, 3)])
    mesh.update()
    mesh.uv_layers.new(name="UVMap")
    for loop, uv in zip(mesh.uv_layers.active.data, [(0, 0), (1, 0), (1, 1), (0, 1)]):
        loop.uv = uv

    obj = bpy.data.objects.new(f"ReplicaProduct_{index:04d}_{safe_id}", mesh)
    collection.objects.link(obj)
    obj.location = placement_location(placement) + offset
    obj.data.materials.append(material)
    obj["source_detection_id"] = detection_id
    obj["source_image"] = placement.get("image_name") or ""
    obj["product_name"] = product_name or "Unknown"
    obj["product_identity_status"] = "unknown" if is_unknown else "recognized"
    obj["source_crop"] = str(crop_file)
    obj["shelf_object"] = shelf_name
    obj["cluster_view_count"] = int(placement.get("cluster_view_count") or 1)
    obj["placement_source"] = placement.get("placement_source") or ""
    obj["image_layout_replica"] = True
    if is_unknown:
        create_unknown_label(index, detection_id, obj.location, facing, width, product_height, collection)
    return obj


def create_unknown_label(
    index: int,
    detection_id: str,
    location: Vector,
    facing: str,
    width: float,
    height: float,
    collection: bpy.types.Collection,
) -> bpy.types.Object:
    safe_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", detection_id)
    curve = bpy.data.curves.new(f"UnknownLabelCurve_{index:04d}_{safe_id}", "FONT")
    curve.body = "Unknown"
    curve.align_x = "CENTER"
    curve.align_y = "CENTER"
    curve.size = min(0.05, max(0.028, width * 0.24))
    label = bpy.data.objects.new(f"UnknownLabel_{index:04d}_{safe_id}", curve)
    collection.objects.link(label)
    label.location = location + Vector((0.0, 0.0, height * 0.52))
    if facing == "-Y":
        label.rotation_euler = (math.radians(90), 0.0, 0.0)
    elif facing == "+Y":
        label.rotation_euler = (math.radians(90), 0.0, math.pi)
    elif facing == "+X":
        label.rotation_euler = (math.radians(90), 0.0, math.radians(90))
    else:
        label.rotation_euler = (math.radians(90), 0.0, math.radians(-90))
    label.data.materials.append(make_material("UnknownProduct_LabelDark", (0.05, 0.05, 0.05, 1.0)))
    label["product_name"] = "Unknown"
    label["product_identity_status"] = "unknown"
    return label


def create_product_box(
    index: int,
    placement: dict,
    collection: bpy.types.Collection,
    box_mat: bpy.types.Material,
    height: float,
    depth: float,
) -> bpy.types.Object:
    shelf_name = str(placement.get("shelf_object") or "")
    facing = unit_facing(shelf_unit(shelf_name))
    loc = placement_location(placement)
    width = 0.14
    if facing in {"-Y", "+Y"}:
        dims = (width, depth, height)
        offset = Vector((0.0, -depth * 0.5 if facing == "-Y" else depth * 0.5, height * 0.5 + 0.012))
    else:
        dims = (depth, width, height)
        offset = Vector((depth * 0.5 if facing == "+X" else -depth * 0.5, 0.0, height * 0.5 + 0.012))

    detection_id = str(placement.get("detection_id") or f"{index:04d}")
    safe_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", detection_id)
    bpy.ops.mesh.primitive_cube_add(size=1.0, location=loc + offset)
    obj = bpy.context.object
    obj.name = f"ReplicaProductBox_{index:04d}_{safe_id}"
    obj.dimensions = dims
    bpy.context.view_layer.update()
    obj.data.materials.append(box_mat)
    obj["source_detection_id"] = detection_id
    product_name = str(placement.get("product_name") or "").strip()
    obj["product_name"] = product_name or "Unknown"
    obj["product_identity_status"] = "recognized" if product_name else "unknown"
    obj["shelf_object"] = shelf_name
    obj["image_layout_replica"] = True
    link_to_collection(obj, collection)
    return obj


def main() -> None:
    args = parse_args()
    payload = json.loads(Path(args.placements_json).read_text(encoding="utf-8"))
    placements = payload["placements"] if isinstance(payload, dict) else payload

    collection = collection_for(args.collection, replace=args.replace)
    ensure_missing_shelves(
        placements,
        collection,
        shelf_padding=args.shelf_padding,
        shelf_thickness=args.shelf_thickness,
    )

    box_mat = make_material("ReplicaProductBox_TransparentGreen", (0.1, 0.75, 0.28, 0.38))
    placed = 0
    boxes = 0
    for index, placement in enumerate(placements):
        create_product_plane(index, placement, collection, args.product_height)
        placed += 1
        if args.product_boxes:
            create_product_box(index, placement, collection, box_mat, args.product_height, args.product_depth)
            boxes += 1

    bpy.ops.wm.save_as_mainfile(filepath=args.output)
    print("PLACEMENTS_USED", len(placements))
    print("PRODUCT_PLANES_CREATED", placed)
    print("PRODUCT_BOXES_CREATED", boxes)
    print("SAVED", args.output)


if __name__ == "__main__":
    main()
