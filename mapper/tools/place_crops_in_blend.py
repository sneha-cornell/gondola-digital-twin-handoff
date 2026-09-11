import argparse
import json
import math
import sys
from pathlib import Path

import bpy
from mathutils import Vector


DEFAULT_MAPPING = {
    "Unit1": "00037",
    "Unit2": "00025",
    "Unit3": "00012",
    "Unit4": "00058",
}


def parse_mapping(value: str) -> dict[str, str]:
    mapping = dict(DEFAULT_MAPPING)
    if not value:
        return mapping
    for item in value.split(","):
        unit, image_id = item.split("=", 1)
        mapping[unit.strip()] = image_id.strip().removesuffix(".jpg")
    return mapping


def world_bounds(obj) -> tuple[tuple[float, float], tuple[float, float], tuple[float, float]]:
    corners = [obj.matrix_world @ Vector(corner) for corner in obj.bound_box]
    xs = [corner.x for corner in corners]
    ys = [corner.y for corner in corners]
    zs = [corner.z for corner in corners]
    return (min(xs), max(xs)), (min(ys), max(ys)), (min(zs), max(zs))


def local_crop_path(path_value: str | None, repo_root: Path) -> Path | None:
    if not path_value:
        return None
    path = Path(path_value)
    if path.exists():
        return path
    marker = "product_detection1/"
    text = str(path_value)
    if marker in text:
        candidate = repo_root / text.split(marker, 1)[1]
        if candidate.exists():
            return candidate
    return None


def load_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}


def load_detections(job_dir: Path, image_id: str, repo_root: Path) -> list[dict]:
    payload = load_json(job_dir / "detections" / f"{image_id}.json")
    detections = payload.get("detections", [])
    enriched = []
    for detection in detections:
        det = dict(detection)
        name_payload = load_json(job_dir / "product_names" / f"{det['id']}.json")
        product_name = (
            name_payload.get("product_name")
            or name_payload.get("catalog_candidate")
            or name_payload.get("candidate")
            or det.get("display_label")
            or det.get("label")
        )
        det["product_name"] = product_name
        det["crop_file"] = local_crop_path(
            det.get("crop_path") or name_payload.get("selected_crop_path") or det.get("raw_crop_path"),
            repo_root,
        )
        if det["crop_file"] and det["crop_file"].exists():
            enriched.append(det)
    return enriched


def kmeans_1d(values: list[float], k: int, iterations: int = 30) -> list[int]:
    if not values:
        return []
    sorted_values = sorted(values)
    centers = [
        sorted_values[min(len(sorted_values) - 1, round((i + 0.5) * len(sorted_values) / k))]
        for i in range(k)
    ]
    assignments = [0] * len(values)
    for _ in range(iterations):
        assignments = [
            min(range(k), key=lambda idx: abs(value - centers[idx]))
            for value in values
        ]
        new_centers = []
        for idx in range(k):
            members = [value for value, assignment in zip(values, assignments) if assignment == idx]
            new_centers.append(sum(members) / len(members) if members else centers[idx])
        if all(abs(a - b) < 1e-3 for a, b in zip(centers, new_centers)):
            break
        centers = new_centers
    order = {old: new for new, old in enumerate(sorted(range(k), key=lambda idx: centers[idx]))}
    return [order[assignment] for assignment in assignments]


def make_image_material(name: str, image_path: Path):
    image = bpy.data.images.load(str(image_path), check_existing=True)
    material = bpy.data.materials.new(name)
    material.use_nodes = True
    material.blend_method = "BLEND"
    material.show_transparent_back = True
    nodes = material.node_tree.nodes
    bsdf = nodes.get("Principled BSDF")
    tex = nodes.new("ShaderNodeTexImage")
    tex.image = image
    material.node_tree.links.new(tex.outputs["Color"], bsdf.inputs["Base Color"])
    if "Alpha" in tex.outputs and "Alpha" in bsdf.inputs:
        material.node_tree.links.new(tex.outputs["Alpha"], bsdf.inputs["Alpha"])
    return material


def create_plane(name: str, center: Vector, width: float, height: float, facing: str, material):
    mesh = bpy.data.meshes.new(f"{name}_mesh")
    half_w = width / 2.0
    if facing in {"-Y", "+Y"}:
        verts = [
            (-half_w, 0.0, 0.0),
            (half_w, 0.0, 0.0),
            (half_w, 0.0, height),
            (-half_w, 0.0, height),
        ]
    else:
        verts = [
            (0.0, -half_w, 0.0),
            (0.0, half_w, 0.0),
            (0.0, half_w, height),
            (0.0, -half_w, height),
        ]
    mesh.from_pydata(verts, [], [(0, 1, 2, 3)])
    mesh.update()
    mesh.uv_layers.new(name="UVMap")
    uv = mesh.uv_layers.active.data
    for loop, coord in zip(uv, [(0, 0), (1, 0), (1, 1), (0, 1)]):
        loop.uv = coord
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.collection.objects.link(obj)
    obj.location = center
    obj.data.materials.append(material)
    if facing == "+Y":
        obj.rotation_euler.z = math.pi
    elif facing == "+X":
        obj.rotation_euler.z = math.pi
    return obj


def add_label(name: str, text: str, location: Vector, facing: str, size: float):
    font_curve = bpy.data.curves.new(name, "FONT")
    font_curve.body = text[:42]
    font_curve.align_x = "CENTER"
    font_curve.size = size
    obj = bpy.data.objects.new(name, font_curve)
    bpy.context.collection.objects.link(obj)
    obj.location = location
    if facing in {"-Y", "+Y"}:
        obj.rotation_euler.x = math.radians(75)
        if facing == "+Y":
            obj.rotation_euler.z = math.pi
    else:
        obj.rotation_euler.x = math.radians(75)
        obj.rotation_euler.z = math.radians(90 if facing == "-X" else -90)
    return obj


def shelf_objects_for_unit(unit: str):
    shelves = [
        obj for obj in bpy.data.objects
        if obj.type == "MESH" and obj.name.startswith(f"{unit}_Shelf_")
    ]
    return sorted(shelves, key=lambda obj: world_bounds(obj)[2][0])


def place_unit(unit: str, image_id: str, job_dir: Path, repo_root: Path, max_items: int, add_text: bool):
    shelves = shelf_objects_for_unit(unit)
    detections = load_detections(job_dir, image_id, repo_root)
    if not shelves or not detections:
        print(f"SKIP {unit}: shelves={len(shelves)} detections={len(detections)}")
        return 0

    centers_y = [(det["bbox"][1] + det["bbox"][3]) * 0.5 for det in detections]
    row_ranks = kmeans_1d(centers_y, len(shelves))
    by_row: dict[int, list[dict]] = {idx: [] for idx in range(len(shelves))}
    for det, top_down_rank in zip(detections, row_ranks):
        shelf_index = len(shelves) - 1 - top_down_rank
        by_row.setdefault(shelf_index, []).append(det)

    if unit in {"Unit1", "Unit3"}:
        facing = "-Y" if unit == "Unit1" else "+Y"
        axis = "x"
    else:
        facing = "+X" if unit == "Unit2" else "-X"
        axis = "y"

    placed = 0
    for shelf_index, shelf in enumerate(shelves):
        row = sorted(by_row.get(shelf_index, []), key=lambda det: (det["bbox"][0] + det["bbox"][2]) * 0.5)
        if not row:
            continue
        if max_items > 0:
            row = row[:max_items]

        xb, yb, zb = world_bounds(shelf)
        axis_bounds = xb if axis == "x" else yb
        centers_x = [(det["bbox"][0] + det["bbox"][2]) * 0.5 for det in row]
        min_px, max_px = min(centers_x), max(centers_x)
        if max_px <= min_px:
            max_px = min_px + 1.0

        for det in row:
            x1, y1, x2, y2 = det["bbox"]
            center_px = (x1 + x2) * 0.5
            t = (center_px - min_px) / (max_px - min_px)
            t = max(0.02, min(0.98, t))
            world_axis = axis_bounds[0] + t * (axis_bounds[1] - axis_bounds[0])

            crop_file = det["crop_file"]
            try:
                crop_image = bpy.data.images.load(str(crop_file), check_existing=True)
                aspect = crop_image.size[0] / max(1, crop_image.size[1])
            except Exception:
                aspect = max(0.45, min(1.8, (x2 - x1) / max(1.0, y2 - y1)))

            shelf_height = zb[1] - zb[0]
            next_top = shelves[min(shelf_index + 1, len(shelves) - 1)]
            next_z = world_bounds(next_top)[2][0] if shelf_index + 1 < len(shelves) else 2.95
            clear_height = max(0.16, next_z - zb[1] - 0.06)
            height = max(0.12, min(clear_height * 0.82, 0.16 + ((y2 - y1) / 3840.0) * 2.3))
            width = max(0.08, min((axis_bounds[1] - axis_bounds[0]) / max(1, len(row)) * 0.94, height * aspect))

            if axis == "x":
                y = yb[1] + 0.018 if facing == "-Y" else yb[0] - 0.018
                center = Vector((world_axis, y, zb[1] + 0.012))
            else:
                x = xb[1] + 0.018 if facing == "+X" else xb[0] - 0.018
                center = Vector((x, world_axis, zb[1] + 0.012))

            safe_id = det["id"].replace("-", "_")
            material = make_image_material(f"CropMat_{safe_id}", crop_file)
            obj = create_plane(f"Crop_{unit}_{det['id']}", center, width, height, facing, material)
            obj["source_image"] = f"{image_id}.jpg"
            obj["source_detection_id"] = det["id"]
            obj["product_name"] = det.get("product_name") or ""
            obj["source_crop"] = str(crop_file)
            if add_text and det.get("product_name"):
                label_z = center.z + height + 0.015
                add_label(f"Label_{unit}_{det['id']}", det["product_name"], Vector((center.x, center.y, label_z)), facing, 0.035)
            placed += 1

    print(f"PLACED {unit} image={image_id} count={placed}")
    return placed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--job-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--mapping", default="")
    parser.add_argument("--max-items-per-shelf", type=int, default=24)
    parser.add_argument("--labels", action="store_true")
    script_args = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    args = parser.parse_args(script_args)

    job_dir = Path(args.job_dir)
    repo_root = Path("/home/ec2-user/SageMaker/product_detection1")
    mapping = parse_mapping(args.mapping)

    collection = bpy.data.collections.new("Placed_Crops")
    bpy.context.scene.collection.children.link(collection)
    bpy.context.view_layer.active_layer_collection = bpy.context.view_layer.layer_collection.children[collection.name]

    total = 0
    for unit, image_id in mapping.items():
        total += place_unit(unit, image_id, job_dir, repo_root, args.max_items_per_shelf, args.labels)

    bpy.ops.wm.save_as_mainfile(filepath=args.output)
    print(f"SAVED {args.output} placed={total}")


if __name__ == "__main__":
    main()
