"""
Populate rack shelves from one best image per rack face.

This pass deliberately avoids using ray-hit world coordinates for product
spacing. The previous projection pass is used only to decide which image best
sees each rack face. Inside that image, products are laid out by their 2D
bounding-box order and visual shelf rows.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import bpy
from mathutils import Vector


SHELF_RE = re.compile(r"^(Unit\d+)_Shelf_(\d+)$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--job-dir", required=True)
    parser.add_argument("--seed-placements-json", required=True)
    parser.add_argument("--placements-json", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--collection", default="Single_Best_Image_Layout")
    parser.add_argument("--product-height", type=float, default=0.22)
    parser.add_argument("--product-depth", type=float, default=0.055)
    parser.add_argument("--min-unit-image-count", type=int, default=8)
    parser.add_argument("--row-gap-factor", type=float, default=1.05)
    parser.add_argument("--region-padding", type=float, default=0.18)
    parser.add_argument("--shelf-edge-margin", type=float, default=0.08)
    parser.add_argument("--blank-gap-factor", type=float, default=1.85)
    parser.add_argument("--slot-spacing", type=float, default=0.16)
    parser.add_argument("--blank-width", type=float, default=0.11)
    parser.add_argument("--blank-height", type=float, default=0.18)
    parser.add_argument("--blank-depth", type=float, default=0.045)
    parser.add_argument("--occupied-padding", type=float, default=0.035)
    parser.add_argument(
        "--ordered-row-shelves",
        action="store_true",
        help="Map visual image rows to Blender shelves by top-to-bottom order instead of seed shelf guides.",
    )
    parser.add_argument("--no-fill-empty-slots", action="store_false", dest="fill_empty_slots")
    parser.set_defaults(fill_empty_slots=True)
    parser.add_argument("--replace", action="store_true")
    parser.add_argument(
        "--unit-image",
        action="append",
        default=[],
        metavar="UNIT=image.jpg[,x1,y1,x2,y2]",
        help="Force a specific image (and optional pixel region) for a unit, "
             "bypassing seed-based scoring. E.g. Unit2=00029.jpg or "
             "Unit2=00029.jpg,1200,0,2160,4032",
    )
    return parser.parse_args(sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else [])


def load_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}


def resolve_path(value: str | None, job_dir: Path) -> Path | None:
    if not value:
        return None
    path = Path(value)
    if path.exists():
        return path
    candidate = job_dir / path
    if candidate.exists():
        return candidate
    marker = "product_detection1/"
    text = str(value)
    if marker in text:
        repo_root = Path.cwd().parent
        candidate = repo_root / text.split(marker, 1)[1]
        if candidate.exists():
            return candidate
    return None


_CATALOG_CANDIDATE_MIN_CONF = 0.70  # below this, catalog_candidate is too weak to trust

def product_name_for(job_dir: Path, detection: dict) -> tuple[str, float, str]:
    names = load_json(job_dir / "product_names" / f"{detection['id']}.json")
    confidence = float(names.get("catalog_confidence") or names.get("confidence") or 0.0)
    product_name = names.get("product_name")
    if not product_name:
        # Use catalog_candidate only when confidence is high enough to be trustworthy
        candidate = names.get("catalog_candidate") or names.get("candidate")
        if candidate and confidence >= _CATALOG_CANDIDATE_MIN_CONF:
            product_name = candidate
        else:
            product_name = detection.get("recognized_product_name")
    if isinstance(product_name, str) and product_name.strip():
        return product_name.strip(), confidence, "recognized"
    return "Unknown", confidence, "unknown"


def load_detections(job_dir: Path) -> tuple[dict[str, dict], dict[str, list[dict]]]:
    by_id: dict[str, dict] = {}
    by_image: dict[str, list[dict]] = defaultdict(list)
    for path in sorted((job_dir / "detections").glob("*.json")):
        payload = load_json(path)
        for raw in payload.get("detections", []):
            detection = dict(raw)
            detection["id"] = str(detection["id"])
            product_name, confidence, status = product_name_for(job_dir, detection)
            detection["product_name"] = product_name
            detection["name_confidence"] = confidence
            detection["product_identity_status"] = status
            detection["crop_file"] = resolve_path(
                detection.get("crop_path") or detection.get("raw_crop_path"),
                job_dir,
            )
            by_id[detection["id"]] = detection
            by_image[str(detection.get("image_name") or path.stem + ".jpg")].append(detection)
    return by_id, by_image


def bbox(detection: dict) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = detection["bbox"]
    return float(x1), float(y1), float(x2), float(y2)


def bbox_center(detection: dict) -> tuple[float, float]:
    x1, y1, x2, y2 = bbox(detection)
    return 0.5 * (x1 + x2), 0.5 * (y1 + y2)


def bbox_width(detection: dict) -> float:
    x1, _y1, x2, _y2 = bbox(detection)
    return max(1.0, x2 - x1)


def bbox_height(detection: dict) -> float:
    _x1, y1, _x2, y2 = bbox(detection)
    return max(1.0, y2 - y1)


def world_bounds(obj: bpy.types.Object):
    corners = [obj.matrix_world @ Vector(corner) for corner in obj.bound_box]
    xs = [corner.x for corner in corners]
    ys = [corner.y for corner in corners]
    zs = [corner.z for corner in corners]
    return (min(xs), max(xs)), (min(ys), max(ys)), (min(zs), max(zs))


def make_material(name: str, color: tuple[float, float, float, float]) -> bpy.types.Material:
    mat = bpy.data.materials.get(name) or bpy.data.materials.new(name)
    mat.diffuse_color = color
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    if bsdf:
        bsdf.inputs["Base Color"].default_value = color
        bsdf.inputs["Alpha"].default_value = color[3]
    mat.blend_method = "BLEND"
    return mat


def crop_material(detection: dict):
    crop_file = detection.get("crop_file")
    if not crop_file or not Path(crop_file).exists():
        if detection["product_identity_status"] == "unknown":
            return make_material("LayoutUnknown_BlankLabel", (0.88, 0.88, 0.84, 1.0)), None
        return make_material("LayoutProduct_NoCrop", (0.62, 0.68, 0.72, 1.0)), None
    image = bpy.data.images.load(str(crop_file), check_existing=True)
    safe_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", detection["id"])
    mat = bpy.data.materials.new(f"LayoutCropMat_{safe_id}")
    mat.use_nodes = True
    mat.blend_method = "BLEND"
    tex = mat.node_tree.nodes.new("ShaderNodeTexImage")
    tex.image = image
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    if bsdf:
        mat.node_tree.links.new(tex.outputs["Color"], bsdf.inputs["Base Color"])
    return mat, image


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


def unit_for(shelf_name: str) -> str:
    match = SHELF_RE.match(shelf_name)
    return match.group(1) if match else shelf_name.split("_", 1)[0]


def facing_for_unit(unit: str) -> str:
    if unit == "Unit1":
        return "-Y"
    if unit == "Unit3":
        return "+Y"
    if unit == "Unit2":
        return "+X"
    if unit == "Unit4":
        return "-X"
    return "-Y"


def horizontal_axis(unit: str) -> str:
    return "x" if unit in {"Unit1", "Unit3"} else "y"


def shelf_depth_coord(unit: str, shelf_obj: bpy.types.Object) -> float:
    xb, yb, _zb = world_bounds(shelf_obj)
    facing = facing_for_unit(unit)
    if facing == "-Y":
        return yb[0]
    if facing == "+Y":
        return yb[1]
    if facing == "+X":
        return xb[1]
    return xb[0]


def shelf_horizontal_span(unit: str, shelf_obj: bpy.types.Object, margin: float) -> tuple[float, float]:
    xb, yb, _zb = world_bounds(shelf_obj)
    lo, hi = xb if horizontal_axis(unit) == "x" else yb
    return lo + margin, hi - margin


def shelf_top_z(shelf_obj: bpy.types.Object) -> float:
    _xb, _yb, zb = world_bounds(shelf_obj)
    return zb[1]


def score_unit_images(
    seed_placements: list[dict],
    detections_by_id: dict[str, dict],
    detections_by_image: dict[str, list[dict]],
    region_padding: float,
    row_gap_factor: float,
) -> dict[str, dict]:
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for placement in seed_placements:
        shelf_name = str(placement.get("shelf_object") or "")
        if not SHELF_RE.match(shelf_name):
            continue
        detection = detections_by_id.get(str(placement.get("detection_id") or ""))
        if not detection:
            continue
        grouped[(unit_for(shelf_name), str(placement.get("image_name") or detection.get("image_name")))].append(detection)

    best: dict[str, dict] = {}
    for (unit, image_name), detections in grouped.items():
        xs = []
        ys = []
        areas = []
        for detection in detections:
            x1, y1, x2, y2 = bbox(detection)
            xs.extend([x1, x2])
            ys.extend([y1, y2])
            areas.append((x2 - x1) * (y2 - y1))
        seed_region = expand_region(detections, region_padding)
        region_detections = detections_in_region(detections_by_image.get(image_name, []), seed_region)
        y_clusters = cluster_rows(region_detections, row_gap_factor)
        x_coverage = max(xs) - min(xs) if xs else 0.0
        y_coverage = max(ys) - min(ys) if ys else 0.0
        avg_area = statistics.mean(areas) if areas else 0.0
        score = (
            len(region_detections) * 10.0
            + len(y_clusters) * 20.0
            + len(detections) * 2.0
            + x_coverage / 90.0
            + y_coverage / 160.0
            + avg_area / 25000.0
        )
        item = {
            "unit": unit,
            "image_name": image_name,
            "seed_detection_count": len(detections),
            "region_detection_count": len(region_detections),
            "region_row_count": len(y_clusters),
            "score": score,
            "bbox_region": seed_region,
        }
        if unit not in best or score > best[unit]["score"]:
            best[unit] = item
    return best


def expand_region(detections: list[dict], padding_fraction: float) -> tuple[float, float, float, float]:
    xs = []
    ys = []
    for detection in detections:
        x1, y1, x2, y2 = bbox(detection)
        xs.extend([x1, x2])
        ys.extend([y1, y2])
    if not xs or not ys:
        return (0.0, 0.0, 2160.0, 3840.0)
    x1, x2 = min(xs), max(xs)
    y1, y2 = min(ys), max(ys)
    pad_x = (x2 - x1) * padding_fraction
    pad_y = (y2 - y1) * padding_fraction
    return x1 - pad_x, y1 - pad_y, x2 + pad_x, y2 + pad_y


def detections_in_region(detections: list[dict], region: tuple[float, float, float, float]) -> list[dict]:
    x1, y1, x2, y2 = region
    selected = []
    for detection in detections:
        cx, cy = bbox_center(detection)
        if x1 <= cx <= x2 and y1 <= cy <= y2:
            selected.append(detection)
    return selected


def _det_in_region(detection: dict, region: tuple) -> bool:
    x1, y1, x2, y2 = region
    cx, cy = bbox_center(detection)
    return x1 <= cx <= x2 and y1 <= cy <= y2


def cluster_rows(detections: list[dict], row_gap_factor: float) -> list[list[dict]]:
    if not detections:
        return []
    heights = [bbox_height(detection) for detection in detections]
    threshold = max(85.0, statistics.median(heights) * row_gap_factor)
    ordered = sorted(detections, key=lambda detection: bbox_center(detection)[1])
    clusters: list[list[dict]] = []
    centers: list[float] = []
    for detection in ordered:
        cy = bbox_center(detection)[1]
        if not clusters or abs(cy - centers[-1]) > threshold:
            clusters.append([detection])
            centers.append(cy)
        else:
            clusters[-1].append(detection)
            centers[-1] = statistics.mean([bbox_center(item)[1] for item in clusters[-1]])
    return [sorted(row, key=lambda detection: bbox_center(detection)[0]) for row in clusters]


def unit_shelves() -> dict[str, list[bpy.types.Object]]:
    grouped: dict[str, list[bpy.types.Object]] = defaultdict(list)
    for obj in bpy.data.objects:
        if obj.type != "MESH" or not SHELF_RE.match(obj.name):
            continue
        grouped[unit_for(obj.name)].append(obj)
    for unit in grouped:
        grouped[unit].sort(key=shelf_top_z, reverse=True)
    return grouped


def estimate_unit_reverse(unit: str, seed_placements: list[dict], detections_by_id: dict[str, dict]) -> bool:
    pairs = []
    axis = horizontal_axis(unit)
    for placement in seed_placements:
        shelf_name = str(placement.get("shelf_object") or "")
        if not shelf_name.startswith(unit + "_"):
            continue
        detection = detections_by_id.get(str(placement.get("detection_id") or ""))
        if not detection:
            continue
        image_x = bbox_center(detection)[0]
        location = placement.get("location") or [0, 0, 0]
        coord = float(location[0] if axis == "x" else location[1])
        pairs.append((image_x, coord))
    if len(pairs) < 3:
        return False
    mean_x = statistics.mean(x for x, _coord in pairs)
    mean_y = statistics.mean(coord for _x, coord in pairs)
    denom = sum((x - mean_x) ** 2 for x, _coord in pairs)
    if denom <= 1e-6:
        return False
    slope = sum((x - mean_x) * (coord - mean_y) for x, coord in pairs) / denom
    return slope < 0


def seed_row_guides(
    unit: str,
    image_name: str,
    seed_placements: list[dict],
    detections_by_id: dict[str, dict],
) -> list[tuple[float, str]]:
    guides = []
    for placement in seed_placements:
        shelf_name = str(placement.get("shelf_object") or "")
        if not shelf_name.startswith(unit + "_") or placement.get("image_name") != image_name:
            continue
        detection = detections_by_id.get(str(placement.get("detection_id") or ""))
        if not detection:
            continue
        _cx, cy = bbox_center(detection)
        guides.append((cy, shelf_name))
    return sorted(guides)


def shelf_for_visual_row(
    row: list[dict],
    row_index: int,
    shelves: list[bpy.types.Object],
    guides: list[tuple[float, str]],
    shelf_objects_by_name: dict[str, bpy.types.Object],
) -> bpy.types.Object:
    if not guides:
        return shelves[min(row_index, len(shelves) - 1)]
    y_values = [bbox_center(item)[1] for item in row]
    row_min = min(y_values)
    row_max = max(y_values)
    inside = [shelf_name for y, shelf_name in guides if row_min <= y <= row_max]
    if inside:
        counts = defaultdict(int)
        for shelf_name in inside:
            counts[shelf_name] += 1
        shelf_name = max(counts.items(), key=lambda item: (item[1], item[0]))[0]
        if shelf_name in shelf_objects_by_name:
            return shelf_objects_by_name[shelf_name]
    row_center = statistics.mean(y_values)
    nearest_shelf = min(guides, key=lambda item: abs(item[0] - row_center))[1]
    return shelf_objects_by_name.get(nearest_shelf, shelves[min(row_index, len(shelves) - 1)])


def shelf_for_ordered_visual_row(row_index: int, shelves: list[bpy.types.Object]) -> bpy.types.Object:
    return shelves[min(row_index, len(shelves) - 1)]


def create_blank_label(
    index: int,
    safe_id: str,
    location: Vector,
    facing: str,
    width: float,
    height: float,
    collection,
    body: str,
    min_size: float = 0.026,
) -> None:
    curve = bpy.data.curves.new(f"LayoutLabelCurve_{index:04d}_{safe_id}", "FONT")
    curve.body = body
    curve.align_x = "CENTER"
    curve.align_y = "CENTER"
    curve.size = min(0.052, max(min_size, width * 0.33))
    label = bpy.data.objects.new(f"LayoutLabel_{index:04d}_{safe_id}", curve)
    collection.objects.link(label)
    label.location = location + Vector((0, 0, height * 0.52))
    if facing == "-Y":
        label.rotation_euler = (math.radians(90), 0, 0)
    elif facing == "+Y":
        label.rotation_euler = (math.radians(90), 0, math.pi)
    elif facing == "+X":
        label.rotation_euler = (math.radians(90), 0, math.radians(90))
    else:
        label.rotation_euler = (math.radians(90), 0, math.radians(-90))
    label.data.materials.append(make_material("LayoutLabel_Dark", (0.04, 0.04, 0.04, 1.0)))


def location_for(unit: str, shelf_obj: bpy.types.Object, coord: float, product_depth: float, product_height: float) -> Vector:
    depth = shelf_depth_coord(unit, shelf_obj)
    top_z = shelf_top_z(shelf_obj)
    facing = facing_for_unit(unit)
    if horizontal_axis(unit) == "x":
        base = Vector((coord, depth, top_z))
    else:
        base = Vector((depth, coord, top_z))
    if facing in {"-Y", "+Y"}:
        offset = Vector((0, -product_depth * 0.5 if facing == "-Y" else product_depth * 0.5, product_height * 0.5 + 0.012))
    else:
        offset = Vector((product_depth * 0.5 if facing == "+X" else -product_depth * 0.5, 0, product_height * 0.5 + 0.012))
    return base + offset


def create_product_box(index: int, unit: str, shelf_obj: bpy.types.Object, detection: dict, coord: float, width: float, args, collection) -> bpy.types.Object:
    facing = facing_for_unit(unit)
    location = location_for(unit, shelf_obj, coord, args.product_depth, args.product_height)
    if facing in {"-Y", "+Y"}:
        dims = (width, args.product_depth, args.product_height)
    else:
        dims = (args.product_depth, width, args.product_height)
    bpy.ops.mesh.primitive_cube_add(size=1.0, location=location)
    obj = bpy.context.object
    safe_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", detection["id"])
    obj.name = f"LayoutProductBox_{index:04d}_{safe_id}"
    obj.dimensions = dims
    bpy.context.view_layer.update()
    if detection["product_identity_status"] in {"unknown", "empty_slot"}:
        mat = make_material("LayoutUnknown_Box", (0.88, 0.88, 0.84, 1.0))
    else:
        mat = make_material("LayoutProduct_BoxBlue", (0.38, 0.56, 0.72, 1.0))
    obj.data.materials.append(mat)
    link_to_collection(obj, collection)
    obj["source_detection_id"] = detection["id"]
    obj["source_image"] = detection.get("image_name") or ""
    obj["product_name"] = detection["product_name"]
    obj["product_identity_status"] = detection["product_identity_status"]
    obj["shelf_object"] = shelf_obj.name
    obj["placement_source"] = "single_best_image_layout"
    if detection["product_identity_status"] == "unknown":
        create_blank_label(index, safe_id, location, facing, width, args.product_height, collection, "Unknown")
    return obj


def create_product_plane(index: int, unit: str, shelf_obj: bpy.types.Object, detection: dict, coord: float, width: float, args, collection) -> bpy.types.Object:
    mat, image = crop_material(detection)
    facing = facing_for_unit(unit)
    height = args.product_height
    if image:
        aspect_width = max(0.055, min(width * 1.25, height * image.size[0] / max(1, image.size[1])))
    else:
        aspect_width = width
    if facing in {"-Y", "+Y"}:
        verts = [(-aspect_width / 2, 0, 0), (aspect_width / 2, 0, 0), (aspect_width / 2, 0, height), (-aspect_width / 2, 0, height)]
        offset = Vector((0, -0.018 if facing == "-Y" else 0.018, 0.012))
    else:
        verts = [(0, -aspect_width / 2, 0), (0, aspect_width / 2, 0), (0, aspect_width / 2, height), (0, -aspect_width / 2, height)]
        offset = Vector((0.018 if facing == "+X" else -0.018, 0, 0.012))
    base = location_for(unit, shelf_obj, coord, 0.0, 0.0) + offset
    safe_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", detection["id"])
    mesh = bpy.data.meshes.new(f"LayoutProduct_{index:04d}_{safe_id}_mesh")
    mesh.from_pydata(verts, [], [(0, 1, 2, 3)])
    mesh.update()
    mesh.uv_layers.new(name="UVMap")
    for loop, uv in zip(mesh.uv_layers.active.data, [(0, 0), (1, 0), (1, 1), (0, 1)]):
        loop.uv = uv
    obj = bpy.data.objects.new(f"LayoutProduct_{index:04d}_{safe_id}", mesh)
    collection.objects.link(obj)
    obj.location = base
    obj.data.materials.append(mat)
    obj["source_detection_id"] = detection["id"]
    obj["source_image"] = detection.get("image_name") or ""
    obj["product_name"] = detection["product_name"]
    obj["product_identity_status"] = detection["product_identity_status"]
    obj["shelf_object"] = shelf_obj.name
    obj["placement_source"] = "single_best_image_layout"
    return obj


def create_empty_slot(
    index: int,
    unit: str,
    shelf_obj: bpy.types.Object,
    coord: float,
    width: float,
    args,
    collection,
    status: str = "empty_slot",
) -> bpy.types.Object:
    facing = facing_for_unit(unit)
    location = location_for(unit, shelf_obj, coord, args.blank_depth, args.blank_height)
    if facing in {"-Y", "+Y"}:
        dims = (width, args.blank_depth, args.blank_height)
    else:
        dims = (args.blank_depth, width, args.blank_height)
    bpy.ops.mesh.primitive_cube_add(size=1.0, location=location)
    obj = bpy.context.object
    is_unknown_like = status in {"unknown", "inferred_unknown_slot"}
    obj.name = f"Layout{'UnknownSlot' if is_unknown_like else 'BlankSlot'}_{index:04d}"
    obj.dimensions = dims
    bpy.context.view_layer.update()
    if is_unknown_like:
        obj.data.materials.append(make_material("LayoutUnknown_Filler", (0.58, 0.66, 0.72, 0.86)))
    else:
        obj.data.materials.append(make_material("LayoutBlank_SoftGray", (0.78, 0.78, 0.74, 0.72)))
    link_to_collection(obj, collection)
    obj["product_name"] = "Unknown" if is_unknown_like else "Blank"
    obj["product_identity_status"] = status
    obj["shelf_object"] = shelf_obj.name
    obj["placement_source"] = (
        "single_best_image_layout_unknown_product_slot"
        if is_unknown_like
        else "single_best_image_layout_full_shelf_blank"
    )
    label = "Unknown" if is_unknown_like else "Blank"
    create_blank_label(index, f"{status}_{index:04d}", obj.location, facing, width, args.blank_height, collection, label, min_size=0.03)
    return obj


def coordinate_for(row_x: float, row_min: float, row_max: float, shelf_lo: float, shelf_hi: float, reverse: bool) -> float:
    if row_max <= row_min:
        norm = 0.5
    else:
        norm = (row_x - row_min) / (row_max - row_min)
    norm = max(0.0, min(1.0, norm))
    return shelf_hi - norm * (shelf_hi - shelf_lo) if reverse else shelf_lo + norm * (shelf_hi - shelf_lo)


def shelf_slots(shelf_obj: bpy.types.Object, unit: str, spacing: float, margin: float) -> list[float]:
    lo, hi = shelf_horizontal_span(unit, shelf_obj, margin)
    if hi <= lo:
        return []
    count = max(1, int(math.floor((hi - lo) / spacing)) + 1)
    if count == 1:
        return [(lo + hi) * 0.5]
    actual_spacing = (hi - lo) / float(count - 1)
    return [lo + idx * actual_spacing for idx in range(count)]


def is_coord_occupied(coord: float, occupied: list[tuple[float, float]], padding: float) -> bool:
    for occupied_coord, occupied_width in occupied:
        if abs(coord - occupied_coord) <= occupied_width * 0.5 + padding:
            return True
    return False


def fill_empty_shelf_slots(
    shelves_by_unit: dict[str, list[bpy.types.Object]],
    occupied_by_shelf: dict[str, list[tuple[float, float]]],
    source_units: set[str],
    args,
    collection,
    next_index: int,
) -> tuple[list[dict], int]:
    blanks = []
    for unit, shelves in sorted(shelves_by_unit.items()):
        for shelf_obj in shelves:
            occupied = occupied_by_shelf.get(shelf_obj.name, [])
            if not occupied:
                # No detections on this shelf — camera never captured it; skip phantom slots
                continue
            for coord in shelf_slots(shelf_obj, unit, args.slot_spacing, args.shelf_edge_margin):
                if is_coord_occupied(coord, occupied, args.occupied_padding):
                    continue
                status = "inferred_unknown_slot" if unit in source_units else "empty_slot"
                obj = create_empty_slot(next_index, unit, shelf_obj, coord, args.blank_width, args, collection, status=status)
                occupied.append((coord, args.blank_width))
                blanks.append(
                    {
                        "object": obj.name,
                        "product_name": "Unknown" if status == "inferred_unknown_slot" else "Blank",
                        "product_identity_status": status,
                        "shelf_object": shelf_obj.name,
                        "location": [float(obj.location.x), float(obj.location.y), float(obj.location.z)],
                        "placement_source": obj["placement_source"],
                    }
                )
                next_index += 1
    return blanks, next_index


def add_gap_blanks(row: list[dict], row_min: float, row_max: float, shelf_obj: bpy.types.Object, unit: str, reverse: bool, args, collection, next_index: int) -> tuple[list[dict], int]:
    if len(row) < 2:
        return [], next_index
    widths = [bbox_width(item) for item in row]
    typical = statistics.median(widths)
    blanks = []
    shelf_lo, shelf_hi = shelf_horizontal_span(unit, shelf_obj, args.shelf_edge_margin)
    intervals = sorted((bbox(item)[0], bbox(item)[2]) for item in row)
    for left, right in zip(intervals, intervals[1:]):
        gap_start = left[1]
        gap_end = right[0]
        gap = gap_end - gap_start
        if gap < typical * args.blank_gap_factor:
            continue
        count = max(1, int(gap // max(typical, 45.0)))
        for slot_idx in range(count):
            image_x = gap_start + (slot_idx + 1) * gap / (count + 1)
            coord = coordinate_for(image_x, row_min, row_max, shelf_lo, shelf_hi, reverse)
            obj = create_empty_slot(next_index, unit, shelf_obj, coord, 0.11, args, collection, status="inferred_unknown_slot")
            blanks.append(
                {
                    "object": obj.name,
                    "product_name": "Unknown",
                    "product_identity_status": "inferred_unknown_slot",
                    "shelf_object": shelf_obj.name,
                    "location": [float(obj.location.x), float(obj.location.y), float(obj.location.z)],
                    "placement_source": "single_best_image_layout_unknown_gap_slot",
                }
            )
            next_index += 1
    return blanks, next_index


def parse_unit_image_overrides(entries: list[str]) -> dict[str, dict]:
    """Parse --unit-image UNIT=image.jpg[,x1,y1,x2,y2] entries."""
    overrides: dict[str, dict] = {}
    for entry in entries:
        if "=" not in entry:
            continue
        unit, rest = entry.split("=", 1)
        parts = rest.split(",")
        image_name = parts[0].strip()
        if len(parts) == 5:
            try:
                region = tuple(float(p) for p in parts[1:5])
            except ValueError:
                region = None
        else:
            region = None
        overrides[unit.strip()] = {"image_name": image_name, "region": region}
    return overrides


def main() -> None:
    args = parse_args()
    job_dir = Path(args.job_dir)
    detections_by_id, detections_by_image = load_detections(job_dir)
    seed_payload = load_json(Path(args.seed_placements_json))
    seed_placements = seed_payload.get("placements", [])
    best_images = score_unit_images(
        seed_placements,
        detections_by_id,
        detections_by_image,
        args.region_padding,
        args.row_gap_factor,
    )

    unit_image_overrides = parse_unit_image_overrides(args.unit_image)
    for unit, override in unit_image_overrides.items():
        image_name = override["image_name"]
        region = override["region"] or (-99999.0, -99999.0, 99999.0, 99999.0)
        image_dets = detections_by_image.get(image_name, [])
        region_dets = [d for d in image_dets if _det_in_region(d, region)]
        best_images[unit] = {
            "unit": unit,
            "image_name": image_name,
            "seed_detection_count": max(args.min_unit_image_count, len(region_dets)),
            "region_detection_count": len(region_dets),
            "region_row_count": 0,
            "score": 99999.0,
            "bbox_region": list(region),
        }
        print(f"unit-image override: {unit} → {image_name} region={region} ({len(region_dets)} dets)")
    shelves_by_unit = unit_shelves()
    shelf_objects_by_name = {obj.name: obj for shelves in shelves_by_unit.values() for obj in shelves}
    collection = collection_for(args.collection, args.replace)

    placements = []
    occupied_by_shelf: dict[str, list[tuple[float, float]]] = defaultdict(list)
    next_index = 0
    skipped_units = []
    selected_units = {}

    for unit in sorted(shelves_by_unit):
        best = best_images.get(unit)
        if not best or best["seed_detection_count"] < args.min_unit_image_count:
            skipped_units.append({"unit": unit, "reason": "not_enough_seed_detections", "best": best})
            continue
        image_detections = detections_by_image.get(best["image_name"], [])
        region = tuple(best["bbox_region"])
        selected = detections_in_region(image_detections, region)
        if len(selected) < max(4, best["seed_detection_count"] // 2):
            seed_ids = {
                str(item.get("detection_id"))
                for item in seed_placements
                if str(item.get("shelf_object") or "").startswith(unit + "_") and item.get("image_name") == best["image_name"]
            }
            selected = [detections_by_id[item] for item in seed_ids if item in detections_by_id]
        rows = cluster_rows(selected, args.row_gap_factor)
        shelves = shelves_by_unit[unit]
        reverse = estimate_unit_reverse(unit, seed_placements, detections_by_id)
        guides = seed_row_guides(unit, best["image_name"], seed_placements, detections_by_id)
        selected_units[unit] = {
            "image_name": best["image_name"],
            "seed_detection_count": best["seed_detection_count"],
            "selected_detection_count": len(selected),
            "row_count": len(rows),
            "shelf_count": len(shelves),
            "reverse_image_x": reverse,
            "region": list(region),
            "seed_row_guides": [{"image_y": y, "shelf_object": shelf_name} for y, shelf_name in guides],
        }
        for row_index, row in enumerate(rows[: len(shelves)]):
            if args.ordered_row_shelves:
                shelf_obj = shelf_for_ordered_visual_row(row_index, shelves)
            else:
                shelf_obj = shelf_for_visual_row(row, row_index, shelves, guides, shelf_objects_by_name)
            shelf_lo, shelf_hi = shelf_horizontal_span(unit, shelf_obj, args.shelf_edge_margin)
            row_x1 = min(bbox(item)[0] for item in row)
            row_x2 = max(bbox(item)[2] for item in row)
            row_pad = statistics.median([bbox_width(item) for item in row]) * 0.45
            row_min = row_x1 - row_pad
            row_max = row_x2 + row_pad
            span = max(0.05, shelf_hi - shelf_lo)
            for detection in row:
                cx, _cy = bbox_center(detection)
                coord = coordinate_for(cx, row_min, row_max, shelf_lo, shelf_hi, reverse)
                width = max(0.075, min(0.22, bbox_width(detection) / max(1.0, row_max - row_min) * span * 0.92))
                box = create_product_box(next_index, unit, shelf_obj, detection, coord, width, args, collection)
                create_product_plane(next_index, unit, shelf_obj, detection, coord, width, args, collection)
                occupied_by_shelf[shelf_obj.name].append((coord, width))
                placements.append(
                    {
                        "object": box.name,
                        "detection_id": detection["id"],
                        "image_name": detection.get("image_name") or best["image_name"],
                        "product_name": detection["product_name"],
                        "product_identity_status": detection["product_identity_status"],
                        "crop_file": str(detection.get("crop_file") or ""),
                        "shelf_object": shelf_obj.name,
                        "location": [float(box.location.x), float(box.location.y), float(box.location.z)],
                        "image_bbox": [float(value) for value in detection["bbox"]],
                        "image_row_index": row_index,
                        "placement_source": "single_best_image_layout",
                    }
                )
                next_index += 1
            blanks, next_index = add_gap_blanks(row, row_min, row_max, shelf_obj, unit, reverse, args, collection, next_index)
            placements.extend(blanks)
            for blank in blanks:
                loc = blank.get("location") or [0, 0, 0]
                coord = float(loc[0] if horizontal_axis(unit) == "x" else loc[1])
                occupied_by_shelf[shelf_obj.name].append((coord, args.blank_width))

    if args.fill_empty_slots:
        blanks, next_index = fill_empty_shelf_slots(
            shelves_by_unit,
            occupied_by_shelf,
            set(selected_units),
            args,
            collection,
            next_index,
        )
        placements.extend(blanks)

    output_payload = {
        "method": "single best image per rack face; 2D row layout mapped to Blender shelf spans",
        "seed_placements_json": str(args.seed_placements_json),
        "selected_units": selected_units,
        "skipped_units": skipped_units,
        "placed_products": sum(1 for item in placements if item["product_identity_status"] in {"recognized", "unknown", "inferred_unknown_slot"}),
        "recognized_products": sum(1 for item in placements if item["product_identity_status"] == "recognized"),
        "unknown_products": sum(1 for item in placements if item["product_identity_status"] == "unknown"),
        "inferred_unknown_slots": sum(1 for item in placements if item["product_identity_status"] == "inferred_unknown_slot"),
        "empty_slots": sum(1 for item in placements if item["product_identity_status"] == "empty_slot"),
        "placements": placements,
    }
    Path(args.placements_json).write_text(json.dumps(output_payload, indent=2), encoding="utf-8")
    bpy.ops.wm.save_as_mainfile(filepath=args.output)
    print("SELECTED_UNITS", json.dumps(selected_units, sort_keys=True))
    print("PLACED_PRODUCTS", output_payload["placed_products"])
    print("UNKNOWN_PRODUCTS", output_payload["unknown_products"])
    print("EMPTY_SLOTS", output_payload["empty_slots"])
    print("OUTPUT", args.output)


if __name__ == "__main__":
    main()
