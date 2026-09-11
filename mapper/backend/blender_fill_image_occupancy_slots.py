"""
Fill blank product slots only inside image-derived occupied shelf regions.

This pass starts from the strict existing-shelves Blender file and placement
JSON. It uses original detection bounding boxes to estimate occupied intervals
per shelf row, maps those image intervals back to Blender shelf coordinates,
and adds blank placeholders only inside those intervals.
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


SHELF_RE = re.compile(r"^Unit\d+_Shelf_\d+$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--job-dir", required=True)
    parser.add_argument("--placements-json", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--collection", default="Image_Occupancy_Blank_Slots")
    parser.add_argument("--slot-spacing", type=float, default=0.14)
    parser.add_argument("--edge-padding", type=float, default=0.025)
    parser.add_argument("--occupied-threshold", type=float, default=0.075)
    parser.add_argument("--product-height", type=float, default=0.22)
    parser.add_argument("--product-width", type=float, default=0.12)
    parser.add_argument("--product-depth", type=float, default=0.055)
    parser.add_argument("--min-fit-points", type=int, default=3)
    parser.add_argument("--replace", action="store_true")
    return parser.parse_args(sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else [])


def load_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}


def load_detections(job_dir: Path) -> dict[str, dict]:
    detections = {}
    for path in sorted((job_dir / "detections").glob("*.json")):
        payload = load_json(path)
        for detection in payload.get("detections", []):
            detections[str(detection["id"])] = detection
    return detections


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
    return shelf_name.split("_", 1)[0]


def horizontal_axis(shelf_name: str) -> str:
    return "x" if unit_for(shelf_name) in {"Unit1", "Unit3"} else "y"


def facing_for(shelf_name: str) -> str:
    unit = unit_for(shelf_name)
    if unit == "Unit1":
        return "-Y"
    if unit == "Unit3":
        return "+Y"
    if unit == "Unit2":
        return "+X"
    if unit == "Unit4":
        return "-X"
    return "-Y"


def placement_location(placement: dict) -> Vector:
    x, y, z = placement["location"]
    return Vector((float(x), float(y), float(z)))


def horizontal_coord(shelf_name: str, point: Vector) -> float:
    return point.x if horizontal_axis(shelf_name) == "x" else point.y


def shelf_depth_coord(shelf_name: str, shelf_obj: bpy.types.Object) -> float:
    xb, yb, _ = world_bounds(shelf_obj)
    facing = facing_for(shelf_name)
    if facing == "-Y":
        return yb[0]
    if facing == "+Y":
        return yb[1]
    if facing == "+X":
        return xb[1]
    return xb[0]


def fit_image_x_to_shelf_coord(pairs: list[tuple[float, float]]) -> tuple[float, float] | None:
    if len(pairs) < 2:
        return None
    xs = [item[0] for item in pairs]
    ys = [item[1] for item in pairs]
    mean_x = statistics.mean(xs)
    mean_y = statistics.mean(ys)
    denom = sum((x - mean_x) ** 2 for x in xs)
    if denom <= 1e-6:
        return None
    slope = sum((x - mean_x) * (y - mean_y) for x, y in pairs) / denom
    intercept = mean_y - slope * mean_x
    return slope, intercept


def merge_image_intervals(detections: list[dict]) -> list[tuple[float, float]]:
    if not detections:
        return []
    boxes = [tuple(float(v) for v in detection["bbox"]) for detection in detections]
    widths = [max(1.0, x2 - x1) for x1, _y1, x2, _y2 in boxes]
    typical_width = statistics.median(widths)
    max_gap = max(typical_width * 1.35, 35.0)

    intervals = sorted((x1, x2) for x1, _y1, x2, _y2 in boxes)
    merged: list[list[float]] = []
    for start, end in intervals:
        if not merged or start - merged[-1][1] > max_gap:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    return [(start, end) for start, end in merged]


def slot_coords(interval: tuple[float, float], spacing: float, padding: float) -> list[float]:
    start = interval[0] + padding
    end = interval[1] - padding
    if end <= start:
        return []
    count = max(1, int(math.floor((end - start) / spacing)) + 1)
    if count == 1:
        return [(start + end) * 0.5]
    actual_spacing = (end - start) / float(count - 1)
    return [start + actual_spacing * idx for idx in range(count)]


def is_slot_occupied(slot: float, occupied: list[float], threshold: float) -> bool:
    return any(abs(slot - coord) <= threshold for coord in occupied)


def create_blank_box(
    name: str,
    shelf_name: str,
    slot: float,
    depth_coord: float,
    shelf_top_z: float,
    collection: bpy.types.Collection,
    material: bpy.types.Material,
    product_width: float,
    product_depth: float,
    product_height: float,
) -> bpy.types.Object:
    if horizontal_axis(shelf_name) == "x":
        base = Vector((slot, depth_coord, shelf_top_z))
    else:
        base = Vector((depth_coord, slot, shelf_top_z))

    facing = facing_for(shelf_name)
    if facing in {"-Y", "+Y"}:
        dims = (product_width, product_depth, product_height)
        offset = Vector((0, -product_depth * 0.5 if facing == "-Y" else product_depth * 0.5, product_height * 0.5 + 0.012))
    else:
        dims = (product_depth, product_width, product_height)
        offset = Vector((product_depth * 0.5 if facing == "+X" else -product_depth * 0.5, 0, product_height * 0.5 + 0.012))

    bpy.ops.mesh.primitive_cube_add(size=1.0, location=base + offset)
    obj = bpy.context.object
    obj.name = name
    obj.dimensions = dims
    bpy.context.view_layer.update()
    obj.data.materials.append(material)
    obj["product_name"] = ""
    obj["product_identity_status"] = "image_occupancy_blank_slot"
    obj["shelf_object"] = shelf_name
    obj["image_layout_replica"] = True
    link_to_collection(obj, collection)
    return obj


def clamp_interval(interval: tuple[float, float], shelf_obj: bpy.types.Object, shelf_name: str) -> tuple[float, float] | None:
    xb, yb, _ = world_bounds(shelf_obj)
    bounds = xb if horizontal_axis(shelf_name) == "x" else yb
    start = max(min(interval), bounds[0])
    end = min(max(interval), bounds[1])
    if end - start < 0.05:
        return None
    return start, end


def main() -> None:
    args = parse_args()
    job_dir = Path(args.job_dir)
    detections_by_id = load_detections(job_dir)
    payload = json.loads(Path(args.placements_json).read_text(encoding="utf-8"))
    placements = payload["placements"] if isinstance(payload, dict) else payload

    shelves = {
        obj.name: obj
        for obj in bpy.data.objects
        if obj.type == "MESH" and SHELF_RE.match(obj.name)
    }

    by_shelf_image: dict[tuple[str, str], list[dict]] = defaultdict(list)
    occupied_by_shelf: dict[str, list[float]] = defaultdict(list)
    for placement in placements:
        shelf_name = str(placement.get("shelf_object") or "")
        detection = detections_by_id.get(str(placement.get("detection_id") or ""))
        if shelf_name not in shelves or detection is None:
            continue
        point = placement_location(placement)
        coord = horizontal_coord(shelf_name, point)
        occupied_by_shelf[shelf_name].append(coord)
        image_name = str(placement.get("image_name") or detection.get("image_name") or "")
        enriched = dict(detection)
        enriched["_shelf_coord"] = coord
        by_shelf_image[(shelf_name, image_name)].append(enriched)

    collection = collection_for(args.collection, args.replace)
    blank_mat = make_material("ImageOccupancyBlankProduct_NeutralGray", (0.72, 0.76, 0.72, 0.70))

    created = 0
    intervals_used = 0
    skipped_fits = 0
    for (shelf_name, _image_name), group in sorted(by_shelf_image.items()):
        if len(group) < args.min_fit_points:
            continue
        pairs = [
            ((float(item["bbox"][0]) + float(item["bbox"][2])) * 0.5, float(item["_shelf_coord"]))
            for item in group
        ]
        fit = fit_image_x_to_shelf_coord(pairs)
        if fit is None:
            skipped_fits += 1
            continue
        slope, intercept = fit
        shelf_obj = shelves[shelf_name]
        depth_coord = shelf_depth_coord(shelf_name, shelf_obj)
        _, _, zb = world_bounds(shelf_obj)
        for image_interval in merge_image_intervals(group):
            world_interval = (
                slope * image_interval[0] + intercept,
                slope * image_interval[1] + intercept,
            )
            clamped = clamp_interval(world_interval, shelf_obj, shelf_name)
            if clamped is None:
                continue
            intervals_used += 1
            for slot in slot_coords(clamped, args.slot_spacing, args.edge_padding):
                if is_slot_occupied(slot, occupied_by_shelf[shelf_name], args.occupied_threshold):
                    continue
                create_blank_box(
                    f"ImageOccBlank_{created:04d}_{shelf_name}",
                    shelf_name,
                    slot,
                    depth_coord,
                    zb[1],
                    collection,
                    blank_mat,
                    args.product_width,
                    args.product_depth,
                    args.product_height,
                )
                created += 1

    bpy.ops.wm.save_as_mainfile(filepath=args.output)
    print("SHELF_IMAGE_GROUPS", len(by_shelf_image))
    print("INTERVALS_USED", intervals_used)
    print("SKIPPED_FITS", skipped_fits)
    print("BLANK_SLOTS_CREATED", created)
    print("SAVED", args.output)


if __name__ == "__main__":
    main()
