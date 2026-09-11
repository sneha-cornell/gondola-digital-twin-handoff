"""
Fresh Blender placement pass from images + COLMAP + an existing rack .blend.

This script does not read an old placements JSON. It reads:
- detection JSON files from a job directory
- product-name JSON files when available
- COLMAP cameras/images/points3D text files
- shelf meshes already present in the opened Blender file

Run from Blender:

blender ../shelf_22-2.blend --background --python blender_project_from_images_colmap.py -- \
  --job-dir workspace/iphone16-2_named_high_recall_v4_job \
  --colmap-text-dir workspace/iphone16-2/text \
  --placements-json workspace/iphone16-2_named_high_recall_v4_job/blender_fresh_placements.json \
  --output /tmp/shelf_22-2_fresh_image_replica.blend \
  --product-boxes
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path

import bpy
from mathutils import Matrix, Vector


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--job-dir", required=True)
    parser.add_argument("--colmap-text-dir", required=True)
    parser.add_argument("--placements-json", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--collection", default="Fresh_Image_COLMAP_Placements")
    parser.add_argument("--cluster-radius", type=float, default=0.075)
    parser.add_argument("--product-height", type=float, default=0.22)
    parser.add_argument("--product-depth", type=float, default=0.055)
    parser.add_argument("--product-boxes", action="store_true")
    parser.add_argument(
        "--infer-missing-shelves",
        action="store_true",
        help="Create new shelf boards for detections that do not hit existing shelf meshes.",
    )
    parser.add_argument("--replace", action="store_true")
    return parser.parse_args(sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else [])


def qvec_to_matrix(qvec: list[float]) -> Matrix:
    qw, qx, qy, qz = qvec
    return Matrix(
        (
            (1 - 2 * qy * qy - 2 * qz * qz, 2 * qx * qy - 2 * qw * qz, 2 * qx * qz + 2 * qw * qy),
            (2 * qx * qy + 2 * qw * qz, 1 - 2 * qx * qx - 2 * qz * qz, 2 * qy * qz - 2 * qw * qx),
            (2 * qx * qz - 2 * qw * qy, 2 * qy * qz + 2 * qw * qx, 1 - 2 * qx * qx - 2 * qy * qy),
        )
    )


def load_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}


def load_camera(text_dir: Path) -> dict:
    for line in (text_dir / "cameras.txt").read_text().splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        parts = line.split()
        model = parts[1]
        params = [float(value) for value in parts[4:]]
        if model in {"SIMPLE_PINHOLE", "SIMPLE_RADIAL", "SIMPLE_RADIAL_FISHEYE"}:
            return {"fx": params[0], "fy": params[0], "cx": params[1], "cy": params[2]}
        if model in {"PINHOLE", "OPENCV", "OPENCV_FISHEYE"}:
            return {"fx": params[0], "fy": params[1], "cx": params[2], "cy": params[3]}
        raise RuntimeError(f"Unsupported COLMAP camera model: {model}")
    raise RuntimeError(f"No camera found in {text_dir / 'cameras.txt'}")


def load_images(text_dir: Path) -> dict[str, dict]:
    lines = [
        line
        for line in (text_dir / "images.txt").read_text().splitlines()
        if not line.lstrip().startswith("#")
    ]
    images: dict[str, dict] = {}
    i = 0
    while i < len(lines):
        header = lines[i].strip()
        obs_line = lines[i + 1] if i + 1 < len(lines) else ""
        i += 2
        if not header:
            continue
        parts = header.split(maxsplit=9)
        if len(parts) < 10:
            continue
        qvec = [float(value) for value in parts[1:5]]
        tvec = Vector((float(parts[5]), float(parts[6]), float(parts[7])))
        rotation = qvec_to_matrix(qvec)
        center = -(rotation.transposed() @ tvec)
        observations = []
        obs_parts = obs_line.split()
        for obs_idx in range(0, len(obs_parts) - 2, 3):
            observations.append(
                (
                    float(obs_parts[obs_idx]),
                    float(obs_parts[obs_idx + 1]),
                    int(obs_parts[obs_idx + 2]),
                )
            )
        images[parts[9]] = {"rotation": rotation, "center": center, "observations": observations}
    return images


def load_points3d(text_dir: Path) -> dict[int, Vector]:
    points = {}
    for line in (text_dir / "points3D.txt").read_text().splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        parts = line.split()
        points[int(parts[0])] = Vector((float(parts[1]), float(parts[2]), float(parts[3])))
    return points


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


def product_name_for(job_dir: Path, detection: dict) -> tuple[str, float, bool]:
    names = load_json(job_dir / "product_names" / f"{detection['id']}.json")
    product_name = (
        names.get("product_name")
        or names.get("catalog_candidate")
        or names.get("candidate")
        or detection.get("recognized_product_name")
    )
    confidence = float(names.get("catalog_confidence") or names.get("confidence") or 0.0)
    if isinstance(product_name, str) and product_name.strip():
        return product_name.strip(), confidence, False
    return "Unknown", confidence, True


def iter_detections(job_dir: Path):
    for path in sorted((job_dir / "detections").glob("*.json")):
        payload = load_json(path)
        for detection in payload.get("detections", []):
            item = dict(detection)
            product_name, name_confidence, is_unknown = product_name_for(job_dir, item)
            item["product_name"] = product_name
            item["name_confidence"] = name_confidence
            item["product_identity_status"] = "unknown" if is_unknown else "recognized"
            item["crop_file"] = resolve_path(
                item.get("crop_path") or item.get("raw_crop_path"),
                job_dir,
            )
            yield item


def ray_for_pixel(image: dict, camera: dict, u: float, v: float) -> tuple[Vector, Vector]:
    direction_camera = Vector(
        ((u - camera["cx"]) / camera["fx"], (v - camera["cy"]) / camera["fy"], 1.0)
    ).normalized()
    direction_world = (image["rotation"].transposed() @ direction_camera).normalized()
    return image["center"], direction_world


def rays_for_detection(detection: dict, camera: dict, images: dict[str, dict]) -> list[tuple[Vector, Vector]]:
    image = images.get(detection.get("image_name"))
    if image is None:
        return []
    x1, y1, x2, y2 = detection["bbox"]
    cx = 0.5 * (x1 + x2)
    cy = 0.5 * (y1 + y2)
    anchors = [
        (cx, y2),
        (x1, y2),
        (x2, y2),
        (cx, cy),
        (x1, cy),
        (x2, cy),
        (cx, y1 + 0.75 * (y2 - y1)),
        (x1 + 0.25 * (x2 - x1), y2),
        (x1 + 0.75 * (x2 - x1), y2),
    ]
    return [ray_for_pixel(image, camera, u, v) for u, v in anchors]


def object_bounds(obj: bpy.types.Object):
    corners = [obj.matrix_world @ Vector(corner) for corner in obj.bound_box]
    xs = [corner.x for corner in corners]
    ys = [corner.y for corner in corners]
    zs = [corner.z for corner in corners]
    return (min(xs), max(xs)), (min(ys), max(ys)), (min(zs), max(zs))


def shelf_hit(origin: Vector, direction: Vector, shelf_objects: list[bpy.types.Object]) -> dict | None:
    best = None
    for shelf_obj in shelf_objects:
        inv = shelf_obj.matrix_world.inverted()
        local_origin = inv @ origin
        local_direction = (inv.to_3x3() @ direction).normalized()
        hit, loc, normal, _face_index = shelf_obj.ray_cast(local_origin, local_direction, distance=100.0)
        if not hit:
            continue
        world_loc = shelf_obj.matrix_world @ loc
        distance = (world_loc - origin).length
        if distance <= 0:
            continue
        if best is None or distance < best["distance"]:
            best = {
                "distance": distance,
                "location": world_loc,
                "normal": (shelf_obj.matrix_world.to_3x3() @ normal).normalized(),
                "shelf": shelf_obj,
                "source": "ray",
            }
    return best


def observed_points_in_bbox(detection: dict, images: dict[str, dict], points3d: dict[int, Vector]) -> list[Vector]:
    image = images.get(detection.get("image_name"))
    if image is None:
        return []
    x1, y1, x2, y2 = detection["bbox"]
    points = []
    for obs_x, obs_y, point_id in image["observations"]:
        if point_id >= 0 and x1 <= obs_x <= x2 and y1 <= obs_y <= y2 and point_id in points3d:
            points.append(points3d[point_id])
    return points


def shelf_from_observed_points(
    detection: dict,
    images: dict[str, dict],
    points3d: dict[int, Vector],
    shelf_objects: list[bpy.types.Object],
    min_points: int = 3,
    tolerance: float = 0.16,
) -> dict | None:
    points = observed_points_in_bbox(detection, images, points3d)
    if len(points) < min_points:
        return None
    center = sum(points, Vector()) / len(points)
    best = None
    for shelf_obj in shelf_objects:
        xb, yb, zb = object_bounds(shelf_obj)
        if not (xb[0] - tolerance <= center.x <= xb[1] + tolerance):
            continue
        if not (yb[0] - tolerance <= center.y <= yb[1] + tolerance):
            continue
        top_z = zb[1]
        if center.z < top_z - 0.08:
            continue
        projected = Vector((
            min(max(center.x, xb[0]), xb[1]),
            min(max(center.y, yb[0]), yb[1]),
            top_z,
        ))
        score = (abs(center.z - top_z), (projected - center).length)
        if best is None or score < best["score"]:
            best = {
                "distance": score[1],
                "location": projected,
                "normal": Vector((0, 0, 1)),
                "shelf": shelf_obj,
                "score": score,
                "source": "observed_points",
                "observed_point_count": len(points),
            }
    return best


def cluster_hits(hits: list[dict], radius: float) -> list[dict]:
    clusters = []
    for hit in hits:
        assigned = None
        for cluster in clusters:
            if cluster["shelf_name"] != hit["shelf_name"]:
                continue
            center = sum((member["location"] for member in cluster["members"]), Vector()) / len(cluster["members"])
            if (hit["location"] - center).length <= radius:
                assigned = cluster
                break
        if assigned is None:
            clusters.append({"shelf_name": hit["shelf_name"], "members": [hit]})
        else:
            assigned["members"].append(hit)
    return clusters


def choose_member(cluster: dict) -> dict:
    return max(
        cluster["members"],
        key=lambda hit: (
            float(hit["detection"].get("name_confidence") or 0.0),
            float(hit["detection"].get("confidence") or 0.0),
        ),
    )


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


def observed_center(detection: dict, images: dict[str, dict], points3d: dict[int, Vector], min_points: int = 3) -> Vector | None:
    points = observed_points_in_bbox(detection, images, points3d)
    if len(points) < min_points:
        return None
    return sum(points, Vector()) / len(points)


def create_inferred_shelf(
    index: int,
    centers: list[Vector],
    collection: bpy.types.Collection,
    thickness: float = 0.045,
    padding: float = 0.18,
) -> bpy.types.Object:
    xs = [point.x for point in centers]
    ys = [point.y for point in centers]
    zs = [point.z for point in centers]
    x0, x1 = min(xs) - padding, max(xs) + padding
    y0, y1 = min(ys) - padding, max(ys) + padding
    top_z = min(zs)
    center = ((x0 + x1) * 0.5, (y0 + y1) * 0.5, top_z - thickness * 0.5)
    bpy.ops.mesh.primitive_cube_add(size=1.0, location=center)
    shelf = bpy.context.object
    shelf.name = f"Inferred_Shelf_{index:02d}"
    shelf.dimensions = (max(x1 - x0, 0.35), max(y1 - y0, 0.25), thickness)
    bpy.context.view_layer.update()
    shelf.data.materials.append(make_material("InferredMissingShelf_LightGray", (0.82, 0.86, 0.90, 1.0)))
    shelf["generated_from_colmap_unplaced_products"] = True
    link_to_collection(shelf, collection)
    return shelf


def hits_from_unplaced_candidates(
    candidates: list[dict],
    collection: bpy.types.Collection,
    z_radius: float = 0.14,
) -> list[dict]:
    groups: list[list[dict]] = []
    for candidate in candidates:
        assigned = None
        for group in groups:
            group_z = sum(item["center"].z for item in group) / len(group)
            if abs(candidate["center"].z - group_z) <= z_radius:
                assigned = group
                break
        if assigned is None:
            groups.append([candidate])
        else:
            assigned.append(candidate)

    inferred_hits = []
    for index, group in enumerate(groups):
        shelf = create_inferred_shelf(index, [item["center"] for item in group], collection)
        _, _, zb = object_bounds(shelf)
        for item in group:
            center = item["center"]
            location = Vector((center.x, center.y, zb[1]))
            inferred_hits.append(
                {
                    "distance": 0.0,
                    "location": location,
                    "normal": Vector((0, 0, 1)),
                    "shelf": shelf,
                    "source": "inferred_missing_shelf",
                    "detection": item["detection"],
                    "shelf_name": shelf.name,
                }
            )
    return inferred_hits


def material_for_detection(detection: dict):
    if detection["product_identity_status"] == "unknown":
        return make_material("UnknownProduct_BlankLightGray", (0.88, 0.88, 0.84, 1.0)), None
    crop_file = detection.get("crop_file")
    if not crop_file or not Path(crop_file).exists():
        return make_material("ProductNoCrop_Neutral", (0.65, 0.70, 0.75, 1.0)), None
    image = bpy.data.images.load(str(crop_file), check_existing=True)
    safe_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(detection["id"]))
    mat = bpy.data.materials.new(f"FreshCropMat_{safe_id}")
    mat.use_nodes = True
    mat.blend_method = "BLEND"
    tex = mat.node_tree.nodes.new("ShaderNodeTexImage")
    tex.image = image
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    if bsdf:
        mat.node_tree.links.new(tex.outputs["Color"], bsdf.inputs["Base Color"])
    return mat, image


def unit_facing(shelf_name: str) -> str:
    unit = shelf_name.split("_", 1)[0]
    if unit == "Unit1":
        return "-Y"
    if unit == "Unit3":
        return "+Y"
    if unit == "Unit2":
        return "+X"
    if unit == "Unit4":
        return "-X"
    return "-Y"


def create_product_plane(index: int, hit: dict, collection: bpy.types.Collection, height: float):
    detection = hit["detection"]
    material, image = material_for_detection(detection)
    aspect = image.size[0] / max(1, image.size[1]) if image else 0.7
    width = max(0.055, min(0.42, height * aspect))
    facing = unit_facing(hit["shelf_name"])
    if facing in {"-Y", "+Y"}:
        verts = [(-width / 2, 0, 0), (width / 2, 0, 0), (width / 2, 0, height), (-width / 2, 0, height)]
        offset = Vector((0, -0.018 if facing == "-Y" else 0.018, 0.012))
    else:
        verts = [(0, -width / 2, 0), (0, width / 2, 0), (0, width / 2, height), (0, -width / 2, height)]
        offset = Vector((0.018 if facing == "+X" else -0.018, 0, 0.012))

    safe_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(detection["id"]))
    mesh = bpy.data.meshes.new(f"FreshProduct_{index:04d}_{safe_id}_mesh")
    mesh.from_pydata(verts, [], [(0, 1, 2, 3)])
    mesh.update()
    mesh.uv_layers.new(name="UVMap")
    for loop, uv in zip(mesh.uv_layers.active.data, [(0, 0), (1, 0), (1, 1), (0, 1)]):
        loop.uv = uv
    obj = bpy.data.objects.new(f"FreshProduct_{index:04d}_{safe_id}", mesh)
    collection.objects.link(obj)
    obj.location = hit["location"] + offset
    obj.data.materials.append(material)
    obj["source_detection_id"] = detection["id"]
    obj["source_image"] = detection.get("image_name") or ""
    obj["product_name"] = detection["product_name"]
    obj["product_identity_status"] = detection["product_identity_status"]
    obj["source_crop"] = str(detection.get("crop_file") or "")
    obj["shelf_object"] = hit["shelf_name"]
    obj["cluster_view_count"] = hit["cluster_view_count"]
    obj["placement_source"] = hit.get("source", "unknown")
    if detection["product_identity_status"] == "unknown":
        add_unknown_label(index, safe_id, obj.location, facing, width, height, collection)
    return obj


def add_unknown_label(index: int, safe_id: str, location: Vector, facing: str, width: float, height: float, collection):
    curve = bpy.data.curves.new(f"FreshUnknownLabelCurve_{index:04d}_{safe_id}", "FONT")
    curve.body = "Unknown"
    curve.align_x = "CENTER"
    curve.align_y = "CENTER"
    curve.size = min(0.05, max(0.028, width * 0.24))
    label = bpy.data.objects.new(f"FreshUnknownLabel_{index:04d}_{safe_id}", curve)
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
    label.data.materials.append(make_material("UnknownProduct_LabelDark", (0.05, 0.05, 0.05, 1.0)))


def create_product_box(index: int, hit: dict, collection, height: float, depth: float, mat):
    facing = unit_facing(hit["shelf_name"])
    width = 0.14
    if facing in {"-Y", "+Y"}:
        dims = (width, depth, height)
        offset = Vector((0, -depth * 0.5 if facing == "-Y" else depth * 0.5, height * 0.5 + 0.012))
    else:
        dims = (depth, width, height)
        offset = Vector((depth * 0.5 if facing == "+X" else -depth * 0.5, 0, height * 0.5 + 0.012))
    detection = hit["detection"]
    safe_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(detection["id"]))
    bpy.ops.mesh.primitive_cube_add(size=1.0, location=hit["location"] + offset)
    obj = bpy.context.object
    obj.name = f"FreshProductBox_{index:04d}_{safe_id}"
    obj.dimensions = dims
    bpy.context.view_layer.update()
    obj.data.materials.append(mat)
    obj["product_name"] = detection["product_name"]
    obj["product_identity_status"] = detection["product_identity_status"]
    obj["shelf_object"] = hit["shelf_name"]
    for coll in list(obj.users_collection):
        coll.objects.unlink(obj)
    collection.objects.link(obj)


def main() -> None:
    args = parse_args()
    job_dir = Path(args.job_dir)
    text_dir = Path(args.colmap_text_dir)
    camera = load_camera(text_dir)
    images = load_images(text_dir)
    points3d = load_points3d(text_dir)
    shelf_objects = [obj for obj in bpy.data.objects if obj.type == "MESH" and "_Shelf_" in obj.name]
    if not shelf_objects:
        raise RuntimeError("No shelf mesh objects found. Expected object names containing '_Shelf_'.")
    collection = collection_for(args.collection, args.replace)

    hits = []
    unplaced_candidates = []
    total = 0
    ray_hits = 0
    observed_hits = 0
    unknown_detections = 0
    for detection in iter_detections(job_dir):
        total += 1
        if detection["product_identity_status"] == "unknown":
            unknown_detections += 1
        hit = None
        for origin, direction in rays_for_detection(detection, camera, images):
            candidate = shelf_hit(origin, direction, shelf_objects)
            if candidate:
                hit = candidate
                ray_hits += 1
                break
        if hit is None:
            hit = shelf_from_observed_points(detection, images, points3d, shelf_objects)
            if hit:
                observed_hits += 1
        if hit is None:
            center = observed_center(detection, images, points3d)
            if center is not None:
                unplaced_candidates.append({"detection": detection, "center": center})
            continue
        hit["detection"] = detection
        hit["shelf_name"] = hit["shelf"].name
        hits.append(hit)

    inferred_hits = hits_from_unplaced_candidates(unplaced_candidates, collection) if args.infer_missing_shelves else []
    hits.extend(inferred_hits)
    clusters = cluster_hits(hits, args.cluster_radius)
    chosen_hits = []
    for cluster in clusters:
        hit = choose_member(cluster)
        hit["cluster_view_count"] = len(cluster["members"])
        chosen_hits.append(hit)

    box_mat = make_material("FreshProductBox_TransparentGreen", (0.1, 0.75, 0.28, 0.38))
    placements = []
    for index, hit in enumerate(chosen_hits):
        obj = create_product_plane(index, hit, collection, args.product_height)
        if args.product_boxes:
            create_product_box(index, hit, collection, args.product_height, args.product_depth, box_mat)
        detection = hit["detection"]
        placements.append(
            {
                "object": obj.name,
                "detection_id": detection["id"],
                "image_name": detection.get("image_name"),
                "product_name": detection["product_name"],
                "product_identity_status": detection["product_identity_status"],
                "crop_file": str(detection.get("crop_file") or ""),
                "shelf_object": hit["shelf_name"],
                "location": [float(hit["location"].x), float(hit["location"].y), float(hit["location"].z)],
                "cluster_view_count": hit["cluster_view_count"],
                "placement_source": hit.get("source", "unknown"),
            }
        )

    Path(args.placements_json).write_text(
        json.dumps(
            {
                "method": "fresh image detections + COLMAP camera rays intersected with Blender shelf meshes",
                "detections_with_crops_or_boxes": total,
                "unknown_detections": unknown_detections,
                "shelf_hits": len(hits),
                "direct_ray_hits": ray_hits,
                "observed_point_fallback_hits": observed_hits,
                "inferred_missing_shelf_hits": len(inferred_hits),
                "clustered_products": len(placements),
                "placements": placements,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    bpy.ops.wm.save_as_mainfile(filepath=args.output)
    print("DETECTIONS_READ", total)
    print("UNKNOWN_DETECTIONS", unknown_detections)
    print("SHELF_HITS", len(hits))
    print("CLUSTERED_PRODUCTS", len(placements))
    print("SAVED", args.output)


if __name__ == "__main__":
    main()
