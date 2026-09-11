import argparse
import json
import math
import sys
from pathlib import Path

import bpy
from mathutils import Matrix, Vector


REPO = Path("/home/ec2-user/SageMaker/product_detection1")
TEXT_DIR = REPO / "backend/workspace/iphone16-2/text"
JOB_DIR = REPO / "backend/workspace/iphone16-2_named_high_recall_v4_job"


def qvec_to_matrix(qvec):
    qw, qx, qy, qz = qvec
    return Matrix(
        (
            (1 - 2 * qy * qy - 2 * qz * qz, 2 * qx * qy - 2 * qw * qz, 2 * qx * qz + 2 * qw * qy),
            (2 * qx * qy + 2 * qw * qz, 1 - 2 * qx * qx - 2 * qz * qz, 2 * qy * qz - 2 * qw * qx),
            (2 * qx * qz - 2 * qw * qy, 2 * qy * qz + 2 * qw * qx, 1 - 2 * qx * qx - 2 * qy * qy),
        )
    )


def load_camera():
    for line in (TEXT_DIR / "cameras.txt").read_text().splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        parts = line.split()
        return {"f": float(parts[4]), "cx": float(parts[5]), "cy": float(parts[6])}
    raise RuntimeError("No camera found")


def load_images():
    lines = [line for line in (TEXT_DIR / "images.txt").read_text().splitlines() if not line.lstrip().startswith("#")]
    images = {}
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
        if obs_line.strip():
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


def load_points3d():
    points = {}
    for line in (TEXT_DIR / "points3D.txt").read_text().splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        parts = line.split()
        points[int(parts[0])] = Vector((float(parts[1]), float(parts[2]), float(parts[3])))
    return points


def load_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}


def local_path(path_value):
    if not path_value:
        return None
    path = Path(path_value)
    if path.exists():
        return path
    marker = "product_detection1/"
    text = str(path_value)
    if marker in text:
        candidate = REPO / text.split(marker, 1)[1]
        if candidate.exists():
            return candidate
    return None


def iter_detections():
    for path in sorted((JOB_DIR / "detections").glob("*.json")):
        payload = load_json(path)
        for det in payload.get("detections", []):
            item = dict(det)
            names = load_json(JOB_DIR / "product_names" / f"{item['id']}.json")
            item["product_name"] = (
                names.get("product_name")
                or names.get("catalog_candidate")
                or names.get("candidate")
                or item.get("display_label")
                or item.get("label")
            )
            item["name_confidence"] = float(names.get("catalog_confidence") or names.get("confidence") or 0.0)
            item["crop_file"] = local_path(item.get("crop_path") or names.get("selected_crop_path") or item.get("raw_crop_path"))
            if item["crop_file"] and item["crop_file"].exists():
                yield item


def ray_for_pixel(image, camera, u, v):
    direction_camera = Vector(((u - camera["cx"]) / camera["f"], (v - camera["cy"]) / camera["f"], 1.0)).normalized()
    direction_world = (image["rotation"].transposed() @ direction_camera).normalized()
    return image["center"], direction_world


def rays_for_detection(det, camera, images):
    image = images.get(det["image_name"])
    if image is None:
        return []
    x1, _y1, x2, y2 = det["bbox"]
    y1 = det["bbox"][1]
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


def shelf_hit(origin, direction, shelf_objects):
    best = None
    for shelf_obj in shelf_objects:
        inv = shelf_obj.matrix_world.inverted()
        local_origin = inv @ origin
        local_direction = (inv.to_3x3() @ direction).normalized()
        hit, loc, normal, face_index = shelf_obj.ray_cast(local_origin, local_direction, distance=100.0)
        if not hit:
            continue
        world_loc = shelf_obj.matrix_world @ loc
        distance = (world_loc - origin).length
        if distance <= 0:
            continue
        if best is None or distance < best["distance"]:
            best = {"distance": distance, "location": world_loc, "normal": (shelf_obj.matrix_world.to_3x3() @ normal).normalized(), "shelf": shelf_obj}
    return best


def object_bounds(obj):
    corners = [obj.matrix_world @ Vector(corner) for corner in obj.bound_box]
    xs = [corner.x for corner in corners]
    ys = [corner.y for corner in corners]
    zs = [corner.z for corner in corners]
    return (min(xs), max(xs)), (min(ys), max(ys)), (min(zs), max(zs))


def observed_points_in_bbox(det, images, points3d):
    image = images.get(det["image_name"])
    if image is None:
        return []
    x1, y1, x2, y2 = det["bbox"]
    points = []
    for obs_x, obs_y, point_id in image["observations"]:
        if point_id < 0:
            continue
        if x1 <= obs_x <= x2 and y1 <= obs_y <= y2 and point_id in points3d:
            points.append(points3d[point_id])
    return points


def shelf_from_observed_points(det, images, points3d, shelf_objects, min_points=3, tolerance=0.16):
    points = observed_points_in_bbox(det, images, points3d)
    if len(points) < min_points:
        return None
    center = sum(points, Vector()) / len(points)
    best = None
    for shelf_obj in shelf_objects:
        xb, yb, zb = object_bounds(shelf_obj)
        in_xy = (
            xb[0] - tolerance <= center.x <= xb[1] + tolerance
            and yb[0] - tolerance <= center.y <= yb[1] + tolerance
        )
        if not in_xy:
            continue
        top_z = zb[1]
        if center.z < top_z - 0.08:
            continue
        vertical = abs(center.z - top_z)
        projected = Vector((
            min(max(center.x, xb[0]), xb[1]),
            min(max(center.y, yb[0]), yb[1]),
            top_z,
        ))
        score = (vertical, (projected - center).length)
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


def cluster_hits(hits, radius):
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


def choose_member(cluster):
    def score(hit):
        det = hit["detection"]
        return (float(det.get("name_confidence") or 0.0), float(det.get("confidence") or 0.0))

    return max(cluster["members"], key=score)


def material_for_crop(name, crop_file):
    image = bpy.data.images.load(str(crop_file), check_existing=True)
    material = bpy.data.materials.new(name)
    material.use_nodes = True
    material.blend_method = "BLEND"
    nodes = material.node_tree.nodes
    bsdf = nodes.get("Principled BSDF")
    tex = nodes.new("ShaderNodeTexImage")
    tex.image = image
    material.node_tree.links.new(tex.outputs["Color"], bsdf.inputs["Base Color"])
    return material, image


def create_product_plane(name, location, shelf_name, crop_file, height, material):
    image = bpy.data.images.load(str(crop_file), check_existing=True)
    aspect = image.size[0] / max(1, image.size[1])
    width = max(0.06, min(0.38, height * aspect))
    unit = shelf_name.split("_", 1)[0]
    if unit in {"Unit1", "Unit3"}:
        facing = "-Y" if unit == "Unit1" else "+Y"
        verts = [(-width / 2, 0, 0), (width / 2, 0, 0), (width / 2, 0, height), (-width / 2, 0, height)]
        offset = Vector((0, -0.018 if facing == "-Y" else 0.018, 0.01))
    else:
        facing = "-X" if unit == "Unit4" else "+X"
        verts = [(0, -width / 2, 0), (0, width / 2, 0), (0, width / 2, height), (0, -width / 2, height)]
        offset = Vector((-0.018 if facing == "-X" else 0.018, 0, 0.01))

    mesh = bpy.data.meshes.new(f"{name}_mesh")
    mesh.from_pydata(verts, [], [(0, 1, 2, 3)])
    mesh.update()
    mesh.uv_layers.new(name="UVMap")
    for loop, uv in zip(mesh.uv_layers.active.data, [(0, 0), (1, 0), (1, 1), (0, 1)]):
        loop.uv = uv
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.collection.objects.link(obj)
    obj.location = location + offset
    obj.data.materials.append(material)
    return obj


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--placements-json", required=True)
    parser.add_argument("--cluster-radius", type=float, default=0.075)
    parser.add_argument("--height", type=float, default=0.22)
    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else [])

    camera = load_camera()
    images = load_images()
    points3d = load_points3d()
    shelf_objects = [obj for obj in bpy.data.objects if obj.type == "MESH" and "_Shelf_" in obj.name]
    collection = bpy.data.collections.new("COLMAP_Exact_Product_Crops")
    bpy.context.scene.collection.children.link(collection)
    bpy.context.view_layer.active_layer_collection = bpy.context.view_layer.layer_collection.children[collection.name]

    hits = []
    total = 0
    ray_hits = 0
    observed_hits = 0
    for det in iter_detections():
        rays = rays_for_detection(det, camera, images)
        if not rays:
            continue
        total += 1
        hit = None
        for ray in rays:
            candidate = shelf_hit(ray[0], ray[1], shelf_objects)
            if candidate is not None:
                hit = candidate
                hit["source"] = "ray"
                ray_hits += 1
                break
        if hit is None:
            hit = shelf_from_observed_points(det, images, points3d, shelf_objects)
            if hit is not None:
                observed_hits += 1
        if hit is None:
            continue
        hit["detection"] = det
        hit["shelf_name"] = hit["shelf"].name
        hits.append(hit)

    clusters = cluster_hits(hits, args.cluster_radius)
    placements = []
    for index, cluster in enumerate(clusters):
        hit = choose_member(cluster)
        det = hit["detection"]
        safe_id = det["id"].replace("-", "_")
        material, _image = material_for_crop(f"ExactCropMat_{safe_id}", det["crop_file"])
        obj = create_product_plane(
            f"ExactCrop_{index:04d}_{det['id']}",
            hit["location"],
            hit["shelf_name"],
            det["crop_file"],
            args.height,
            material,
        )
        obj["source_detection_id"] = det["id"]
        obj["source_image"] = det["image_name"]
        obj["product_name"] = det.get("product_name") or ""
        obj["source_crop"] = str(det["crop_file"])
        obj["shelf_object"] = hit["shelf_name"]
        obj["cluster_view_count"] = len(cluster["members"])
        obj["placement_source"] = hit.get("source", "unknown")
        placements.append(
            {
                "object": obj.name,
                "detection_id": det["id"],
                "image_name": det["image_name"],
                "product_name": det.get("product_name"),
                "crop_file": str(det["crop_file"]),
                "shelf_object": hit["shelf_name"],
                "location": [float(hit["location"].x), float(hit["location"].y), float(hit["location"].z)],
                "cluster_view_count": len(cluster["members"]),
                "placement_source": hit.get("source", "unknown"),
            }
        )

    Path(args.placements_json).write_text(
        json.dumps(
            {
                "method": "COLMAP camera rays intersected with Blender shelf meshes",
                "detections_with_crops": total,
                "shelf_ray_hits": len(hits),
                "direct_ray_hits": ray_hits,
                "observed_point_fallback_hits": observed_hits,
                "clustered_products": len(placements),
                "placements": placements,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    bpy.ops.wm.save_as_mainfile(filepath=args.output)
    print("DETECTIONS_WITH_CROPS", total)
    print("SHELF_RAY_HITS", len(hits))
    print("DIRECT_RAY_HITS", ray_hits)
    print("OBSERVED_POINT_FALLBACK_HITS", observed_hits)
    print("CLUSTERED_PRODUCTS", len(placements))
    print("SAVED", args.output)


if __name__ == "__main__":
    main()
