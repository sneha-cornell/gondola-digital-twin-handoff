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
        if line.strip() and not line.startswith("#"):
            p = line.split()
            return {"f": float(p[4]), "cx": float(p[5]), "cy": float(p[6])}
    raise RuntimeError("camera not found")


def load_images():
    lines = [line for line in (TEXT_DIR / "images.txt").read_text().splitlines() if not line.lstrip().startswith("#")]
    out = {}
    i = 0
    while i < len(lines):
        header = lines[i].strip()
        i += 2
        if not header:
            continue
        p = header.split(maxsplit=9)
        if len(p) < 10:
            continue
        rotation = qvec_to_matrix([float(v) for v in p[1:5]])
        tvec = Vector((float(p[5]), float(p[6]), float(p[7])))
        out[p[9]] = {"rotation": rotation, "center": -(rotation.transposed() @ tvec)}
    return out


def local_path(value):
    if not value:
        return None
    path = Path(value)
    if path.exists():
        return path
    marker = "product_detection1/"
    text = str(value)
    if marker in text:
        candidate = REPO / text.split(marker, 1)[1]
        if candidate.exists():
            return candidate
    return None


def load_json(path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}


def iter_detections():
    for path in sorted((JOB_DIR / "detections").glob("*.json")):
        payload = load_json(path)
        for det in payload.get("detections", []):
            item = dict(det)
            names = load_json(JOB_DIR / "product_names" / f"{item['id']}.json")
            item["product_name"] = names.get("product_name") or names.get("catalog_candidate") or names.get("candidate") or item.get("display_label")
            item["name_confidence"] = float(names.get("catalog_confidence") or names.get("confidence") or 0.0)
            item["crop_file"] = local_path(item.get("crop_path") or names.get("selected_crop_path") or item.get("raw_crop_path"))
            if item["crop_file"] and item["crop_file"].exists():
                yield item


def ray_from_bbox_center(det, camera, images):
    image = images.get(det["image_name"])
    if image is None:
        return None
    x1, y1, x2, y2 = det["bbox"]
    u = 0.5 * (x1 + x2)
    v = 0.5 * (y1 + y2)
    direction_camera = Vector(((u - camera["cx"]) / camera["f"], (v - camera["cy"]) / camera["f"], 1.0)).normalized()
    return image["center"], (image["rotation"].transposed() @ direction_camera).normalized()


def bounds(obj):
    corners = [obj.matrix_world @ Vector(corner) for corner in obj.bound_box]
    xs = [c.x for c in corners]
    ys = [c.y for c in corners]
    zs = [c.z for c in corners]
    return (min(xs), max(xs)), (min(ys), max(ys)), (min(zs), max(zs))


def build_front_planes():
    planes = []
    for unit in ("Unit1", "Unit2", "Unit3", "Unit4"):
        shelves = sorted(
            [o for o in bpy.data.objects if o.type == "MESH" and o.name.startswith(f"{unit}_Shelf_")],
            key=lambda o: bounds(o)[2][0],
        )
        for idx, shelf in enumerate(shelves):
            xb, yb, zb = bounds(shelf)
            next_z = bounds(shelves[idx + 1])[2][0] if idx + 1 < len(shelves) else 2.95
            z_min = zb[1] + 0.02
            z_max = max(z_min + 0.08, next_z - 0.04)
            if unit == "Unit1":
                plane = {"unit": unit, "shelf": shelf.name, "axis": "y", "value": yb[1] + 0.012, "u": "x", "u_bounds": xb, "z_bounds": (z_min, z_max)}
            elif unit == "Unit3":
                plane = {"unit": unit, "shelf": shelf.name, "axis": "y", "value": yb[0] - 0.012, "u": "x", "u_bounds": xb, "z_bounds": (z_min, z_max)}
            elif unit == "Unit2":
                plane = {"unit": unit, "shelf": shelf.name, "axis": "x", "value": xb[0] - 0.012, "u": "y", "u_bounds": yb, "z_bounds": (z_min, z_max)}
            else:
                plane = {"unit": unit, "shelf": shelf.name, "axis": "x", "value": xb[1] + 0.012, "u": "y", "u_bounds": yb, "z_bounds": (z_min, z_max)}
            planes.append(plane)
    return planes


def intersect_front_planes(origin, direction, planes):
    best = None
    for plane in planes:
        axis_i = 1 if plane["axis"] == "y" else 0
        denom = direction[axis_i]
        if abs(denom) < 1e-7:
            continue
        t = (plane["value"] - origin[axis_i]) / denom
        if t <= 0:
            continue
        point = origin + direction * t
        u_value = point.x if plane["u"] == "x" else point.y
        if not (plane["u_bounds"][0] - 0.03 <= u_value <= plane["u_bounds"][1] + 0.03):
            continue
        if not (plane["z_bounds"][0] <= point.z <= plane["z_bounds"][1]):
            continue
        if best is None or t < best["distance"]:
            best = {"location": point, "distance": t, "plane": plane}
    return best


def cluster_hits(hits, radius):
    clusters = []
    for hit in hits:
        target = None
        for cluster in clusters:
            if cluster["shelf"] != hit["plane"]["shelf"]:
                continue
            center = sum((h["location"] for h in cluster["members"]), Vector()) / len(cluster["members"])
            if (hit["location"] - center).length <= radius:
                target = cluster
                break
        if target is None:
            clusters.append({"shelf": hit["plane"]["shelf"], "members": [hit]})
        else:
            target["members"].append(hit)
    return clusters


def choose(cluster):
    return max(cluster["members"], key=lambda h: (h["det"].get("name_confidence", 0), h["det"].get("confidence", 0)))


def material(name, color):
    mat = bpy.data.materials.new(name)
    mat.diffuse_color = color
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    if bsdf:
        bsdf.inputs["Base Color"].default_value = color
        bsdf.inputs["Alpha"].default_value = color[3]
    mat.blend_method = "BLEND"
    return mat


def add_box(hit, idx, box_mat, dot_mat):
    det = hit["det"]
    point = hit["location"]
    unit = hit["plane"]["unit"]
    width, depth, height = 0.12, 0.045, 0.18
    bpy.ops.mesh.primitive_cube_add(size=1, location=point)
    box = bpy.context.object
    box.name = f"FrontPlaneProduct_{idx:04d}_{det['id']}"
    if unit in {"Unit1", "Unit3"}:
        box.dimensions = (width, depth, height)
    else:
        box.dimensions = (depth, width, height)
    box.data.materials.append(box_mat)
    box["product_name"] = det.get("product_name") or ""
    box["shelf_object"] = hit["plane"]["shelf"]
    box["source_detection_id"] = det["id"]
    box["source_image"] = det["image_name"]
    box["coordinate"] = [float(point.x), float(point.y), float(point.z)]
    bpy.ops.mesh.primitive_uv_sphere_add(segments=16, ring_count=8, radius=0.018, location=point)
    dot = bpy.context.object
    dot.name = f"FrontPlaneDot_{idx:04d}_{det['id']}"
    dot.data.materials.append(dot_mat)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--coordinates-json", required=True)
    parser.add_argument("--cluster-radius", type=float, default=0.055)
    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else [])

    camera = load_camera()
    images = load_images()
    planes = build_front_planes()
    hits = []
    for det in iter_detections():
        ray = ray_from_bbox_center(det, camera, images)
        if ray is None:
            continue
        hit = intersect_front_planes(ray[0], ray[1], planes)
        if hit is None:
            continue
        hit["det"] = det
        hits.append(hit)

    clusters = cluster_hits(hits, args.cluster_radius)
    box_mat = material("FrontPlaneProductBlue", (0.05, 0.35, 1.0, 0.55))
    dot_mat = material("FrontPlaneDotRed", (1.0, 0.02, 0.02, 1.0))
    collection = bpy.data.collections.new("Front_Plane_Product_Positions")
    bpy.context.scene.collection.children.link(collection)
    bpy.context.view_layer.active_layer_collection = bpy.context.view_layer.layer_collection.children[collection.name]

    rows = []
    for idx, cluster in enumerate(clusters, start=1):
        hit = choose(cluster)
        add_box(hit, idx, box_mat, dot_mat)
        det = hit["det"]
        p = hit["location"]
        rows.append({
            "index": idx,
            "product_name": det.get("product_name") or "",
            "shelf_object": hit["plane"]["shelf"],
            "x": float(p.x),
            "y": float(p.y),
            "z": float(p.z),
            "detection_id": det["id"],
            "image_name": det["image_name"],
            "cluster_view_count": len(cluster["members"]),
            "crop_file": str(det["crop_file"]),
        })

    Path(args.coordinates_json).write_text(json.dumps(rows, indent=2), encoding="utf-8")
    bpy.ops.wm.save_as_mainfile(filepath=args.output)
    print("FRONT_PLANE_HITS", len(hits))
    print("FRONT_PLANE_PRODUCTS", len(rows))
    print("SAVED", args.output)


if __name__ == "__main__":
    main()
