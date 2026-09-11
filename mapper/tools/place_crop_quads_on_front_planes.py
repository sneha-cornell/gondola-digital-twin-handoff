import argparse
import json
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


def load_json(path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}


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
            z_max = max(z_min + 0.08, next_z - 0.035)
            if unit == "Unit1":
                planes.append({"unit": unit, "shelf": shelf.name, "axis": "y", "value": yb[1] + 0.016, "u": "x", "u_bounds": xb, "z_bounds": (z_min, z_max), "offset": Vector((0, -0.018, 0))})
            elif unit == "Unit3":
                planes.append({"unit": unit, "shelf": shelf.name, "axis": "y", "value": yb[0] - 0.016, "u": "x", "u_bounds": xb, "z_bounds": (z_min, z_max), "offset": Vector((0, 0.018, 0))})
            elif unit == "Unit2":
                planes.append({"unit": unit, "shelf": shelf.name, "axis": "x", "value": xb[0] - 0.016, "u": "y", "u_bounds": yb, "z_bounds": (z_min, z_max), "offset": Vector((0.018, 0, 0))})
            else:
                planes.append({"unit": unit, "shelf": shelf.name, "axis": "x", "value": xb[1] + 0.016, "u": "y", "u_bounds": yb, "z_bounds": (z_min, z_max), "offset": Vector((-0.018, 0, 0))})
    return planes


def ray_for_pixel(image, camera, u, v):
    direction_camera = Vector(((u - camera["cx"]) / camera["f"], (v - camera["cy"]) / camera["f"], 1.0)).normalized()
    return image["center"], (image["rotation"].transposed() @ direction_camera).normalized()


def intersect_plane(origin, direction, plane):
    axis_i = 1 if plane["axis"] == "y" else 0
    denom = direction[axis_i]
    if abs(denom) < 1e-7:
        return None
    t = (plane["value"] - origin[axis_i]) / denom
    if t <= 0:
        return None
    return origin + direction * t


def in_plane_bounds(point, plane, margin=0.02):
    u_value = point.x if plane["u"] == "x" else point.y
    return (
        plane["u_bounds"][0] - margin <= u_value <= plane["u_bounds"][1] + margin
        and plane["z_bounds"][0] - margin <= point.z <= plane["z_bounds"][1] + margin
    )


def clamp_to_plane(point, plane):
    u_value = point.x if plane["u"] == "x" else point.y
    u_value = max(plane["u_bounds"][0], min(plane["u_bounds"][1], u_value))
    z_value = max(plane["z_bounds"][0], min(plane["z_bounds"][1], point.z))
    if plane["u"] == "x":
        return Vector((u_value, plane["value"], z_value)) + plane["offset"]
    return Vector((plane["value"], u_value, z_value)) + plane["offset"]


def best_plane_for_detection(det, camera, images, planes):
    image = images.get(det["image_name"])
    if image is None:
        return None
    x1, y1, x2, y2 = det["bbox"]
    origin, direction = ray_for_pixel(image, camera, 0.5 * (x1 + x2), 0.5 * (y1 + y2))
    best = None
    for plane in planes:
        p = intersect_plane(origin, direction, plane)
        if p is None or not in_plane_bounds(p, plane):
            continue
        dist = (p - origin).length
        if best is None or dist < best["dist"]:
            best = {"plane": plane, "dist": dist, "center": p}
    return best


def quad_for_detection(det, camera, images, plane):
    image = images[det["image_name"]]
    x1, y1, x2, y2 = det["bbox"]
    pixels = [(x1, y2), (x2, y2), (x2, y1), (x1, y1)]
    verts = []
    for u, v in pixels:
        origin, direction = ray_for_pixel(image, camera, u, v)
        p = intersect_plane(origin, direction, plane)
        if p is None:
            return None
        verts.append(clamp_to_plane(p, plane))
    # Reject pathological quads that collapse or span too much shelf space.
    width = max((verts[1] - verts[0]).length, (verts[2] - verts[3]).length)
    height = max((verts[3] - verts[0]).length, (verts[2] - verts[1]).length)
    if width < 0.025 or height < 0.025 or width > 0.9 or height > 1.0:
        return None
    return verts


def cluster_hits(hits, radius):
    clusters = []
    for hit in hits:
        target = None
        for cluster in clusters:
            if cluster["plane"]["shelf"] != hit["plane"]["shelf"]:
                continue
            center = sum((h["center"] for h in cluster["members"]), Vector()) / len(cluster["members"])
            if (hit["center"] - center).length <= radius:
                target = cluster
                break
        if target is None:
            clusters.append({"plane": hit["plane"], "members": [hit]})
        else:
            target["members"].append(hit)
    return clusters


def choose(cluster):
    return max(cluster["members"], key=lambda h: (h["det"].get("name_confidence", 0), h["det"].get("confidence", 0)))


def material_for_crop(name, crop_file):
    image = bpy.data.images.load(str(crop_file), check_existing=True)
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    mat.blend_method = "BLEND"
    nodes = mat.node_tree.nodes
    bsdf = nodes.get("Principled BSDF")
    tex = nodes.new("ShaderNodeTexImage")
    tex.image = image
    mat.node_tree.links.new(tex.outputs["Color"], bsdf.inputs["Base Color"])
    return mat


def add_crop_quad(index, hit):
    det = hit["det"]
    mat = material_for_crop(f"ProjectedCropMat_{det['id'].replace('-', '_')}", det["crop_file"])
    mesh = bpy.data.meshes.new(f"ProjectedCrop_{index:04d}_{det['id']}_mesh")
    mesh.from_pydata([tuple(v) for v in hit["quad"]], [], [(0, 1, 2, 3)])
    mesh.update()
    mesh.uv_layers.new(name="UVMap")
    for loop, uv in zip(mesh.uv_layers.active.data, [(0, 0), (1, 0), (1, 1), (0, 1)]):
        loop.uv = uv
    obj = bpy.data.objects.new(f"ProjectedCrop_{index:04d}_{det['id']}", mesh)
    bpy.context.collection.objects.link(obj)
    obj.data.materials.append(mat)
    obj["product_name"] = det.get("product_name") or ""
    obj["shelf_object"] = hit["plane"]["shelf"]
    obj["source_detection_id"] = det["id"]
    obj["source_image"] = det["image_name"]
    obj["crop_file"] = str(det["crop_file"])
    obj["center_coordinate"] = [float(hit["center"].x), float(hit["center"].y), float(hit["center"].z)]
    return obj


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
        best = best_plane_for_detection(det, camera, images, planes)
        if best is None:
            continue
        quad = quad_for_detection(det, camera, images, best["plane"])
        if quad is None:
            continue
        best["det"] = det
        best["quad"] = quad
        hits.append(best)

    clusters = cluster_hits(hits, args.cluster_radius)
    collection = bpy.data.collections.new("Projected_Crop_Quads")
    bpy.context.scene.collection.children.link(collection)
    bpy.context.view_layer.active_layer_collection = bpy.context.view_layer.layer_collection.children[collection.name]

    rows = []
    for idx, cluster in enumerate(clusters, start=1):
        hit = choose(cluster)
        obj = add_crop_quad(idx, hit)
        c = hit["center"]
        det = hit["det"]
        rows.append({
            "index": idx,
            "object": obj.name,
            "product_name": det.get("product_name") or "",
            "shelf_object": hit["plane"]["shelf"],
            "x": float(c.x),
            "y": float(c.y),
            "z": float(c.z),
            "detection_id": det["id"],
            "image_name": det["image_name"],
            "cluster_view_count": len(cluster["members"]),
            "crop_file": str(det["crop_file"]),
        })

    Path(args.coordinates_json).write_text(json.dumps(rows, indent=2), encoding="utf-8")
    bpy.ops.wm.save_as_mainfile(filepath=args.output)
    print("PROJECTED_QUAD_HITS", len(hits))
    print("PROJECTED_QUAD_PRODUCTS", len(rows))
    print("SAVED", args.output)


if __name__ == "__main__":
    main()
