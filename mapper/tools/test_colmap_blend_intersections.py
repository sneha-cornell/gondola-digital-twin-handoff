import json
import math
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
        return {
            "model": parts[1],
            "width": int(parts[2]),
            "height": int(parts[3]),
            "f": float(parts[4]),
            "cx": float(parts[5]),
            "cy": float(parts[6]),
        }
    raise RuntimeError("No camera found")


def load_images():
    lines = [
        line for line in (TEXT_DIR / "images.txt").read_text().splitlines()
        if not line.lstrip().startswith("#")
    ]
    images = {}
    i = 0
    while i < len(lines):
        header = lines[i].strip()
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
        images[parts[9]] = {"rotation": rotation, "center": center}
    return images


def iter_detections():
    for path in sorted((JOB_DIR / "detections").glob("*.json")):
        payload = json.loads(path.read_text())
        for det in payload.get("detections", []):
            yield det


def ray_for_detection(det, camera, images, anchor="center"):
    image = images.get(det["image_name"])
    if image is None:
        return None
    x1, y1, x2, y2 = det["bbox"]
    u = 0.5 * (x1 + x2)
    if anchor == "bottom":
        v = y2
    else:
        v = 0.5 * (y1 + y2)
    direction_camera = Vector(((u - camera["cx"]) / camera["f"], (v - camera["cy"]) / camera["f"], 1.0)).normalized()
    direction_world = (image["rotation"].transposed() @ direction_camera).normalized()
    return image["center"], direction_world


def main():
    camera = load_camera()
    images = load_images()
    depsgraph = bpy.context.evaluated_depsgraph_get()

    shelf_objects = [
        obj for obj in bpy.data.objects
        if obj.type == "MESH" and "_Shelf_" in obj.name
    ]
    total = 0
    hits = 0
    hit_by_object = {}
    examples = []
    for det in iter_detections():
        ray = ray_for_detection(det, camera, images, anchor="bottom")
        if ray is None:
            continue
        total += 1
        origin, direction = ray
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
            if best is None or distance < best[0]:
                best = (distance, world_loc, shelf_obj)
        if best is not None:
            _, loc, obj = best
            hits += 1
            hit_by_object[obj.name] = hit_by_object.get(obj.name, 0) + 1
            if len(examples) < 5:
                examples.append((det["id"], det["image_name"], obj.name, tuple(round(v, 4) for v in loc)))

    print("TOTAL_DETECTIONS", total)
    print("SHELF_HITS", hits)
    print("HIT_OBJECTS", sorted(hit_by_object.items(), key=lambda item: (-item[1], item[0]))[:30])
    print("EXAMPLES", examples)


if __name__ == "__main__":
    main()
