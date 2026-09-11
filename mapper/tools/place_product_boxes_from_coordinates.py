import argparse
import json
import math
import sys
from pathlib import Path

import bpy
from mathutils import Vector


DEFAULT_COORDS = Path(
    "/home/ec2-user/SageMaker/product_detection1/backend/workspace/"
    "iphone16-2_named_high_recall_v4_job/product_coordinates_v2.json"
)


def make_material(name: str, color: tuple[float, float, float, float]):
    mat = bpy.data.materials.new(name)
    mat.diffuse_color = color
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    if bsdf:
        bsdf.inputs["Base Color"].default_value = color
        bsdf.inputs["Alpha"].default_value = color[3]
    mat.blend_method = "BLEND"
    return mat


def shelf_unit(shelf_name: str) -> str:
    return shelf_name.split("_", 1)[0]


def create_box(name: str, placement: dict, box_mat, marker_mat, label_mat, add_labels: bool):
    loc = Vector((placement["x"], placement["y"], placement["z"]))
    unit = shelf_unit(placement["shelf_object"])

    # Product marker dimensions in shelf coordinates. These are deliberately
    # small blocks for validation, not estimated package dimensions.
    width = 0.12
    depth = 0.055
    height = 0.18

    bpy.ops.mesh.primitive_cube_add(size=1.0, location=loc + Vector((0, 0, height / 2 + 0.01)))
    box = bpy.context.object
    box.name = name

    if unit in {"Unit1", "Unit3"}:
        box.dimensions = (width, depth, height)
        y_offset = -depth / 2 if unit == "Unit1" else depth / 2
        box.location.y += y_offset
    else:
        box.dimensions = (depth, width, height)
        x_offset = depth / 2 if unit == "Unit2" else -depth / 2
        box.location.x += x_offset
    box.data.materials.append(box_mat)
    box["product_name"] = placement.get("product_name") or ""
    box["shelf_object"] = placement["shelf_object"]
    box["source_detection_id"] = placement["detection_id"]
    box["source_image"] = placement["image_name"]
    box["exact_coordinate"] = [placement["x"], placement["y"], placement["z"]]

    bpy.ops.mesh.primitive_uv_sphere_add(
        segments=16,
        ring_count=8,
        radius=0.025,
        location=loc + Vector((0, 0, 0.035)),
    )
    marker = bpy.context.object
    marker.name = f"CoordDot_{placement['index']:04d}_{placement['detection_id']}"
    marker.data.materials.append(marker_mat)
    marker["product_name"] = placement.get("product_name") or ""
    marker["shelf_object"] = placement["shelf_object"]
    marker["exact_coordinate"] = [placement["x"], placement["y"], placement["z"]]

    if add_labels and placement.get("product_name"):
        curve = bpy.data.curves.new(f"LabelCurve_{placement['index']:04d}", "FONT")
        curve.body = placement["product_name"][:36]
        curve.align_x = "CENTER"
        curve.size = 0.045
        label = bpy.data.objects.new(f"CoordLabel_{placement['index']:04d}_{placement['detection_id']}", curve)
        bpy.context.collection.objects.link(label)
        label.location = box.location + Vector((0, 0, height / 2 + 0.045))
        if unit in {"Unit1", "Unit3"}:
            label.rotation_euler.x = math.radians(70)
            if unit == "Unit3":
                label.rotation_euler.z = math.pi
        else:
            label.rotation_euler.x = math.radians(70)
            label.rotation_euler.z = math.radians(-90 if unit == "Unit2" else 90)
        label.data.materials.append(label_mat)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--coordinates", default=str(DEFAULT_COORDS))
    parser.add_argument("--output", required=True)
    parser.add_argument("--labels", action="store_true")
    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else [])

    placements = json.loads(Path(args.coordinates).read_text(encoding="utf-8"))

    collection = bpy.data.collections.new("Exact_Product_Box_Markers")
    bpy.context.scene.collection.children.link(collection)
    bpy.context.view_layer.active_layer_collection = bpy.context.view_layer.layer_collection.children[collection.name]

    box_mat = make_material("ExactProductBox_TransparentGreen", (0.1, 0.85, 0.25, 0.55))
    marker_mat = make_material("ExactCoordinateDot_Red", (1.0, 0.05, 0.02, 1.0))
    label_mat = make_material("ExactProductLabel_Black", (0.0, 0.0, 0.0, 1.0))

    for placement in placements:
        create_box(
            f"ProductBox_{placement['index']:04d}_{placement['detection_id']}",
            placement,
            box_mat,
            marker_mat,
            label_mat,
            args.labels,
        )

    bpy.ops.wm.save_as_mainfile(filepath=args.output)
    print(f"PLACED_BOXES {len(placements)}")
    print(f"SAVED {args.output}")


if __name__ == "__main__":
    main()
