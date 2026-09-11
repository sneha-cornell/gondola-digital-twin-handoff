"""
Apply identified inferred slot product names to a Blender shelf file.

Reads the identified JSON (output of identify_inferred_slots.py) and updates
LayoutUnknownSlot_* objects with their assigned product names and colors.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import bpy


INDEX_RE = re.compile(r"_(\d{4})(?:_|$)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--identified-json", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--kb-dir", default=None, help="Path to knowledge_base/ directory")
    return parser.parse_args(sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else [])


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


def crop_image_material(detection_id: str, crop_file: str) -> bpy.types.Material | None:
    """Return a material with the YOLO focus-crop image as texture, or None if file missing."""
    img_path = Path(crop_file) if crop_file else None
    if not img_path or not img_path.exists():
        return None
    mat_name = f"CropMat_{detection_id}"
    mat = bpy.data.materials.get(mat_name)
    if mat:
        return mat
    mat = bpy.data.materials.new(mat_name)
    mat.use_nodes = True
    mat.blend_method = "OPAQUE"
    mat.use_backface_culling = False
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()
    output = nodes.new("ShaderNodeOutputMaterial")
    bsdf = nodes.new("ShaderNodeBsdfPrincipled")
    bsdf.inputs["Alpha"].default_value = 1.0
    output.location = (300, 0)
    bsdf.location = (0, 0)
    links.new(bsdf.outputs["BSDF"], output.inputs["Surface"])
    tex_node = nodes.new("ShaderNodeTexImage")
    tex_node.location = (-300, 0)
    try:
        img = bpy.data.images.load(str(img_path), check_existing=True)
        img.alpha_mode = "NONE"
        tex_node.image = img
        links.new(tex_node.outputs["Color"], bsdf.inputs["Base Color"])
    except Exception:
        return None
    return mat


def kb_image_material(
    product_name: str,
    kb_dir: Path | None,
    fallback_color: tuple[float, float, float, float] = (0.38, 0.56, 0.72, 1.0),
) -> bpy.types.Material:
    """Return a material using the KB front-of-pack image, or a flat color fallback."""
    img_path: Path | None = None
    if kb_dir and product_name not in ("Unknown", "Blank", ""):
        front_dir = kb_dir / product_name / "front"
        if front_dir.is_dir():
            candidates = sorted(front_dir.glob("*.jpg")) + sorted(front_dir.glob("*.png"))
            if candidates:
                img_path = candidates[0]

    mat_name = f"KBTexture_{product_name}"
    mat = bpy.data.materials.get(mat_name)
    if mat:
        return mat

    mat = bpy.data.materials.new(mat_name)
    mat.use_nodes = True
    mat.blend_method = "OPAQUE"
    mat.use_backface_culling = False
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()

    output = nodes.new("ShaderNodeOutputMaterial")
    bsdf = nodes.new("ShaderNodeBsdfPrincipled")
    bsdf.inputs["Alpha"].default_value = 1.0
    output.location = (300, 0)
    bsdf.location = (0, 0)
    links.new(bsdf.outputs["BSDF"], output.inputs["Surface"])

    if img_path:
        tex_node = nodes.new("ShaderNodeTexImage")
        tex_node.location = (-300, 0)
        try:
            img = bpy.data.images.load(str(img_path), check_existing=True)
            img.alpha_mode = "NONE"
            tex_node.image = img
            links.new(tex_node.outputs["Color"], bsdf.inputs["Base Color"])
        except Exception:
            bsdf.inputs["Base Color"].default_value = fallback_color
    else:
        bsdf.inputs["Base Color"].default_value = fallback_color

    return mat


def set_material(obj: bpy.types.Object, mat: bpy.types.Material) -> None:
    obj.data.materials.clear()
    obj.data.materials.append(mat)


def shorten_label(value: str, limit: int = 28) -> str:
    value = " ".join(value.split())
    if len(value) <= limit:
        return value
    return value[:limit - 1].rstrip() + "."


def object_index(name: str) -> str | None:
    match = INDEX_RE.search(name)
    return match.group(1) if match else None


def update_label(index: str, product_name: str) -> int:
    updated = 0
    for obj in bpy.data.objects:
        if obj.type != "FONT" or not obj.name.startswith(f"LayoutLabel_{index}_"):
            continue
        obj.data.body = shorten_label(product_name)
        obj["product_name"] = product_name
        updated += 1
    return updated


def main() -> None:
    args = parse_args()
    payload = json.loads(Path(args.identified_json).read_text(encoding="utf-8"))
    placements = payload.get("placements") or []
    kb_dir = Path(args.kb_dir) if args.kb_dir else None

    # Flat-color fallbacks for slots without a KB image
    mat_neighbor_high = make_material(
        "IdentifiedNeighborHigh", (0.85, 0.60, 0.12, 1.0)  # amber
    )
    mat_neighbor_low = make_material(
        "IdentifiedNeighborLow", (0.85, 0.85, 0.25, 1.0)  # pale yellow
    )

    updated_objects = 0
    updated_labels = 0
    missing_objects = 0

    for placement in placements:
        status = placement.get("product_identity_status", "")
        product_name = str(placement.get("product_name") or "Unknown")

        # Process all recognized single_best products (may be stuck with grey material
        # because blender_apply_merged_layout_names.py only handles strict-COLMAP objects)
        # AND all inferred_neighbor products identified by this script.
        # AND unknown/inferred_unknown_slot products — apply crop image so shelf looks full.
        handle_recognized = (
            status == "recognized"
            and product_name not in ("Unknown", "Blank")
            and not placement.get("merged_from_strict_detection_id")
        )
        handle_inferred = status in ("inferred_neighbor", "inferred_neighbor_low")
        handle_unknown = status in ("unknown", "inferred_unknown_slot")

        if not handle_recognized and not handle_inferred and not handle_unknown:
            continue

        obj_name = str(placement.get("object") or "")
        obj = bpy.data.objects.get(obj_name)
        if obj is None:
            missing_objects += 1
            continue

        id_source = str(placement.get("identification_source") or "")
        prop_conf = str(placement.get("propagated_neighbor_confidence") or "")
        visual_conf = float(placement.get("visual_confidence") or 0.0)

        obj["product_name"] = product_name
        obj["product_identity_status"] = status
        obj["placement_source"] = str(placement.get("placement_source") or "")
        if id_source:
            obj["identification_source"] = id_source
        if prop_conf:
            obj["propagated_neighbor_confidence"] = prop_conf
        obj["visual_confidence"] = visual_conf

        if obj.type == "MESH":
            if handle_unknown:
                # Show the actual crop photo for unidentified slots
                crop_mat = crop_image_material(
                    str(placement.get("detection_id") or obj_name),
                    str(placement.get("crop_file") or ""),
                )
                if crop_mat:
                    set_material(obj, crop_mat)
            elif handle_recognized or "visual" in id_source:
                set_material(obj, kb_image_material(product_name, kb_dir))
            elif status == "inferred_neighbor":
                set_material(obj, kb_image_material(product_name, kb_dir, (0.85, 0.60, 0.12, 1.0)))
            else:
                set_material(obj, kb_image_material(product_name, kb_dir, (0.85, 0.85, 0.25, 1.0)))

        index = object_index(obj_name)
        if index:
            updated_labels += update_label(index, product_name)

        updated_objects += 1

    bpy.ops.wm.save_as_mainfile(filepath=args.output)
    print("UPDATED_OBJECTS", updated_objects)
    print("UPDATED_LABELS", updated_labels)
    print("MISSING_OBJECTS", missing_objects)
    print("SAVED", args.output)


if __name__ == "__main__":
    main()
