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
    parser.add_argument("--merged-json", required=True)
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


def kb_image_material(
    product_name: str,
    kb_dir: Path | None,
    fallback_color: tuple[float, float, float, float] = (0.24, 0.55, 0.78, 1.0),
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


def object_index(name: str) -> str | None:
    match = INDEX_RE.search(name)
    return match.group(1) if match else None


def set_material(obj: bpy.types.Object, mat: bpy.types.Material) -> None:
    obj.data.materials.clear()
    obj.data.materials.append(mat)


def shorten_label(value: str, limit: int = 28) -> str:
    value = " ".join(value.split())
    if len(value) <= limit:
        return value
    return value[: limit - 1].rstrip() + "."


def update_label(index: str, product_name: str) -> int:
    updated = 0
    for obj in bpy.data.objects:
        if obj.type != "FONT" or not obj.name.startswith(f"LayoutLabel_{index}_"):
            continue
        obj.data.body = shorten_label(product_name)
        obj["product_name"] = product_name
        obj["product_identity_status"] = "recognized"
        updated += 1
    return updated


def main() -> None:
    args = parse_args()
    payload = json.loads(Path(args.merged_json).read_text(encoding="utf-8"))
    placements = payload.get("placements") or []
    kb_dir = Path(args.kb_dir) if args.kb_dir else None
    fallback_mat = make_material("LayoutMergedFromStrict_Product", (0.24, 0.55, 0.78, 1.0))

    updated_objects = 0
    updated_labels = 0
    missing_objects = 0
    for placement in placements:
        if not placement.get("merged_from_strict_detection_id"):
            continue
        obj = bpy.data.objects.get(str(placement.get("object") or ""))
        if obj is None:
            missing_objects += 1
            continue
        product_name = str(placement.get("product_name") or "Unknown")
        obj["product_name"] = product_name
        obj["product_identity_status"] = "recognized"
        obj["source_detection_id"] = str(placement.get("detection_id") or "")
        obj["source_image"] = str(placement.get("image_name") or "")
        obj["source_crop"] = str(placement.get("crop_file") or "")
        obj["placement_source"] = str(placement.get("placement_source") or "")
        obj["merged_from_strict_detection_id"] = str(placement.get("merged_from_strict_detection_id") or "")
        obj["merged_match_distance"] = float(placement.get("merged_match_distance") or 0.0)
        if obj.type == "MESH":
            mat = kb_image_material(product_name, kb_dir) if kb_dir else fallback_mat
            set_material(obj, mat)
        index = object_index(obj.name)
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
