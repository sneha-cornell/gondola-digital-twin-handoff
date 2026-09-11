"""Set product material viewport colors from their texture images.

Blender solid/workbench views use material diffuse colors, not necessarily image
textures. This makes detected products and stocked clones visibly product-like
instead of flat grey in those modes.
"""

from __future__ import annotations

import argparse
import hashlib
import sys

import bpy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--sample-step", type=int, default=80)
    return parser.parse_args(sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else [])


def image_from_material(mat: bpy.types.Material) -> bpy.types.Image | None:
    if not mat.use_nodes or not mat.node_tree:
        return None
    for node in mat.node_tree.nodes:
        if node.bl_idname == "ShaderNodeTexImage" and node.image:
            return node.image
    return None


def fallback_color(seed: str) -> tuple[float, float, float, float]:
    digest = hashlib.sha1(seed.encode("utf-8", errors="ignore")).digest()
    hue = digest[0] / 255.0
    sat = 0.45 + digest[1] / 255.0 * 0.35
    val = 0.55 + digest[2] / 255.0 * 0.35
    import colorsys

    r, g, b = colorsys.hsv_to_rgb(hue, sat, val)
    return r, g, b, 1.0


def average_image_color(image: bpy.types.Image, sample_step: int) -> tuple[float, float, float, float] | None:
    try:
        image.pixels[0]
    except Exception:
        return None

    pixels = image.pixels
    total = len(pixels) // 4
    if total <= 0:
        return None

    step = max(1, min(sample_step, total))
    r = g = b = count = 0.0
    for pixel_index in range(0, total, step):
        base = pixel_index * 4
        alpha = float(pixels[base + 3])
        if alpha < 0.05:
            continue
        r += float(pixels[base])
        g += float(pixels[base + 1])
        b += float(pixels[base + 2])
        count += 1.0
    if count <= 0:
        return None
    color = (r / count, g / count, b / count)
    # Lift contrast slightly so pale package crops are still visible on grey shelves.
    boosted = tuple(min(1.0, max(0.08, channel * 1.15)) for channel in color)
    return boosted[0], boosted[1], boosted[2], 1.0


def set_principled_color(mat: bpy.types.Material, color: tuple[float, float, float, float]) -> None:
    mat.diffuse_color = color
    if not mat.use_nodes or not mat.node_tree:
        return
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    if bsdf and "Base Color" in bsdf.inputs:
        # Keep texture links intact; this affects fallback/material previews.
        if not bsdf.inputs["Base Color"].links:
            bsdf.inputs["Base Color"].default_value = color


def main() -> None:
    args = parse_args()
    updated_materials = 0
    updated_objects = 0

    for mat in bpy.data.materials:
        if not (
            mat.name.startswith(("FreshCropMat_", "ReplicaCropMat_", "LayoutCropMat_"))
            or image_from_material(mat)
        ):
            continue
        image = image_from_material(mat)
        color = average_image_color(image, args.sample_step) if image else None
        if color is None:
            color = fallback_color(mat.name)
        set_principled_color(mat, color)
        updated_materials += 1

    for obj in bpy.data.objects:
        if obj.type != "MESH":
            continue
        if obj.name.startswith(("FreshProduct_", "StockedClone_", "LayoutProduct_")):
            obj.color = obj.active_material.diffuse_color if obj.active_material else fallback_color(obj.name)
            updated_objects += 1

    bpy.ops.wm.save_as_mainfile(filepath=args.output)
    print("UPDATED_MATERIALS", updated_materials)
    print("UPDATED_OBJECTS", updated_objects)
    print("SAVED", args.output)


if __name__ == "__main__":
    main()
