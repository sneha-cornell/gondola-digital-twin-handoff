"""Render each unit face from the identified blend file."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import bpy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args(sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else [])


def render_camera(camera_name: str, output_path: str) -> bool:
    cam = bpy.data.objects.get(camera_name)
    if cam is None:
        print(f"SKIP {camera_name} — not found")
        return False
    bpy.context.scene.camera = cam
    bpy.context.scene.render.filepath = output_path
    bpy.context.scene.render.image_settings.file_format = "PNG"
    bpy.context.scene.render.resolution_x = 3840
    bpy.context.scene.render.resolution_y = 2160
    bpy.ops.render.render(write_still=True)
    print(f"RENDERED {camera_name} -> {output_path}")
    return True


def main() -> None:
    args = parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    cameras = [
        ("CameraUnit1", str(out / "unit1_labeled.png")),
        ("CameraUnit2", str(out / "unit2_labeled.png")),
        ("CameraUnit3", str(out / "unit3_labeled.png")),
        ("CameraUnit4", str(out / "unit4_labeled.png")),
    ]

    # Fall back to any cameras that exist if named ones not found
    available = {o.name for o in bpy.data.objects if o.type == "CAMERA"}
    print("Available cameras:", sorted(available))

    rendered = 0
    for cam_name, path in cameras:
        if render_camera(cam_name, path):
            rendered += 1

    # If none of the named cameras matched, render from the active camera
    if rendered == 0:
        print("No named cameras matched — rendering from active camera")
        active_cam = bpy.context.scene.camera
        if active_cam:
            bpy.context.scene.render.filepath = str(out / "unit1_labeled.png")
            bpy.context.scene.render.image_settings.file_format = "PNG"
            bpy.ops.render.render(write_still=True)
            print(f"RENDERED active camera ({active_cam.name}) -> unit1_labeled.png")

    print("DONE")


if __name__ == "__main__":
    main()
