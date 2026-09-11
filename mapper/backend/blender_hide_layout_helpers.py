"""Hide layout helper boxes and labels so product image planes are visible."""

from __future__ import annotations

import argparse
import sys

import bpy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    return parser.parse_args(sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else [])


def main() -> None:
    args = parse_args()
    hidden = 0
    for obj in bpy.data.objects:
        if obj.name.startswith(("LayoutProductBox_", "LayoutLabel_", "LayoutLabelCurve_")):
            obj.hide_viewport = True
            obj.hide_render = True
            hidden += 1
    bpy.ops.wm.save_as_mainfile(filepath=args.output)
    print("HELPERS_HIDDEN", hidden)
    print("SAVED", args.output)


if __name__ == "__main__":
    main()
