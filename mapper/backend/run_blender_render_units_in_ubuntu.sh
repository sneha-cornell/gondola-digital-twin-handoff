#!/usr/bin/env bash
set -euo pipefail

export DEBIAN_FRONTEND=noninteractive

apt-get update -qq
apt-get install -y -qq \
  libdbus-1-3 libegl1 libgl1 libice6 libsm6 \
  libx11-6 libxext6 libxfixes3 libxi6 libxkbcommon0 \
  libxrender1 libxxf86vm1

BLEND_IN="workspace/iphone16-2_named_high_recall_v4_job/shelf_22-2_single_best_image_layout_filled.blend"
OUTPUT_DIR="/app/backend/workspace/iphone16-2_named_high_recall_v4_job"

/tmp/blender-5.1.2-stable+v51.ec6e62d40fa9-linux.x86_64-release/blender \
  "$BLEND_IN" \
  --background \
  --python blender_render_units.py \
  -- \
  --output-dir "$OUTPUT_DIR"

echo "Done rendering"
