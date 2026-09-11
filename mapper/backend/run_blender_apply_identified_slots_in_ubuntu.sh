#!/usr/bin/env bash
set -euo pipefail

export DEBIAN_FRONTEND=noninteractive

apt-get update -qq
apt-get install -y -qq \
  libdbus-1-3 libegl1 libgl1 libice6 libsm6 \
  libx11-6 libxext6 libxfixes3 libxi6 libxkbcommon0 \
  libxrender1 libxxf86vm1

BLEND_IN="workspace/iphone16-2_named_high_recall_v4_job/shelf_22-2_single_best_image_layout_merged.blend"
IDENTIFIED_JSON="workspace/iphone16-2_named_high_recall_v4_job/blender_single_best_image_layout_identified.json"
BLEND_OUT="/app/backend/workspace/iphone16-2_named_high_recall_v4_job/shelf_22-2_single_best_image_layout_identified.blend"

/tmp/blender-5.1.2-stable+v51.ec6e62d40fa9-linux.x86_64-release/blender \
  "$BLEND_IN" \
  --background \
  --python blender_apply_identified_slots.py \
  -- \
  --identified-json "$IDENTIFIED_JSON" \
  --kb-dir /app/backend/knowledge_base \
  --output "$BLEND_OUT"

echo "Done: $BLEND_OUT"
