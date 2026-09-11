#!/usr/bin/env bash
set -euo pipefail

export DEBIAN_FRONTEND=noninteractive

apt-get update
apt-get install -y \
  libdbus-1-3 \
  libegl1 \
  libgl1 \
  libice6 \
  libsm6 \
  libx11-6 \
  libxext6 \
  libxfixes3 \
  libxi6 \
  libxkbcommon0 \
  libxrender1 \
  libxxf86vm1

/tmp/blender-5.1.2-stable+v51.ec6e62d40fa9-linux.x86_64-release/blender \
  ../shelf_22-2.blend \
  --background \
  --python blender_project_from_images_colmap.py \
  -- \
  --job-dir workspace/iphone16-2_named_high_recall_v4_job \
  --colmap-text-dir workspace/iphone16-2/text \
  --placements-json workspace/iphone16-2_named_high_recall_v4_job/blender_fresh_placements.json \
  --output /tmp/shelf_22-2_fresh_image_replica.blend \
  --product-boxes \
  --replace
