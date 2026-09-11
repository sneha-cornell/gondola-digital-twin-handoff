#!/usr/bin/env bash
set -euo pipefail

export LOCAL_VLM_ENABLED="${LOCAL_VLM_ENABLED:-1}"
export LOCAL_VLM_DEVICE="${LOCAL_VLM_DEVICE:-cuda}"
export LOCAL_VLM_MODEL="${LOCAL_VLM_MODEL:-Qwen/Qwen2.5-VL-3B-Instruct}"
export VLM_FALLBACK_ENABLED="${VLM_FALLBACK_ENABLED:-0}"
export OPEN_WORLD_ENABLED="${OPEN_WORLD_ENABLED:-1}"
export PRODUCT_NAME_EMBEDDING_FALLBACK="${PRODUCT_NAME_EMBEDDING_FALLBACK:-1}"
export PRODUCT_NAME_OCR_ENGINE="${PRODUCT_NAME_OCR_ENGINE:-none}"

exec "$(dirname "$0")/.venv/bin/python" "$(dirname "$0")/backend/test_recognition.py" "$@"
