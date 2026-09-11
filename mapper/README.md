# Digital Twin Shelf Mapper

This repo contains a two-part application:

- `backend/`: FastAPI API for job discovery, optional COLMAP automation, object detection, sparse-model parsing, shelf-plane fitting, and `results.json` generation
- `frontend/`: React + Three.js viewer for browsing jobs and visualizing shelves and projected product markers

## Local Development

Local development does not require a COLMAP binary if a job already contains exported sparse model text files:

- `backend/workspace/<job-id>/text/cameras.txt`
- `backend/workspace/<job-id>/text/images.txt`
- `backend/workspace/<job-id>/text/points3D.txt`

Start the backend:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r backend/requirements.txt
python3 backend/main.py
```

Start the frontend:

```bash
cd frontend
npm install
npm run dev
```

## Automation / CI

Automation can use the backend container in `docker/backend.Dockerfile`, which installs COLMAP and the optional YOLO dependency set. That image is intended for reconstruction-enabled environments where jobs may arrive with images only and require sparse mapping before analysis.

Bring up both services with Docker:

```bash
docker compose up --build
```

## Retraining The Detector

The repo includes a local custom-dataset workflow for the shelf-product detector:

1. Build a YOLO dataset from workspace jobs, optionally bootstrapping missing labels from the current detector:

```bash
python training/build_workspace_dataset.py \
  --workspace-dir backend/workspace \
  --output-dir training/workspace_dataset \
  --pseudo-label \
  --confidence 0.10
```

2. Review and correct the generated labels if you want real model improvement on side-angle and endcap views.

3. Train a new detector:

```bash
python training/train_custom.py \
  --data training/workspace_dataset/dataset.yaml \
  --model backend/models/best.pt \
  --epochs 80 \
  --imgsz 960
```

`training/build_workspace_dataset.py` supports manual labels too: if a job contains `labels/<image>.txt` YOLO annotations, those are copied into the dataset and preferred over pseudo labels.

### Class-based dataset (per-SKU detector)

Passing `--class-labels` builds a **multi-class** dataset instead of the single generic
`product` class: each job's `recognition_results.json` is joined with its cached
`detections/` bboxes, and every confidently identified detection contributes a box with
its stable class id from `backend/knowledge_base/classes.json`. Unidentified detections
are dropped from the labels. `--min-identity-confidence` (default 0.65) controls the cut.

```bash
python training/build_workspace_dataset.py \
  --workspace-dir backend/workspace \
  --output-dir training/class_dataset \
  --class-labels
```

The emitted `dataset.yaml` carries the full per-SKU class map, so `train_custom.py`
trains a detector whose predictions are class-based records directly.

## Class Registry & Class-Based Records

`backend/product_classes.py` assigns every catalog product a **permanent integer class
id**, persisted in `backend/knowledge_base/classes.json` (append-only — ids are never
renumbered). After identification, every detection carries `class_id` / `class_name` /
`identity_stage`, and both `recognition_results.json` and `results.json` include a
`class_records` aggregation: per-class detection counts, image counts, source breakdown,
and confidence stats. Names identified outside the catalog (open-world hits) appear with
`class_id: null` so they can be reviewed and promoted into the catalog.

The identification summary also includes a `pipeline` array reporting each stage (ocr,
embedding, clip, vlm, open_world) with its enabled state, attempted/named counts, and —
when disabled — the concrete reason. Stages no longer fail silently; disabled stages are
logged as warnings at the start of every run.

Note: the DINOv2 embedding stage is now **enabled by default** (it was previously opt-in
via `PRODUCT_NAME_EMBEDDING_FALLBACK`, which meant API-triggered runs silently skipped
the highest-yield naming stage). Set `PRODUCT_NAME_EMBEDDING_FALLBACK=0` to opt out.

### One-command SKU-110K finetune

To rebuild a stronger `best.pt` from scratch using the public SKU-110K dataset:

```bash
# 1. Download SKU110K_fixed (~10GB) from the dataset's official source.
# 2. Convert + train + promote the new checkpoint into backend/models/best.pt:
python training/finetune_sku110k.py \
  --data-dir /path/to/SKU110K_fixed \
  --model yolo11l.pt \
  --epochs 60
```

`training/finetune_sku110k.py` runs `convert_sku110k.py` + `train_custom.py` end-to-end and copies the resulting `best.pt` into `backend/models/`. The previous checkpoint is preserved as `best.previous.pt`. Pass `--model yolo11x.pt` for maximum recall at higher inference cost, or `--model yolo11m.pt` for faster training/inference. When `backend/models/best.pt` is missing, the runtime detector falls back to `yolo11l.pt` (override with `YOLO_FALLBACK_MODEL`).

## Exact Product Names

Exact product-name recovery now uses a layered identification pipeline:

1. A detector finds every package crop.
2. `backend/product_identifier.py` combines OCR with catalog image matching from `backend/knowledge_base/` (closed-set, ~32 products in the bundled catalog).
3. **Open-world fallback** (`backend/open_world_identifier.py`) identifies *any* grocery product on Earth for crops the closed-set stages could not name: a free-form Claude vision call, verified against Open Food Facts (and Open Beauty / Products Facts on request), with optional barcode lookup.

## Open-World Identification

The open-world stage runs as the final fallback in `ProductIdentifier.enrich_detections` and writes the same `display_label` / `label` / `product_identity_label` fields as the closed-set stages, so identified products flow into the viewer automatically. It additionally surfaces an `open_world` payload on each detection containing the brand, Open Food Facts code, image URL, and any alternate candidates.

Required (one of these):

- `ANTHROPIC_API_KEY` — pay-per-call Claude Sonnet 4 (~$0.006 per crop).
- **OR** `LOCAL_VLM_ENABLED=1` — run an open-source VLM (Qwen2.5-VL) locally with no API costs. See **Local VLM Backend** below.

Without either backend, the open-world stage is silently disabled and the closed-set pipeline runs as before.

Optional environment knobs:

| Variable | Default | Purpose |
| --- | --- | --- |
| `OPEN_WORLD_ENABLED` | `1` | Master switch for the open-world stage. |
| `OPEN_WORLD_PROVIDERS` | `openfoodfacts` | Comma list; any of `openfoodfacts`, `openbeautyfacts`, `openproductsfacts`. |
| `OPEN_WORLD_USER_AGENT` | project UA | Custom User-Agent for the Open*Facts API (recommended for production). |
| `OPEN_WORLD_HTTP_TIMEOUT` | `8` | Per-request HTTP timeout in seconds. |
| `OPEN_WORLD_MIN_INTERVAL` | `0.25` | Minimum gap between OFF API calls (seconds). |
| `OPEN_WORLD_MIN_SIDE` | `96` | Skip crops with a shorter side below this value. |
| `OPEN_WORLD_ACCEPT_CONFIDENCE` | `0.55` | Confidence threshold to promote an open-world identification to a label. |
| `OPEN_WORLD_MATCH_THRESHOLD` | `0.55` | Fuzzy-match similarity required to canonicalize the VLM name against an Open Food Facts entry. |
| `OPEN_WORLD_BARCODE_ENABLED` | `1` | Enable pyzbar barcode lookup if the library + `libzbar` are installed. |
| `OPEN_WORLD_AUTO_CATALOG` | `0` | When a confident open-world hit also matches Open Food Facts strongly, copy the focused crop into `backend/knowledge_base/<product>/auto/` and append a `catalog.json` entry so subsequent runs use the cheap DINOv2 / SigLIP path. |

Caches live under:

- `backend/knowledge_base/open_world_cache/` — HTTP responses keyed by query hash, shared across jobs.
- `backend/workspace/<job>/open_world/<detection-id>.json` — per-detection identification results.

Delete either directory to force a re-query.

## Local VLM Backend

When `LOCAL_VLM_ENABLED=1` is set, both the catalog VLM stage (stage 4) and the open-world VLM stage (stage 5) delegate to an on-device open-source vision-language model — no Anthropic API key required, zero per-call cost.

**Default model:** `Qwen/Qwen2.5-VL-3B-Instruct` (~6 GB unified memory on Apple Silicon, ~1–2 s per crop in fp16). For higher accuracy on partially-occluded labels, set `LOCAL_VLM_MODEL=Qwen/Qwen2.5-VL-7B-Instruct` (needs ~16 GB RAM).

Optional environment knobs:

| Variable | Default | Purpose |
| --- | --- | --- |
| `LOCAL_VLM_ENABLED` | `0` | Master switch. Set to `1` to use the local VLM in place of Anthropic. |
| `LOCAL_VLM_MODEL` | `Qwen/Qwen2.5-VL-3B-Instruct` | Any HF model id with the Qwen2-VL or Qwen2.5-VL architecture. |
| `LOCAL_VLM_DEVICE` | auto | Force device: `mps`, `cuda`, `cpu`. Auto-detect picks MPS on Apple Silicon, then CUDA, then CPU. |
| `LOCAL_VLM_MAX_TOKENS` | `320` | Max generated tokens per call. |
| `LOCAL_VLM_MAX_PIXELS` | `512*28*28` (~401k, ~640×640) | Visual-token budget per image. Lower for less memory; raise for sharper packaging text. |
| `LOCAL_VLM_MIN_PIXELS` | `256*28*28` (~201k, ~448×448) | Lower bound to keep tiny crops legible. |
| `LOCAL_VLM_LONG_SIDE_PX` | `768` | PIL pre-resize cap. Set to `0` to disable. |
| `PYTORCH_MPS_HIGH_WATERMARK_RATIO` | (unset) | macOS only. Set to `0.0` in the **shell** before launching to let MPS use the full unified memory pool — required when running the **7B variant** on a 16 GB Mac. Setting it from inside Python is too late. |

Verify your setup before running a full job:

```bash
LOCAL_VLM_ENABLED=1 python backend/test_recognition.py --local-vlm-check
```

This downloads the weights on first run (~6 GB for the 3B variant), loads them on the auto-detected device, decodes one image, and prints timings + a JSON identification.

Once it passes, run a real job with the local backend:

```bash
LOCAL_VLM_ENABLED=1 python backend/test_recognition.py iphone16-2
```

Caveats:

- Apple Silicon needs `attn_implementation="eager"` (handled automatically). FlashAttention is CUDA-only.
- The 3B variant is materially weaker than Claude Sonnet 4 on small / partially occluded labels; expect ~10–15% lower top-1 accuracy. Use the 7B variant or fall back to Anthropic for production-quality identification.
- The local VLM stage is *not* batched. ~1–2 s per crop on an M3 Pro at 3B; ~3–5 s at 7B. For 100+ crops consider running on CUDA or batching in a future iteration.

The reference catalog supports both flat and multi-view layouts:

- `backend/knowledge_base/<product>/*.jpg`
- `backend/knowledge_base/<product>/front/*.jpg`
- `backend/knowledge_base/<product>/angle_left/*.jpg`
- `backend/knowledge_base/<product>/angle_right/*.jpg`
- `backend/knowledge_base/<product>/detail/*.jpg`

Add several reference images per product, especially:

- front view
- left-angle shelf view
- right-angle shelf view
- zoomed packaging detail if the product text is small

Optional aliases live in `backend/knowledge_base/catalog.json`. Those aliases help OCR convert partial text like `DOTS HONEY` into the exact catalog name `Dot's Homestyle Pretzels Honey Mustard`.

The embedding matcher depends on `transformers`, which is already listed in `backend/requirements.txt`.

To scaffold a new product entry with multi-view folders:

```bash
python training/register_catalog_product.py \
  --product "Kettle Brand Pepperoncini" \
  --alias "Kettle Pepperoncini"
```

## Workspace Layout

Each job lives under `backend/workspace/<job-id>/`:

- `images/`: source images
- `text/`: exported COLMAP sparse model text files
- `detections/`: cached detection metadata
- `crops/`: saved product crops
- `results.json`: final visualization payload

## API

- `GET /api/health`
- `GET /api/jobs`
- `GET /api/jobs/{job_id}`
- `GET /api/results/{job_id}`
- `POST /api/analyze/{job_id}`
- `POST /api/reconstruct/{job_id}`
