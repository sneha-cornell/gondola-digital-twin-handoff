# Gondola Digital Twin — handoff package

This folder combines two previously separate codebases plus the glue code
written to connect them, so someone else can test the pipeline and open the
result in Blender without hunting across three different folders on a
different machine.

```
gondola-digital-twin-handoff/
├── mapper/        digital-twin-shelf-mapper — detection + shelf-plane +
│                  ray-casting backend. Real repo, unmodified, includes
│                  real model weights and one real completed job.
├── alignment/     Scan↔fixture frame alignment (colmap_alignment transform,
│                  1-D shelf-height ICP, measured shelf heights, planogram
│                  render/report tools). Written last week, verified against
│                  synthetic round-trip data.
├── bridge/        This week's scripts: run mapper code + last week's
│                  alignment code together against a real photo. Two
│                  approaches, see "What's real" below.
├── data/full_scan/  Sparse COLMAP model for the full 72-image capture
│                  (images themselves NOT included — see the .txt inside).
└── README.md      This file.
```

## The one real, fully-baked example: `single_view_00014`

`mapper/backend/workspace/single_view_00014/` is a **complete real job**,
already run, sitting in the repo:

- `detections/00014.json` — 58 real bounding boxes from the real detector
  (`backend/models/best.pt`, an actual YOLO checkpoint, confidence scores
  included). Not hand-drawn, not synthetic.
- `results.json` — the mapper's real `geometry.py` output: 2 detected shelf
  planes, 35 products (after multi-detection consensus merging), each with a
  real `p3d`, `shelf_local {u, v, n}`, and `shelf_id`.
- `aligned/gondola.json`, `aligned/gondola_with_products.json`,
  `aligned/alignment_report.json` — last week's `align_products_to_gondola.py`
  already run against this real job's real shelves and products.
- `crops/`, `focus_crops/` — the actual cropped product images per detection.
- `images/00014.jpg` — the actual source photo.

**Start here.** Everything else in this repo either produced this, or is an
alternative way of producing something like it.

## What's real vs. what's a stand-in

| Component | Status |
|---|---|
| YOLO detections in `single_view_00014` | **Real.** Actual `best.pt` inference. |
| Shelf planes in `single_view_00014/results.json` | **Real.** Actual `geometry.py::detect_shelf_planes()`. |
| Product→shelf assignment in `results.json` | **Real.** Actual `geometry.py::project_detections_to_shelves()`. |
| `aligned/gondola_with_products.json` | **Real.** Actual `align_products_to_gondola.py` run against the above. |
| `bridge/work/plot_real_shelf_pipeline.py` | Real `geometry.py` + real `isolate_subject.py`, but **hand-drawn boxes** (torch/ultralytics weren't installed when this was built — see Known Issues). Superseded by `single_view_00014` above for anything that job already covers. |
| `bridge/work/plot_center_edge_positions.py` | Real COLMAP parser + real 3D points, but **fully custom** shelf-axis/edge math (not `geometry.py`). Hand-drawn boxes. Kept for its center+edge extraction, which `results.json` doesn't compute. |
| `data/full_scan/` | Real sparse COLMAP model, images not included (see the `.txt` in that folder). |

## Setup

```bash
cd mapper/backend
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt          # heavy: torch, ultralytics, paddleocr, transformers
```

Model weights are already in the repo (`mapper/backend/models/best.pt`,
`mapper/yolov8n.pt` fallback) — no download needed to re-run detection.

For just the alignment/analysis scripts (no detector, much lighter):

```bash
pip install numpy scipy scikit-learn Pillow
```

`ultralytics`/`torch` were **not installed** while building this handoff
(declined for time/cost) — anything that needs a live detector run was
either skipped or backed by the pre-computed `single_view_00014` job instead.

## Running things

**Inspect the real job (no setup needed beyond `numpy`):**
```bash
python3 -c "import json; d=json.load(open('mapper/backend/workspace/single_view_00014/results.json')); print(d['shelves'][0]['id'], d['products'][0])"
```

**Re-run the alignment step against a different gondola fit:**
```bash
cd alignment/code
python3 build_gondola_from_scan.py --colmap-txt ../../data/full_scan --anchor-dimension height --anchor-meters 1.8 --anchor-assumed --output /tmp/gondola.json
python3 align_products_to_gondola.py --results ../../mapper/backend/workspace/single_view_00014/results.json --gondola /tmp/gondola.json --output /tmp/aligned.json --report /tmp/report.json
python3 render_planogram.py --aligned /tmp/aligned.json --output /tmp/planogram.svg
```

**Round-trip test the alignment math on synthetic data (sanity check, no real data needed):**
```bash
cd alignment/code
python3 test_alignment_roundtrip.py --gondola ../output/gondola_from_aws_scan.json
```

## Getting into Blender

Two source trees have Blender-facing code:

1. **`alignment/code/export_blender_scene.py`** — takes a gondola JSON
   (`generate_parametric_gondola.py` / `build_gondola_from_scan.py` output,
   or `single_view_00014/aligned/gondola_with_products.json`) and builds a
   `.blend`:
   ```bash
   blender --background --python alignment/code/export_blender_scene.py -- \
     --results mapper/backend/workspace/single_view_00014/aligned/gondola_with_products.json \
     --output /tmp/single_view_00014.blend
   ```
2. **`mapper/backend/blender_*.py`** (19 scripts) — the mapper repo's own
   Blender tooling: importing/replicating shelf layouts, filling gaps,
   coloring materials, generating shelf reports, etc. These read as an
   iteration history (`blender_fill_existing_shelf_gaps.py` →
   `blender_fill_full_existing_shelf_slots.py` →
   `blender_fill_image_occupancy_slots.py`, etc.) rather than one blessed
   entry point. **Not individually vetted for this handoff** — read each
   script's own docstring/argparse help before running; `blender_project_from_images_colmap.py`
   and `blender_scene_shelf_report.py` looked like the most load-bearing ones
   on a skim, but confirm before relying on that.

## Known issues (carried over + newly found)

- **Single-image jobs only resolve ~2 broad shelf planes**, not the ~7-8
  physical shelves. Confirmed twice independently: `single_view_00014`'s real
  pipeline run got 2 shelves, and re-running `detect_shelf_planes()` fresh in
  `bridge/work/plot_real_shelf_pipeline.py` also got 2. `main.py`'s intended
  path runs shelf detection on a **dense** reconstruction (DepthAnything +
  Open3D TSDF), which needs `torch`; without it, `detect_shelf_planes()` sees
  the much sparser/noisier raw COLMAP points and can't resolve fine shelf
  spacing. Confirmed by the mapper's own render script warning:
  `shelf normal vs camera-up = 0.684` on this job — well below a healthy
  alignment. **Multi-view capture + the dense path is the real fix**, not a
  parameter tweak.
- **Metre-sized thresholds on non-metric units.** `geometry.py` has several
  constants written as if COLMAP units were meters (`DBSCAN eps=0.12`,
  `OBSERVED_SHELF_MATCH_MAX_HEIGHT_DELTA_M=0.22`, `MIN_SHELF_AREA_M2=0.20`).
  They happen to behave reasonably on this scan because its fitted scale is
  close to 1 unit ≈ 1 m — that's a property of this scan, not something the
  code enforces. A scan at a different scale would need these re-tuned.
- **Background-point contamination in bbox-based 3D positioning.** When a
  box spans a gap between products, real reconstructed points from the wall/
  ceiling behind the shelf fall inside it and corrupt a naive median/percentile
  position. Found and fixed in `bridge/work/plot_center_edge_positions.py`
  (`reject_background()`, 0.6-unit radius, validated against every box in
  this dataset) — **not yet ported into `geometry.py` itself.**
- **A small latent bug in `geometry.py::_normalize()`**: it guards an exact-
  zero vector norm but not a near-zero one, so a degenerate RANSAC sample
  occasionally produces a near-infinite unit vector (`RuntimeWarning: divide
  by zero / overflow in matmul`). Confirmed real with `np.seterr(all="raise")`
  — not just a spurious warning. Appears harmless to the final result here
  (NaN distances always lose the "most inliers" comparison) but is real,
  unfixed code.
- **Fixture height is still assumed, not measured** (`1.8 m`, `anchor_assumed:
  true` in every gondola JSON in this repo). Every absolute distance this
  pipeline reports scales with that one assumption.
- **Which long face is "Side A"** is not determined by geometry on a
  symmetric double-sided gondola — `align_products_to_gondola.py --flip
  on/off` needs a reference photo to resolve, not more data from the same scan.

## Provenance

- `mapper/` — cloned from the `digital-twin-shelf-mapper` repo
  (`engineersneha/digital-twin-shelf-mapper`, private).
- `alignment/` — written last week against a COLMAP scan of this gondola
  (`data/full_scan`), verified with a synthetic round-trip test (scale
  recovered to ±0.06%, position to ~2.8 cm at 1 cm input noise).
- `bridge/` — written this week, real 3D data throughout, hand-drawn boxes
  where noted above.
