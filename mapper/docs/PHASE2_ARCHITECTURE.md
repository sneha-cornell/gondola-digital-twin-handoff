# Phase 2 Architecture Plan: Multi-Class Detector, Fixture Detection, Real-Time Path

Status: planning document, written 2026-07-21. Phase 1 (class registry, per-stage
pipeline observability, `--class-labels` dataset builder) is merged — see
`backend/product_classes.py`, `backend/product_identifier.py`'s `pipeline_status()`,
and `training/build_workspace_dataset.py`. This document plans the next phase: a
real multi-class detector, fixture detection, and a real-time inference path.

## Ground-truth findings

**There is currently only one real physical scene in the entire system.**
`backend/workspace/` has 9 job folders, but `iphone16-2`, `iphone16-2_clean_v5_job`,
`iphone16-2_detect_high_recall_v1_job`, `iphone16-2_exact_clean_v5_job`,
`iphone16-2_exact_clean_v6_job`, `iphone16-2_named_high_recall_v2_job`,
`iphone16-2_named_high_recall_v3_job`, `iphone16-2_named_high_recall_v4_job`, and
`later_seed_probe_v1` all reference the identical 59 photos (`00001.jpg`…`00059.jpg`)
of one micro-market snack aisle. Every job name is a different *experiment
configuration* re-run against the same capture, not a different shelf. The scene is
one gondola run of dry snacks (chips/pretzels/popcorn) plus one hook/pegboard endcap
holding hanging bags, and one ambiguous red boxy object in the background that might
be a cooler or a kiosk — not clearly identifiable from the images alone, worth
checking on-site.

**Only one job has a completed identification pass, and as of this writing it
predates Phase 1's schema** (no `class_records` key, no `class_id`/`class_name` on
individual products) — `iphone16-2_named_high_recall_v4_job/recognition_results.json`
was written by an older run. Its `summary` block: `named_product_count: 561` out of
3922, sourced only from `embedding_classifier` (500) and `clip_classifier` (61) — OCR
and VLM/open-world contributed zero, because no OCR engine and no Anthropic key are
configured in this environment. Re-running `test_recognition.py` against current code
is step 0 below.

Recomputing with a 0.65 identity-confidence floor (the default
`--min-identity-confidence` in `build_workspace_dataset.py --class-labels`): **516
usable boxes across 23 of the 32 registered classes.** Distribution is severely
long-tailed:

```
180 Kettle Brand Dill Pickle              8  Kettle Brand Riffles
135 Reese's Popcorn                       8  Natierra Organic Strawberries
 39 Dot's Homestyle Pretzels              7  Snyder's Pretzel Sticks
 31 Kettle Brand Pepperoncini             7  Popchips Sea Salt & Vinegar
 26 Crunchmaster Multi-Seed               6  Wilde Protein Chips
 19 New York Style Panetini               5  Mavuno Harvest Organic Papaya
 15 Popcorn for the People S&S Kettle     4  Popchips BBQ / Mavuno Jackfruit
  8 Kettle Brand Krinkle Cut Dill         3  Mavuno Coconut / Fruit Bliss Apricots
                                          2  PopCorners / Popchips Sweet Heat / Dot's BBQ
                                          1  Popchips / Wilde Nashville Hot
```

9 of the 23 represented classes have ≤3 examples; the other 9 registered catalog
classes have zero confidently-identified detections in this job at all.

Catalog reference imagery (`backend/knowledge_base/<product>/*.jpg`) is thin and
clean-background: 2–42 images per class (median ~5), scraped hero shots — useful for
embedding/CLIP matching and as a synthetic-compositing source, not as detector
training data directly.

## 1. Training data reality check

516 boxes across 23 classes from one scene is enough to validate the training
*pipeline*, and not enough to ship a production multi-class detector. Two independent
problems compound here:

- **Class imbalance** (180 vs 1) — fixable with data engineering (see §2).
- **Zero scene diversity** — not fixable with data engineering. Every box comes from
  the same lighting, same fixture, same camera, same 59 overlapping COLMAP
  walk-around frames. `training/build_workspace_dataset.py`'s `stable_split()` hashes
  on `image_rel` (job/filename) for an 80/20 split, but because the 59 frames are
  dense multi-view photogrammetry captures (heavy frame-to-frame overlap, by design),
  a hash-based per-image split puts near-duplicate viewpoints of the same physical
  products on both sides. Any validation mAP from this dataset alone is inflated and
  does not indicate generalization to a different shelf, angle, or store.

**Recommendation:** run the first multi-class fine-tune now, as a pipeline-validation
smoke test — does `--class-labels` → `train_custom.py` → eval work end to end — but
treat it as go/no-go, not a shippable checkpoint. The real Phase 2 blocker is
capturing 5–10 more independent scenes of the same ~32-SKU catalog: physical-capture
time measured in days, not engineering time. Synthetic compositing (§5, task 4) fixes
the tail-class-count problem but not the scene-diversity problem — do not substitute
one for the other.

## 2. Model choice & training strategy

**Fine-tune from `backend/models/best.pt`, not from a stock COCO checkpoint or a
fresh SKU-110K run.** `best.pt` is already a single-class (`nc:1`, "object") detector
tuned for dense close-range packaged-shelf localization (`detector.py` runs it at
`imgsz=960`, `confidence=0.10`, favoring recall). Its box-localization prior is
exactly the domain needed; only the classification head needs to learn the
fine-grained split. Starting from `yolo11l.pt` COCO weights throws that prior away.
The SKU-110K path (`training/finetune_sku110k.py`) is a good precursor for `best.pt`
itself (box quality) but is the wrong tool for the multi-class step — SKU-110K has no
product-identity labels at all, it's box-only.

**Class-imbalance handling, concretely:**

1. Don't give a dedicated class id to any SKU with <10 confidently-identified boxes.
   With today's data that's 9 of the 23 represented classes. Fold their boxes into a
   single generic `product` catch-all class in the training labels — a small change
   to `load_class_boxes()` in `training/build_workspace_dataset.py` (map
   `resolved is None or count_by_class[class_id] < threshold` to class 0/"product"
   instead of dropping).
2. For the 10–40-example middle tier (Kettle Krinkle Cut, Snyder's, Natierra,
   Popchips S&V, Wilde), oversample by repeating the image entries containing their
   boxes in the dataset manifest — Ultralytics has no native class-balanced sampler,
   so this must happen at dataset-construction time, not the loss level.
3. Loss-side levers (`cls=` gain, label smoothing) are a distant second-order fix
   given how sparse the data is — the missing ingredient is examples, not gradient
   weighting.
4. Read Ultralytics' per-class validation output (`model.val()` reports per-class
   P/R) after every run and use it, not overall mAP, to decide which classes graduate
   to a dedicated id.

**Two-stage fallback (recommended, as a triage, not a full pipeline replacement):**

- Train the multi-class head only on classes crossing the ≥30-confident-box threshold
  (currently ~6: Kettle Dill Pickle, Reese's Popcorn, Dot's Pretzels, Kettle
  Pepperoncini, Crunchmaster, New York Style Panetini) plus one generic `product`
  class for everything else.
- At inference, per detection box: if the predicted class is a specific SKU and box
  confidence is above a threshold (e.g. 0.6), trust it directly and skip
  identification entirely — this is the real-time win, see §4. If predicted class is
  the generic `product` bucket, or confidence is below threshold, route that crop
  through the existing `product_identifier.py` chain as today.
- Needs one new state in `detector.py`'s identity-labeling logic (currently a binary
  `product_identity_labels` flag per model) — a "partial identity" mode where some
  boxes carry a trusted label and others still need
  `ProductIdentifier.enrich_detections`.
- Because `product_classes.py`'s `ClassRegistry` ids are append-only and stable
  (`_sync_with_catalog` only ever adds, never renumbers), this promotion (moving a
  class from "fallback only" to "trained head") can happen incrementally across
  retrains without invalidating past `results.json`/Blender artifacts that reference
  class ids.

## 3. Fixture detection

**Concrete approach: YOLO-World as a second, full-frame open-vocabulary detection
pass.** Verified: the project `.venv` already has `ultralytics==8.3.161` installed and
`from ultralytics import YOLOWorld` imports cleanly — no new dependencies needed.

- **Afternoon of work:** add `backend/fixture_detector.py` mirroring `detector.py`'s
  pattern — load a `yolov8s-worldv2.pt`-class checkpoint once, `model.set_classes([...])`
  with a prompt list (`"refrigerator"`, `"cooler door"`, `"freezer case"`, `"shopping
  basket"`, `"dump bin"`, `"endcap"`, `"pegboard hook fixture"`, `"gondola shelf"`),
  run `model.predict()` per full image (not per crop — fixtures are frame-scale), and
  cache results the same way as `job_dir/detections/*.json` (e.g.
  `job_dir/fixtures/*.json`). Can be validated against the existing 59-image capture
  today.
- **Needs new capture data:** validating that it actually finds
  refrigerators/coolers/dump bins requires images that contain one clearly and
  unambiguously. The one scene in this system is a dry-goods gondola + a pegboard
  endcap — worth running the open-vocab pass on it now (should at least find the
  endcap/gondola shelving, and confirm or rule out the ambiguous red object), but
  "fixture detection works" as a general claim needs a capture session somewhere with
  an actual cooler/freezer case in frame.
- **Needs geometry-layer changes (the real work, not the detector):** `geometry.py`
  today has zero fixture concept — it only fits `Shelf` planes (`detect_shelf_planes`,
  `detect_shelf_planes_from_mesh`) and generates rack meshes from a **manually
  selected** enum (`main.py`'s `FIXTURE_TEMPLATES = ("gondola", "wall_bay")`, chosen
  via a frontend dropdown, not detected). To turn 2D fixture boxes into something
  useful: (1) define a `Fixture` record (type, 3D bounding volume, contained shelf
  ids) distinct from `Shelf`; (2) reuse the existing `pixel_to_world_ray` /
  `_intersect_ray_with_shelf` primitives to project fixture boxes into the point cloud
  the same way `project_detections_to_shelves` already does for products; (3)
  associate detected shelves/products to whichever fixture volume contains them; (4)
  replace the manual `fixture_template` dropdown with an inferred default from
  detected fixture type. Roughly 2–3 days. (5) Beyond that, actually placing new 3D
  fixture geometry in Blender for cooler/basket/dump-bin types is new asset-authoring
  work — today only gondola/wall-bay meshes exist (`build_rack_geometry` in
  `geometry.py`), and there's no reusable pattern for e.g. a cooler-door mesh. Budget
  1–2 weeks for that piece, as separate, lower-priority scope from "detect that a
  fixture exists."

## 4. Real-time path

**The bottleneck is precisely located: `product_identifier.py`'s
`enrich_detections()` processes crops one at a time with no batching anywhere in the
embedding/CLIP stages** — each iterates `for det in enriched: ... Image.open(resolved)
... classifier.classify(single_image)`. With 3922 raw detections in one 59-image job,
that's thousands of individual `Image.open` + single-image forward passes, plus a
thread-pooled OCR stage that's subprocess/API-bound. This is the dominant cost once
COLMAP/dense reconstruction has already run — a Python-loop-and-dispatch problem, not
fundamentally a model-speed problem.

**What's achievable:** collapsing detection + identification into one multi-class
YOLO forward pass is the real fix, not micro-optimizing CLIP. On the verified
hardware (NVIDIA A10G, `torch 2.6.0+cu124`, CUDA confirmed available), a YOLO11-family
detector at batch=1 runs in the ~15–30ms range per published Ultralytics benchmarks at
similar resolutions — compatible with 20–30 FPS live inference for whatever classes
the multi-class head knows directly, because it skips the entire
OCR→embedding→CLIP→VLM chain for those boxes.

**Minimum changes for a usable "live" mode** (distinct from the batch/offline
digital-twin build):

1. Ship the trained multi-class detector + the two-stage triage from §2.
2. Add a new lightweight endpoint (e.g. `POST /api/v1/live/frame`) that runs *only*
   `ObjectDetector`-style inference + direct class lookup on one incoming frame —
   explicitly bypassing `_run_colmap_pipeline`, `run_dense_pipeline`,
   `detect_shelf_planes*`, and the full `enrich_detections` chain (all of which
   `main.py`'s `analyze_job()` currently runs unconditionally in sequence). Returns 2D
   boxes + labels only.
3. The frontend has no live/camera code today (no `getUserMedia`/video/streaming in
   `frontend/src/App.jsx`). Adding a live view is new frontend work (camera capture +
   canvas overlay + polling or streaming to the new endpoint), not a tweak to an
   existing component.
4. For SKUs the multi-class head doesn't cover, live mode should show a generic
   "product" box or, at most, run a CLIP-only fallback — which needs a
   `classify_batch` method added to `clip_classifier.py` (doesn't exist today) to keep
   up with live frame rate. OCR, VLM, and open-world stages must not run in the live
   path at all — they're subprocess/API calls costing 100ms to multiple seconds each,
   unbounded by design.

**Stays batch/offline, unchanged:** COLMAP sparse reconstruction,
`dense_reconstruction.py`'s DepthAnything+Open3D TSDF pass, mesh-based shelf-plane
detection, and all the `blender_*.py` placement scripts. These build the digital twin
once per capture session; live mode is a separate, parallel "point a camera and see
labels" capability, not a replacement for the twin-building workflow.

## 5. Sequencing (ordered, with effort sizing)

| # | Task | Effort | Depends on |
|---|------|--------|------------|
| 0 | Re-run `test_recognition.py` on `iphone16-2_named_high_recall_v4_job` under current Phase-1 code so `recognition_results.json` actually has `class_id`/`class_records` (today's on-disk file predates Phase 1) | 0.5 day | — |
| 1 | Run `training/build_workspace_dataset.py --class-labels` against it, inspect the printed label-count/imbalance summary | 0.5 day | 0 |
| 2 | First multi-class fine-tune from `best.pt`, restricted to the ~6 classes with ≥30 boxes + one generic `product` class for the rest — **pipeline validation, not a ship candidate** | 1 day | 1 |
| 3 | Capture 5–10 additional independent scenes (different shelf/lighting/angle) of the same ~32-SKU catalog; re-run detection+identification per scene | days–weeks, physical capture time, run in parallel with everything else | — |
| 4 | Build a synthetic-compositing script: paste `knowledge_base/<product>/*.jpg` reference crops onto real shelf background patches to bulk up the 9 classes with <10 real examples | 2–3 days | — (independent of 3) |
| 5 | Stand up `backend/fixture_detector.py` using `ultralytics.YOLOWorld` (already installed) over full frames; validate against the existing capture, check whether it even contains a real refrigerator/cooler | 0.5 day | — |
| 6 | Wire fixture boxes into `geometry.py`: new `Fixture` record, reuse `pixel_to_world_ray`/shelf-projection pattern, replace the manual `fixture_template` dropdown with detected type | 2–3 days | 5 |
| 7 | Author new Blender assets/placement for cooler/basket/dump-bin fixture types (only gondola/wall_bay exist today) | 1–2 weeks | 6 |
| 8 | Implement the two-stage triage in `detector.py`/`product_identifier.py`: trust high-confidence known-class boxes, route the rest through the existing chain; add "partial identity" labeling state | 2–3 days | 2 (with a real, imbalance-aware checkpoint) |
| 9 | Minimum live-mode slice: detector-only endpoint bypassing COLMAP/dense/OCR/VLM, plus new frontend camera view (no existing code to build on) | 3–5 days | 8 |
| 10 | As more scenes land (3), periodically re-evaluate per-class box counts against the ≥30 threshold and expand/retrain the multi-class head — safe incrementally since `ClassRegistry` ids are append-only | ongoing | 3 |

### Critical files for implementation

- `training/build_workspace_dataset.py`
- `backend/product_classes.py`
- `backend/detector.py`
- `backend/product_identifier.py`
- `backend/geometry.py`
- `backend/main.py`
- `training/train_custom.py`
