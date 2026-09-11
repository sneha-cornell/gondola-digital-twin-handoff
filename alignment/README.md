# Gondola scan-to-mesh pipeline

Four scripts that take a COLMAP sparse reconstruction of a gondola fixture
and turn it into a clean, editable parametric `.blend` model — no manual
alignment/scaling in Blender.

## Pipeline order

```
COLMAP sparse model (images.txt, points3D.txt)
        |
        v
isolate_subject.py          -- strips background clutter (walls, floor,
                                neighboring fixtures) from the raw point
                                cloud using camera-geometry filters only
        |
        v
build_gondola_from_scan.py  -- fits an oriented rectangle + shelf-height
                                bands to the isolated points, converts
                                COLMAP's arbitrary scale to meters using
                                one measurement you provide, and calls
                                generate_parametric_gondola.py with the
                                fitted numbers
        |
        v
generate_parametric_gondola.py  -- builds the actual box-mesh geometry
                                    (called automatically by the script
                                    above; can also be run standalone with
                                    hand-entered dimensions)
        |
        v
export_blender_scene.py     -- imports the resulting JSON into Blender
                                and saves a .blend
```

## Quick start

```bash
# 1. Fit dimensions + generate the JSON from a COLMAP TXT export
python3 build_gondola_from_scan.py \
  --colmap-txt path/to/sparse_txt \
  --anchor-dimension height --anchor-meters 1.8 \
  --output parametric_gondola.json

# 2. Turn the JSON into a .blend
blender --background --python export_blender_scene.py -- \
  --results parametric_gondola.json \
  --output gondola_fixture.blend
```

`--colmap-txt` needs a folder with `images.txt` and `points3D.txt` — export
one from a COLMAP binary model with:
```bash
colmap model_converter --input_path sparse/0 --output_path sparse/0_txt --output_type TXT
```

**`--anchor-dimension`/`--anchor-meters` are required and matter a lot.**
COLMAP's monocular reconstructions have no inherent scale — everything is
in an arbitrary, self-consistent unit system. You must supply ONE real
measurement (a shelf height you measured, a known width, anything) so the
script can convert to meters. Get this wrong or guess it, and every other
dimension in the output scales off that same error.

If you don't have a real measurement yet, add `--anchor-assumed` alongside
a placeholder value (e.g. a typical 1.8m gondola height) — this stamps the
output JSON's `fit_from_scan.anchor_assumed` field so nobody downstream
mistakes the placeholder for a real measurement.

## If the rectangle fit fails or falls back

`build_gondola_from_scan.py` needs the capture to walk most of the way
around the object — both long faces AND both end-caps. If end-cap coverage
is too thin, it can't find a real 4-sided rectangle and falls back to a
rough oriented-bounding-box estimate, printing a `CAUTION` line and setting
`fit_confidence: "partial_coverage_fallback"` in the output JSON. In that
case, only the well-covered axis is trustworthy — recapture the missing
angles and re-run for a real fit.

You can also override anything the geometry detected — bay width/count,
per-side shelf counts, endcap depth, price rails, top canopy — see
`build_gondola_from_scan.py --help`.

## What's in `output/`

`gondola_from_aws_scan.blend` + `gondola_from_aws_scan.json` — built by
running `build_gondola_from_scan.py` against the real AWS COLMAP capture in
`aws_run_outputs/text/` (97/120 images registered, 60,784 points).

**Read before trusting the numbers:**
- Length (7.09 m) — solid, from the two well-covered long faces.
- Height (1.80 m) — **assumed**, not measured. No real anchor was available
  from the AWS side either (its `layout_config.json` had `reference_width_m`
  and `reference_depth_m` both null), so this used a typical retail-gondola
  height as a placeholder. Everything else scales off this one number.
- Width (1.37 m) and end-cap shelf count (6) — **low confidence**. The
  capture has a real ~60 degree gap in camera coverage around the object
  (roughly 150-210 degrees, zero photos), so `fit_confidence` in the JSON is
  `"partial_coverage_fallback"` — no true 4-sided rectangle was found, this
  came from a rough oriented-bounding-box fallback instead.

To get a trustworthy version: recapture the missing end-cap angles, re-run
COLMAP, then re-run `build_gondola_from_scan.py` with a real
`--anchor-meters` value (not `--anchor-assumed`).

## What's in `aws_run_outputs/`

Pulled from an actual AWS COLMAP run of this gondola (the same one behind
`output/`, before it went through `build_gondola_from_scan.py`):

- `text/` — the portable COLMAP sparse model (`cameras.txt`, `images.txt`,
  `points3D.txt`). This is the direct input to `build_gondola_from_scan.py`
  above.
- `mesh_output/` — a raw (non-parametric) preview: `sparse_points.ply` is
  the sparse point cloud, `gondola_sparse_mesh.ply`/`gondola.blend` is that
  cloud imported into Blender as a point-derived mesh (via `import_mesh.py`),
  plus render preview PNGs and the small scripts (`import_mesh.py`,
  `render_check.py`, `render_points.py`) that produced them. These are
  useful for eyeballing the raw scan, but they are NOT the parametric
  fixture -- for that, use `output/`.

## Aligning product coordinates to the fixture

`digital-twin-shelf-mapper` emits product positions (`results.json` ->
`products[].p3d`) in the **raw COLMAP world frame** of its own scan, and its
`apply_layout_model()` only ever applies a uniform scale to that frame — and
only when `layout_config.json` supplies `reference_width_m`/`reference_depth_m`.
With both null it returns scale `1.0` / `"scene_units"`: untouched COLMAP units,
arbitrary origin, arbitrary orientation.

This pipeline emits the fixture in a canonical frame instead (+Z up with 0.0 at
the floor, +X along the bay run centred on 0.0, +Y depth, meters). So the two
differ by an unknown rotation, translation and scale, and matching their
bounding-box min/max cannot recover it: the product cloud's extremes include
background this pipeline deliberately filtered out, and extents are invariant
to the 180° flip about up, so the front and back faces score identically.

Two pieces close that gap:

**`frame_alignment.py`** — recovers the COLMAP→model similarity transform from a
reconstruction, and `build_gondola_from_scan.py` now writes it into the output
JSON as `colmap_alignment`:

```json
"colmap_alignment": {
  "convention": "p_model = scale * R @ (p_colmap - origin_colmap)",
  "scale_meters_per_colmap_unit": 1.147,
  "rotation_model_from_colmap": [[...], [...], [...]],
  "origin_colmap": [...],
  "matrix_4x4_model_from_colmap": [[...]]
}
```

Anything holding coordinates from **that same reconstruction** can now be mapped
into the mesh's frame exactly — no manual nudging in Blender, no guessed anchors.
Run it standalone with `python3 frame_alignment.py --colmap-txt <dir>
--anchor-dimension height --anchor-meters 1.8`.

**`align_products_to_gondola.py`** — for when the products came from a
*different* scan of the same fixture, where no shared transform exists. It
rebuilds the fixture frame from the product side's own detected shelf planes and
solves scale + floor by matching the two shelf-height sets (1-D ICP):

```bash
python3 align_products_to_gondola.py \
  --results    /path/to/workspace/<job>/results.json \
  --colmap-txt /path/to/workspace/<job>/text \
  --gondola    output/gondola_from_aws_scan.json \
  --output     output/gondola_with_products.json
```

It reports a residual at every step — shelf-match RMS, scale drift versus the
seed, how many products actually seat on a board — so a bad alignment says so
instead of quietly misplacing products. `test_alignment_roundtrip.py` checks it
against a synthetic scan with a known transform (recovers scale to <0.1% and
position to ~2.8 cm at 1 cm input noise).

Two things it cannot resolve on its own:

- **Which long face is Side A.** A double-sided gondola is symmetric under a
  180° turn about up. The tool tries both and keeps whichever seats more
  products, but warns when the two score alike; settle it with `--flip on/off`
  after checking one product against a reference photo.
- **Absolute scale**, if your `--anchor-meters` was assumed rather than
  measured. Every distance inherits that error proportionally.

### Shelf heights are now measured, not evenly spaced

`build_gondola_from_scan.py` used to pass only the *count* of detected shelf
bands to the generator, which then spread that many boards evenly across the
fixture height. Real fixtures are not evenly spaced, so every board landed a few
centimeters off its true plane — enough that products positioned from the same
scan float above or sink through the shelf they actually rest on. The detected
band heights are now passed through, and `layout.shelf_heights_source` records
`"measured"` or `"even_spacing"`. Pass `--even-shelf-spacing` for the old
behaviour, or override a side's shelf count to fall back to even spacing for
that side.

## Standalone use of `generate_parametric_gondola.py`

If you already know the exact real-world dimensions (no scan needed):

```bash
python3 generate_parametric_gondola.py \
  --bay-width 1.2 --depth 0.8 --height 1.8 \
  --side-a-shelves 6 --side-c-shelves 6 \
  --side-b-shelves 5 --side-d-shelves 5 \
  --output parametric_gondola.json
```
