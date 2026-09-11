"""
Automatically isolate the photographed subject (e.g. a gondola fixture) from a
COLMAP sparse reconstruction, separating it from background clutter (walls,
floor, neighboring fixtures) using only geometric signals already present in
the reconstruction -- no manual cropping, no ML segmentation model.

Requires: a COLMAP model exported to TXT
    colmap model_converter --input_path sparse/0 --output_path sparse/0_txt --output_type TXT

Three independent geometric filters are combined:

1. Camera-loop radius filter
   Photographers walk around the subject, so camera centers form a rough loop.
   Points far outside that loop (a distant wall, the far end of a room) are
   background.

2. Viewpoint-diversity filter
   A point on the real subject gets photographed from many different angles
   (people circle it). A point on a flat background surface (e.g. a wall) is
   usually only seen from a narrow range of angles. For each point, average
   the unit vectors from the point to every camera that observed it -- a
   small resultant magnitude means the viewing angles were spread out
   (diverse); a large resultant magnitude means they were bunched together
   (likely background).

3. Height-band filter
   The subject occupies a compact, continuous vertical range. Background
   clutter (ceiling fixtures, floor extending into the distance) tends to be
   sparse or discontinuous in height. Keep only the single largest
   contiguous vertical band whose point density stays above a threshold
   fraction of the peak.

Usage:
    python3 isolate_subject.py --colmap-txt sparse/0_txt --output isolated.ply \
        --preview-prefix /tmp/isolated

Then measure the printed bounding box in raw (arbitrary-scale) units and
anchor it to one known real-world measurement, same as before.
"""

from __future__ import annotations

import argparse
import struct
import zlib
from pathlib import Path

import numpy as np


def parse_images_txt(path: Path):
    images = {}
    lines = [l for l in path.read_text().splitlines() if l and not l.startswith("#")]
    for i in range(0, len(lines), 2):
        parts = lines[i].split()
        img_id = int(parts[0])
        qw, qx, qy, qz = map(float, parts[1:5])
        tx, ty, tz = map(float, parts[5:8])
        images[img_id] = dict(q=(qw, qx, qy, qz), t=np.array([tx, ty, tz]))
    return images


def quat_to_R(q):
    w, x, y, z = q
    return np.array([
        [1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * z * w, 2 * x * z + 2 * y * w],
        [2 * x * y + 2 * z * w, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * x * w],
        [2 * x * z - 2 * y * w, 2 * y * z + 2 * x * w, 1 - 2 * x * x - 2 * y * y],
    ])


def parse_points3d_txt(path: Path):
    pts, cols, tracks = [], [], []
    for line in path.read_text().splitlines():
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        pts.append([float(parts[1]), float(parts[2]), float(parts[3])])
        cols.append([int(parts[4]), int(parts[5]), int(parts[6])])
        track_ids = parts[8:]
        image_ids = [int(track_ids[i]) for i in range(0, len(track_ids), 2)]
        tracks.append(image_ids)
    return np.array(pts), np.array(cols, dtype=np.uint8), tracks


def camera_centers_and_up(images: dict):
    ids = list(images.keys())
    centers = {}
    ups = []
    for img_id in ids:
        d = images[img_id]
        R = quat_to_R(d["q"])
        C = -R.T @ d["t"]
        centers[img_id] = C
        ups.append(R.T @ np.array([0.0, -1.0, 0.0]))
    ups = np.array(ups)
    mean_up = ups.mean(axis=0)
    mean_up /= np.linalg.norm(mean_up)
    alignment = (ups @ mean_up).mean()
    return centers, mean_up, alignment


def build_basis(up: np.ndarray):
    tmp = np.array([1.0, 0.0, 0.0]) if abs(up[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    u0 = np.cross(up, tmp)
    u0 /= np.linalg.norm(u0)
    v0 = np.cross(up, u0)
    v0 /= np.linalg.norm(v0)
    return u0, v0


def proximity_mask(pts, tracks, cam_centers: dict, keep_percentile=70.0):
    """Keep points close to the cameras that actually photographed them.

    A 'camera loop centered on the subject' assumption breaks for a subject
    that can only be approached from one side (a wall of shelving): the
    camera centroid ends up offset from the wall by the standoff distance,
    not centered on it, so a centroid-based radius filter is measuring the
    wrong thing.

    Instead, for each point, look at the mean distance to *its own observing
    cameras*. People photograph the real subject up close; a background
    point glimpsed at the edge of a frame is typically far from the cameras
    that happened to observe it. Relative (percentile) cutoff, same
    self-calibrating reasoning as the diversity filter.
    """
    dist = np.full(len(pts), np.inf)
    for i, track in enumerate(tracks):
        ds = [np.linalg.norm(cam_centers[img_id] - pts[i]) for img_id in track if img_id in cam_centers]
        if ds:
            dist[i] = float(np.mean(ds))
    finite = np.isfinite(dist)
    cutoff = np.percentile(dist[finite], keep_percentile)
    mask = finite & (dist <= cutoff)
    return mask, dist, cutoff


def compute_diversity(pts, tracks, cam_centers: dict, min_track_len=3):
    """Per-point viewpoint diversity in [0,1]. 0 = all observing cameras clustered
    in one direction from the point; 1 = observing directions perfectly spread out.
    Note: a subject that can only be photographed from one side (e.g. a flat wall
    of shelving) has a naturally low ceiling here -- use a *relative* (percentile)
    cutoff against this same distribution, not an absolute one.
    """
    diversity = np.zeros(len(pts))
    track_len = np.array([len(t) for t in tracks])
    has_track = track_len >= min_track_len
    for i in np.nonzero(has_track)[0]:
        p, track = pts[i], tracks[i]
        dirs = []
        for img_id in track:
            c = cam_centers.get(img_id)
            if c is None:
                continue
            v = c - p
            n = np.linalg.norm(v)
            if n > 1e-9:
                dirs.append(v / n)
        if len(dirs) < min_track_len:
            has_track[i] = False
            continue
        dirs = np.array(dirs)
        resultant = np.linalg.norm(dirs.sum(axis=0)) / len(dirs)
        diversity[i] = 1.0 - resultant
    return diversity, has_track


def viewpoint_diversity_mask(pts, tracks, cam_centers: dict, min_track_len=3, keep_percentile=40.0):
    """Keep points whose diversity is at or above `keep_percentile` among points
    with enough observations to score. Relative, so it self-calibrates to whatever
    viewing spread the actual capture pattern achieved (full walk-around vs.
    front-only approach to a wall)."""
    diversity, has_track = compute_diversity(pts, tracks, cam_centers, min_track_len)
    if not has_track.any():
        return np.zeros(len(pts), dtype=bool), diversity, 0.0
    cutoff = np.percentile(diversity[has_track], keep_percentile)
    mask = has_track & (diversity >= cutoff)
    return mask, diversity, cutoff


def height_band_mask(pts, up, density_frac=0.15, nbins=150):
    h = pts @ up
    lo, hi = np.percentile(h, [0.2, 99.8])
    hist, edges = np.histogram(h, bins=nbins, range=(lo, hi))
    thresh = hist.max() * density_frac
    above = hist >= thresh
    # find the largest contiguous run of bins above threshold
    best_start, best_len, cur_start, cur_len = 0, 0, 0, 0
    for i, ok in enumerate(above):
        if ok:
            if cur_len == 0:
                cur_start = i
            cur_len += 1
            if cur_len > best_len:
                best_start, best_len = cur_start, cur_len
        else:
            cur_len = 0
    band_lo = edges[best_start]
    band_hi = edges[best_start + best_len]
    return (h >= band_lo) & (h <= band_hi), band_lo, band_hi


def detect_shelf_heights(pts, up, u0, min_gap_frac=0.06, smooth_win=5, back_frac=0.5, prominence_frac=0.25):
    """Find candidate shelf heights from the back-of-shelf edge, not the front.

    The front face of a shelf is covered in products (SIFT-rich but noisy,
    no clean gaps between shelf levels). The *back* edge, where the shelf
    bracket clips into the vertical pegboard/support, is a harder, more
    consistent physical feature -- often less occluded since products don't
    always fill the full depth of a shelf. Restrict to points near the
    depth-axis midpoint (where the shared back panel sits between the two
    opposite-facing shelf rows) before building the height histogram.

    This is an empirical heuristic, not guaranteed -- validate the printed
    peaks against a couple of reference photos before trusting the count.
    """
    depth = pts @ u0
    center = np.median(depth)
    span = np.percentile(depth, 90) - np.percentile(depth, 10)
    near_back = np.abs(depth - center) < (span * back_frac * 0.5)

    h = pts[near_back] @ up
    if len(h) < 20:
        return [], None

    lo, hi = np.percentile(h, [0.5, 99.5])
    nbins = 150
    hist, edges = np.histogram(h, bins=nbins, range=(lo, hi))
    centers = (edges[:-1] + edges[1:]) / 2

    # simple moving-average smoothing
    kernel = np.ones(smooth_win) / smooth_win
    smoothed = np.convolve(hist, kernel, mode="same")

    min_gap_bins = max(int(nbins * min_gap_frac), 1)
    prominence = smoothed.max() * prominence_frac

    peaks = []
    for i in range(1, len(smoothed) - 1):
        if smoothed[i] > smoothed[i - 1] and smoothed[i] >= smoothed[i + 1] and smoothed[i] >= prominence:
            if not peaks or (i - peaks[-1]) >= min_gap_bins:
                peaks.append(i)
            elif smoothed[i] > smoothed[peaks[-1]]:
                peaks[-1] = i  # keep the taller of two close peaks

    peak_heights = [float(centers[i]) for i in peaks]
    return peak_heights, (lo, hi)


def write_ply(path: Path, pts, cols):
    with open(path, "w") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(pts)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")
        for p, c in zip(pts, cols):
            f.write(f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f} {int(c[0])} {int(c[1])} {int(c[2])}\n")


def write_png(path: Path, img: np.ndarray):
    h, w, _ = img.shape

    def chunk(tag, data):
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

    sig = bytes([137, 80, 78, 71, 13, 10, 26, 10])
    ihdr = struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)
    raw = b"".join(b"\x00" + img[y].tobytes() for y in range(h))
    idat = zlib.compress(raw, 9)
    with open(path, "wb") as f:
        f.write(sig)
        f.write(chunk(b"IHDR", ihdr))
        f.write(chunk(b"IDAT", idat))
        f.write(chunk(b"IEND", b""))


def rasterize(a, b, cols, res=900):
    amin, amax = np.percentile(a, 0.3), np.percentile(a, 99.7)
    bmin, bmax = np.percentile(b, 0.3), np.percentile(b, 99.7)
    scale = res / max(amax - amin, 1e-6)
    W = max(int((amax - amin) * scale) + 1, 10)
    H = max(int((bmax - bmin) * scale) + 1, 10)
    img = np.zeros((H, W, 3), dtype=np.uint8)
    xi = ((a - amin) * scale).astype(int)
    yi = (H - 1 - (b - bmin) * scale).astype(int)
    valid = (xi >= 0) & (xi < W) & (yi >= 0) & (yi < H)
    img[yi[valid], xi[valid]] = cols[valid]
    return img


def main():
    ap = argparse.ArgumentParser(description="Isolate the photographed subject from a COLMAP sparse reconstruction.")
    ap.add_argument("--colmap-txt", required=True, help="Path to COLMAP TXT export dir (has images.txt, points3D.txt)")
    ap.add_argument("--output", required=True, help="Output PLY path for the isolated point cloud")
    ap.add_argument("--preview-prefix", default=None, help="If set, writes <prefix>_before.png / _after.png top-view images")
    ap.add_argument("--proximity-percentile", type=float, default=70.0,
                     help="Keep points at/below this percentile of mean distance-to-observing-cameras")
    ap.add_argument("--min-track-len", type=int, default=3, help="Minimum observations per point to trust it")
    ap.add_argument("--diversity-percentile", type=float, default=40.0,
                     help="Keep points at/above this percentile of viewpoint-diversity (relative, self-calibrating)")
    ap.add_argument("--height-density-frac", type=float, default=0.15, help="Height-band density threshold (fraction of peak)")
    ap.add_argument("--detect-shelves", action="store_true",
                     help="Estimate per-shelf heights from the back-of-shelf/pegboard edge (empirical -- verify against photos)")
    ap.add_argument("--shelf-back-frac", type=float, default=0.5,
                     help="Fraction of the depth span, centered on the midpoint, treated as 'near the back panel'")
    ap.add_argument("--shelf-prominence", type=float, default=0.25,
                     help="Minimum peak height as a fraction of the tallest peak (lower = more shelves detected)")
    ap.add_argument("--shelf-min-gap", type=float, default=0.06,
                     help="Minimum spacing between detected shelves, as a fraction of the height range")
    args = ap.parse_args()

    colmap_dir = Path(args.colmap_txt)
    images = parse_images_txt(colmap_dir / "images.txt")
    pts, cols, tracks = parse_points3d_txt(colmap_dir / "points3D.txt")
    print(f"Loaded {len(images)} camera poses, {len(pts)} 3D points")

    cam_centers, up, alignment = camera_centers_and_up(images)
    print(f"Up-vector alignment across cameras: {alignment:.3f} (1.0 = perfect)")

    u0, v0 = build_basis(up)
    cam_centers_arr = np.array(list(cam_centers.values()))
    centroid = cam_centers_arr.mean(axis=0)

    prox_mask, prox_dist, prox_cutoff = proximity_mask(pts, tracks, cam_centers, args.proximity_percentile)
    print(f"Camera-proximity filter (percentile={args.proximity_percentile}, cutoff={prox_cutoff:.3f}) -> kept {prox_mask.sum()}/{len(pts)}")

    div_mask, diversity, div_cutoff = viewpoint_diversity_mask(pts, tracks, cam_centers, args.min_track_len, args.diversity_percentile)
    print(f"Viewpoint-diversity filter (percentile={args.diversity_percentile}, cutoff={div_cutoff:.3f}) -> kept {div_mask.sum()}/{len(pts)}")

    combined_mask_pre_height = prox_mask & div_mask
    if not combined_mask_pre_height.any():
        raise SystemExit(
            "No points survived the proximity + viewpoint-diversity filters. "
            "Try --proximity-percentile higher and/or --diversity-percentile lower (e.g. 10-20)."
        )
    height_mask, band_lo, band_hi = height_band_mask(pts[combined_mask_pre_height], up, args.height_density_frac)
    print(f"Height band: [{band_lo:.3f}, {band_hi:.3f}]  span={band_hi - band_lo:.3f}")

    final_pts = pts[combined_mask_pre_height][height_mask]
    final_cols = cols[combined_mask_pre_height][height_mask]
    print(f"Final isolated point count: {len(final_pts)} / {len(pts)} ({100 * len(final_pts) / len(pts):.1f}%)")

    rel = final_pts - centroid
    H = rel @ up
    U = rel @ u0
    V = rel @ v0
    print()
    print("Isolated subject bounding box (raw COLMAP units, NOT meters):")
    print(f"  height (up)   span: {H.max() - H.min():.3f}")
    print(f"  width  (v0)   span: {V.max() - V.min():.3f}")
    print(f"  depth  (u0)   span: {U.max() - U.min():.3f}")
    print()
    print("Anchor one of these to a real-world measurement (e.g. --height in meters)")
    print("to get a scale factor, same as generate_parametric_gondola.py expects.")

    write_ply(Path(args.output), final_pts, final_cols)
    print(f"Wrote {args.output}")

    if args.detect_shelves:
        peaks, hrange = detect_shelf_heights(
            final_pts - centroid, up, u0,
            back_frac=args.shelf_back_frac, prominence_frac=args.shelf_prominence,
            min_gap_frac=args.shelf_min_gap,
        )
        print()
        if not peaks:
            print("Shelf detection: not enough near-back-panel points to find peaks.")
        else:
            print(f"Shelf detection (near-back-panel height peaks, raw units, range={hrange}):")
            for h in sorted(peaks):
                print(f"  {h:.3f}")
            print(f"  -> {len(peaks)} candidate shelf(s). This is an empirical heuristic -- "
                  f"cross-check against 1-2 reference photos before trusting the count.")

    if args.preview_prefix:
        rel_all = pts - centroid
        Va, Ha = rel_all @ v0, rel_all @ up
        before = rasterize(Va, Ha, cols)
        write_png(Path(f"{args.preview_prefix}_before.png"), before)

        after = rasterize(V, H, final_cols)
        write_png(Path(f"{args.preview_prefix}_after.png"), after)
        print(f"Wrote {args.preview_prefix}_before.png / _after.png for visual comparison")


if __name__ == "__main__":
    main()
