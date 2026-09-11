"""
Download high-quality product images via Bing Image Search.

Bing embeds direct image URLs (from retailer CDNs like Walmart, Target, Kroger)
in its search results HTML — no API key required. Results are typically
professional white-background product shots at 400–1200px.

Saves to: knowledge_base/<product_name>/front/<slug>_web.jpg

Usage:
  python3 scrape_product_images.py \
      --knowledge-base knowledge_base \
      [--force]          # re-download even if front image already exists
      [--delay 1.2]      # seconds between requests
      [--dry-run]
"""
from __future__ import annotations

import argparse
import io
import json
import re
import time
import urllib.parse
import urllib.request
from pathlib import Path

from PIL import Image as PILImage


HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
    "Accept": "text/html,application/xhtml+xml,*/*",
}

# Prefer images from these domains (clean product photography)
PREFERRED_DOMAINS = [
    "walmartimages.com",
    "target.scene7.com",
    "kroger.com",
    "instacartassets.com",
    "wholesaleclub.ca",
    "costco.com",
    "safeway.com",
    "albertsons.com",
    "amazon.com",
    "images-amazon.com",
]


def slugify(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def bing_image_search(query: str) -> list[dict]:
    """Return list of {murl, turl, width, height} from Bing image search."""
    q = urllib.parse.quote(f"{query} product packaging")
    url = f"https://www.bing.com/images/search?q={q}&form=HDRSC2&first=1"
    req = urllib.request.Request(url, headers=HEADERS)
    try:
        with urllib.request.urlopen(req, timeout=12) as r:
            html = r.read().decode("utf-8", errors="replace")
    except Exception as e:
        print(f"  [bing error] {e}")
        return []

    results = []
    for raw in re.findall(r'm="({[^"]+})"', html):
        try:
            d = json.loads(raw.replace("&quot;", '"'))
            murl = d.get("murl", "")
            if murl and murl.startswith("http"):
                results.append(d)
        except Exception:
            pass
    return results


def score_url(url: str) -> int:
    """Higher = better. Prefer known retailer CDNs."""
    url_lower = url.lower()
    for i, domain in enumerate(PREFERRED_DOMAINS):
        if domain in url_lower:
            return len(PREFERRED_DOMAINS) - i
    return 0


def pick_best(results: list[dict]) -> str | None:
    """Pick best image URL: prefer retailer CDNs, then largest image."""
    if not results:
        return None
    scored = sorted(results, key=lambda d: score_url(d.get("murl", "")), reverse=True)
    return scored[0].get("murl") or None


def download_and_save(url: str, dest: Path, min_px: int = 200) -> bool:
    """Download image, verify minimum size, save as JPEG."""
    req = urllib.request.Request(url, headers=HEADERS)
    try:
        with urllib.request.urlopen(req, timeout=15) as r:
            raw = r.read()
    except Exception as e:
        print(f"  [download error] {e}")
        return False

    try:
        img = PILImage.open(io.BytesIO(raw)).convert("RGB")
        w, h = img.size
        if min(w, h) < min_px:
            print(f"  [skip] image too small: {w}×{h}")
            return False
        dest.parent.mkdir(parents=True, exist_ok=True)
        img.save(dest, format="JPEG", quality=92, optimize=True)
        print(f"  saved {w}×{h} → {dest.name}")
        return True
    except Exception as e:
        print(f"  [image error] {e}")
        return False


def process_product(
    name: str,
    front_dir: Path,
    delay: float,
    force: bool,
    dry_run: bool,
) -> str:
    existing = list(front_dir.glob("*.jpg")) + list(front_dir.glob("*.png"))
    if existing and not force:
        # Check if best existing image is web-sourced (large) or just a crop
        best = max(existing, key=lambda p: p.stat().st_size)
        size_kb = best.stat().st_size // 1024
        if size_kb > 30:   # >30 KB → probably already a good image
            return f"already_have ({best.name}, {size_kb}KB)"

    print(f"\n[{name}]")
    results = bing_image_search(name)
    time.sleep(delay)

    if not results:
        return "no_results"

    best_url = pick_best(results)
    if not best_url:
        return "no_url"

    domain = urllib.parse.urlparse(best_url).netloc
    print(f"  → {domain}  {best_url[:90]}")

    if dry_run:
        return f"would_download ({domain})"

    dest = front_dir / f"{slugify(name)}_web.jpg"
    ok = download_and_save(best_url, dest)
    return "downloaded" if ok else "download_failed"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--knowledge-base", required=True)
    parser.add_argument("--force", action="store_true",
                        help="Re-download even if front image exists")
    parser.add_argument("--delay", type=float, default=1.2)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    kb = Path(args.knowledge_base)
    product_dirs = sorted(
        d for d in kb.iterdir()
        if d.is_dir() and not d.name.startswith(("open_world", "_"))
    )
    print(f"Processing {len(product_dirs)} products  (force={args.force})\n")

    results: dict[str, str] = {}
    for prod_dir in product_dirs:
        name = prod_dir.name
        front_dir = prod_dir / "front"
        front_dir.mkdir(parents=True, exist_ok=True)
        status = process_product(name, front_dir, args.delay, args.force, args.dry_run)
        if "already_have" not in status:
            print(f"  status: {status}")
        results[name] = status

    print("\n── Summary ──")
    from collections import Counter
    for status, n in Counter(results.values()).most_common():
        # Collapse already_have variants for cleaner output
        key = "already_have" if status.startswith("already_have") else status
        print(f"  {key:35s} {n}")

    (kb / "scrape_results.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
