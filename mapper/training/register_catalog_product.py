from __future__ import annotations

import argparse
import json
from pathlib import Path

DEFAULT_CATALOG_DIR = Path(__file__).resolve().parents[1] / "backend" / "knowledge_base"
DEFAULT_VIEW_DIRS = ("front", "angle_left", "angle_right", "detail")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Register a product in the knowledge-base catalog and scaffold reference folders."
    )
    parser.add_argument("--product", required=True, help="Canonical product name.")
    parser.add_argument(
        "--alias",
        action="append",
        default=[],
        help="Alias text that OCR may produce. Repeat for multiple aliases.",
    )
    parser.add_argument(
        "--catalog-dir",
        type=Path,
        default=DEFAULT_CATALOG_DIR,
        help=f"Knowledge-base directory (default: {DEFAULT_CATALOG_DIR})",
    )
    parser.add_argument(
        "--view",
        action="append",
        default=[],
        help="Reference subdirectory to create. Defaults to front/angle_left/angle_right/detail.",
    )
    args = parser.parse_args()

    catalog_dir = args.catalog_dir.resolve()
    catalog_dir.mkdir(parents=True, exist_ok=True)
    catalog_path = catalog_dir / "catalog.json"
    product_dir = catalog_dir / args.product

    payload = _load_catalog(catalog_path)
    products = payload.setdefault("products", [])
    entry = next((item for item in products if item.get("name") == args.product), None)
    if entry is None:
        entry = {"name": args.product, "aliases": []}
        products.append(entry)

    merged_aliases = dict.fromkeys(
        alias.strip()
        for alias in [*entry.get("aliases", []), *args.alias]
        if alias and alias.strip()
    )
    entry["aliases"] = list(merged_aliases)

    views = args.view or list(DEFAULT_VIEW_DIRS)
    for view in views:
        (product_dir / view).mkdir(parents=True, exist_ok=True)

    catalog_path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")

    print(f"Catalog entry updated: {catalog_path}")
    print(f"Reference folders ready under: {product_dir}")
    for view in views:
        print(f"- {product_dir / view}")


def _load_catalog(catalog_path: Path) -> dict:
    if not catalog_path.exists():
        return {"products": []}
    try:
        return json.loads(catalog_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {"products": []}


if __name__ == "__main__":
    main()
