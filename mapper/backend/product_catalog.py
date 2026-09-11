from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

SUPPORTED_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}
DEFAULT_KB_DIR = Path(__file__).resolve().parent / "knowledge_base"
DEFAULT_CATALOG_PATH = DEFAULT_KB_DIR / "catalog.json"


@dataclass(frozen=True)
class CatalogProduct:
    name: str
    aliases: tuple[str, ...]
    reference_dir: Path
    reference_images: tuple[Path, ...]

    @property
    def reference_count(self) -> int:
        return len(self.reference_images)


class ProductCatalog:
    """Structured view of the product knowledge base and optional catalog manifest."""

    def __init__(
        self,
        knowledge_base_dir: Path | None = None,
        catalog_path: Path | None = None,
    ) -> None:
        self.knowledge_base_dir = (knowledge_base_dir or DEFAULT_KB_DIR)
        self.catalog_path = catalog_path or (self.knowledge_base_dir / "catalog.json")
        self.knowledge_base_dir.mkdir(parents=True, exist_ok=True)
        self._products = self._load_products()

    def products(self) -> list[CatalogProduct]:
        return list(self._products.values())

    def product(self, product_name: str) -> CatalogProduct | None:
        return self._products.get(product_name)

    def product_names(self) -> list[str]:
        return sorted(self._products)

    def aliases_for(self, product_name: str) -> tuple[str, ...]:
        product = self.product(product_name)
        if product is None:
            return ()
        return product.aliases

    def reference_images_for(self, product_name: str) -> tuple[Path, ...]:
        product = self.product(product_name)
        if product is None:
            return ()
        return product.reference_images

    def products_with_references(self) -> list[CatalogProduct]:
        return [product for product in self._products.values() if product.reference_count > 0]

    def total_reference_count(self) -> int:
        return sum(product.reference_count for product in self._products.values())

    def _is_ignored_name(self, name: str) -> bool:
        return not name or name.startswith(".") or name.startswith("_")

    def _load_products(self) -> dict[str, CatalogProduct]:
        metadata = self._load_catalog_metadata()
        products: dict[str, CatalogProduct] = {}

        discovered_names = {
            path.name
            for path in self.knowledge_base_dir.iterdir()
            if path.is_dir() and not self._is_ignored_name(path.name)
        }
        all_names = sorted(
            discovered_names | {name for name in metadata if not self._is_ignored_name(name)}
        )

        for name in all_names:
            reference_dir = self.knowledge_base_dir / name
            manifest_aliases = metadata.get(name, ())
            aliases = tuple(dict.fromkeys([name, *manifest_aliases]))
            reference_images = self._discover_reference_images(reference_dir)
            products[name] = CatalogProduct(
                name=name,
                aliases=aliases,
                reference_dir=reference_dir,
                reference_images=reference_images,
            )

        return products

    def _load_catalog_metadata(self) -> dict[str, tuple[str, ...]]:
        if not self.catalog_path.exists():
            return {}
        try:
            payload = json.loads(self.catalog_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return {}

        raw_products = payload.get("products", payload)
        entries: dict[str, list[str]] = {}

        if isinstance(raw_products, dict):
            iterable = (
                {"name": name, **config}
                for name, config in raw_products.items()
                if isinstance(config, dict)
            )
        elif isinstance(raw_products, list):
            iterable = (item for item in raw_products if isinstance(item, dict))
        else:
            iterable = ()

        for item in iterable:
            name = str(item.get("name") or "").strip()
            if not name:
                continue
            aliases = [str(alias).strip() for alias in item.get("aliases", []) if str(alias).strip()]
            entries[name] = aliases

        return {name: tuple(aliases) for name, aliases in entries.items()}

    def _discover_reference_images(self, reference_dir: Path) -> tuple[Path, ...]:
        if not reference_dir.exists():
            return ()
        images = sorted(
            path
            for path in reference_dir.rglob("*")
            if (
                path.is_file()
                and path.suffix.lower() in SUPPORTED_IMAGE_SUFFIXES
                and not any(self._is_ignored_name(part) for part in path.relative_to(reference_dir).parts)
            )
        )
        return tuple(images)
