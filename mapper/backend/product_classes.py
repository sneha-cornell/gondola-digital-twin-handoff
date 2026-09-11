"""Stable class registry for catalog products.

Every catalog product gets a permanent integer class id, persisted in
``knowledge_base/classes.json``. Ids are append-only: new catalog products get
the next free id and existing entries are never renumbered, so YOLO datasets,
recognition records, and downstream planogram/Blender consumers can rely on
them across runs. Names identified outside the catalog (open-world hits) are
reported with ``class_id: null`` so they can be reviewed and promoted into the
catalog explicitly.
"""

from __future__ import annotations

import json
from pathlib import Path

from product_catalog import DEFAULT_KB_DIR, ProductCatalog

REGISTRY_FILENAME = "classes.json"


class ClassRegistry:
    def __init__(
        self,
        catalog: ProductCatalog | None = None,
        registry_path: Path | None = None,
    ) -> None:
        self._catalog = catalog or ProductCatalog()
        self.registry_path = registry_path or (
            self._catalog.knowledge_base_dir / REGISTRY_FILENAME
        )
        self._id_by_name: dict[str, int] = {}
        self._name_by_id: dict[int, str] = {}
        self._load()
        self._sync_with_catalog()
        self._alias_index = self._build_alias_index()

    def classes(self) -> list[dict]:
        return [
            {"class_id": class_id, "class_name": name}
            for class_id, name in sorted(self._name_by_id.items())
        ]

    def class_count(self) -> int:
        return len(self._name_by_id)

    def resolve(self, name: str | None) -> tuple[int, str] | None:
        """Map a recognized product name (canonical or alias, any casing) to
        its (class_id, canonical_name). Returns None for non-catalog names."""
        if not name:
            return None
        class_id = self._id_by_name.get(name)
        if class_id is not None:
            return class_id, name
        canonical = self._alias_index.get(name.strip().lower())
        if canonical is None:
            return None
        return self._id_by_name[canonical], canonical

    def annotate(self, detection: dict) -> None:
        """Stamp class_id / class_name onto a detection record in place."""
        if not detection.get("product_identity_label"):
            detection["class_id"] = None
            detection["class_name"] = None
            return
        name = detection.get("recognized_product_name") or detection.get("display_label")
        resolved = self.resolve(name)
        if resolved is None:
            # Named, but outside the catalog (e.g. open-world hit).
            detection["class_id"] = None
            detection["class_name"] = name
            return
        class_id, canonical = resolved
        detection["class_id"] = class_id
        detection["class_name"] = canonical

    def yolo_names(self) -> dict[int, str]:
        """Class map in the shape Ultralytics dataset YAMLs expect."""
        return dict(sorted(self._name_by_id.items()))

    def _load(self) -> None:
        if not self.registry_path.exists():
            return
        try:
            payload = json.loads(self.registry_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return
        for entry in payload.get("classes", []):
            if not isinstance(entry, dict):
                continue
            name = str(entry.get("class_name") or "").strip()
            class_id = entry.get("class_id")
            if not name or not isinstance(class_id, int):
                continue
            self._id_by_name[name] = class_id
            self._name_by_id[class_id] = name

    def _sync_with_catalog(self) -> None:
        next_id = max(self._name_by_id, default=-1) + 1
        changed = False
        for name in self._catalog.product_names():
            if name in self._id_by_name:
                continue
            self._id_by_name[name] = next_id
            self._name_by_id[next_id] = name
            next_id += 1
            changed = True
        if changed or not self.registry_path.exists():
            self._save()

    def _save(self) -> None:
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)
        self.registry_path.write_text(
            json.dumps({"classes": self.classes()}, indent=2),
            encoding="utf-8",
        )

    def _build_alias_index(self) -> dict[str, str]:
        index: dict[str, str] = {}
        for name in self._id_by_name:
            index[name.strip().lower()] = name
            for alias in self._catalog.aliases_for(name):
                index.setdefault(alias.strip().lower(), name)
        return index


def build_class_records(detections: list[dict], registry: ClassRegistry) -> dict:
    """Aggregate enriched detections into per-class records.

    Returns catalog classes with detection stats, plus non-catalog names
    (class_id null) surfaced separately as candidates for catalog promotion.
    """
    records: dict[str, dict] = {}
    unidentified = 0

    for detection in detections:
        if not detection or not detection.get("product_identity_label"):
            unidentified += 1
            continue
        name = detection.get("recognized_product_name") or detection.get("display_label")
        resolved = registry.resolve(name)
        if resolved is None:
            class_id, canonical = None, str(name)
        else:
            class_id, canonical = resolved

        record = records.setdefault(
            canonical,
            {
                "class_id": class_id,
                "class_name": canonical,
                "in_catalog": resolved is not None,
                "detection_count": 0,
                "images": set(),
                "sources": {},
                "confidences": [],
                "detection_ids": [],
            },
        )
        record["detection_count"] += 1
        if detection.get("image_name"):
            record["images"].add(detection["image_name"])
        source = str(detection.get("product_identity_source") or "unknown")
        record["sources"][source] = record["sources"].get(source, 0) + 1
        confidence = detection.get("product_identity_confidence")
        if isinstance(confidence, (int, float)):
            record["confidences"].append(float(confidence))
        record["detection_ids"].append(detection.get("id"))

    finalized = []
    for record in records.values():
        confidences = record.pop("confidences")
        record["image_count"] = len(record.pop("images"))
        record["confidence"] = {
            "mean": round(sum(confidences) / len(confidences), 4) if confidences else None,
            "min": round(min(confidences), 4) if confidences else None,
            "max": round(max(confidences), 4) if confidences else None,
        }
        finalized.append(record)
    finalized.sort(key=lambda item: (-item["detection_count"], item["class_name"]))

    return {
        "classes": finalized,
        "class_count": len(finalized),
        "identified_detection_count": sum(r["detection_count"] for r in finalized),
        "unidentified_detection_count": unidentified,
        "registry_class_count": registry.class_count(),
    }
