from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

from jsonschema import ValidationError

from src.data.dataset_registry import DatasetSnapshot
from src.data.ingestion.base import (
    BaseIngestionAdapter,
    IngestionConfig,
    IngestionReport,
    compute_content_hash,
    compute_schema_hash,
)
from src.data.ingestion.manifest import IngestionManifest, load_manifest
from src.data.ingestion.adapters.amazon_reviews_2023_meta import AmazonReviews2023MetaAdapter
from src.data.ingestion.adapters.mapped_tabular import MappedTabularAdapter
from src.data.ingestion.adapters.movielens_movies import MovieLensMoviesAdapter
from src.data.ingestion.adapters.movies_catalog import MoviesCatalogAdapter
from src.data.ingestion.adapters.music_catalog import MusicCatalogAdapter
from src.data.ingestion.adapters.books_catalog import BooksCatalogAdapter
from src.domain.contracts import validate_item


@dataclass(frozen=True)
class IngestionResult:
    dataset_snapshot: DatasetSnapshot
    output_items_path: Path
    report: IngestionReport


class AdapterRegistry:
    def __init__(self) -> None:
        self._registry: dict[str, BaseIngestionAdapter] = {}

    def register(self, dataset_id: str, adapter: BaseIngestionAdapter) -> None:
        if dataset_id in self._registry:
            raise ValueError("adapter already registered")
        self._registry[dataset_id] = adapter

    def resolve(self, dataset_id: str) -> BaseIngestionAdapter:
        adapter = self._registry.get(dataset_id)
        if adapter is None:
            raise ValueError("adapter not found")
        return adapter


def build_default_registry() -> AdapterRegistry:
    registry = AdapterRegistry()
    registry.register("movies_catalog", MoviesCatalogAdapter())
    registry.register("movielens_movies", MovieLensMoviesAdapter())
    registry.register("amazon_reviews_2023_meta", AmazonReviews2023MetaAdapter())
    registry.register("mapped_tabular", MappedTabularAdapter())
    registry.register("music_catalog", MusicCatalogAdapter())
    registry.register("books_catalog", BooksCatalogAdapter())
    return registry


def run_ingestion(manifest_path: Path, registry: AdapterRegistry | None = None) -> IngestionResult:
    loaded = load_manifest(manifest_path)
    manifest = loaded.manifest
    extras = {
        key: value
        for key, value in loaded.raw.items()
        if key
        not in {
            "dataset_id",
            "version",
            "domain",
            "source_uri",
            "license_name",
            "snapshot_utc",
            "locale_default",
            "expected_columns",
            "output_items_path",
            "output_snapshot_path",
            "record_limit",
            "strict_validation",
        }
    }
    registry = registry or build_default_registry()
    adapter = registry.resolve(manifest.dataset_id)

    source_path = Path(manifest.source_uri)
    if not source_path.exists():
        raise FileNotFoundError("source_uri not found")

    config = IngestionConfig(
        dataset_id=manifest.dataset_id,
        domain=manifest.domain,
        locale_default=manifest.locale_default,
        expected_columns=manifest.expected_columns,
        record_limit=manifest.record_limit,
        strict_validation=manifest.strict_validation,
        extras=extras,
    )

    data = adapter.load_raw(source_path, config)
    missing = adapter.find_missing_columns(data, manifest.expected_columns)
    if missing:
        raise ValueError("missing columns in source")

    if manifest.record_limit is not None:
        data = data.head(manifest.record_limit)

    items = adapter.normalize(data, config)
    warnings: list[str] = []

    valid_items: list[dict[str, Any]] = []
    for item in items:
        try:
            validate_item(item)
            valid_items.append(item)
        except ValidationError:
            if config.strict_validation:
                raise
            warnings.append("invalid item skipped")

    content_hash = compute_content_hash(source_path)
    schema_hash = compute_schema_hash(list(data.columns))

    output_items_path = Path(manifest.output_items_path)
    output_items_path.parent.mkdir(parents=True, exist_ok=True)
    with output_items_path.open("w", encoding="utf8") as handle:
        for item in valid_items:
            handle.write(json.dumps(item, ensure_ascii=True))
            handle.write("\n")

    output_snapshot_path = Path(manifest.output_snapshot_path)
    output_snapshot_path.parent.mkdir(parents=True, exist_ok=True)
    snapshot = DatasetSnapshot(
        dataset_id=manifest.dataset_id,
        version=manifest.version,
        license_name=manifest.license_name,
        source_uri=manifest.source_uri,
        snapshot_utc=manifest.snapshot_utc,
        content_hash=content_hash,
        schema_hash=schema_hash,
        extra={
            "domain": manifest.domain,
            "record_limit": manifest.record_limit,
            "warnings": warnings,
        },
    )
    output_snapshot_path.write_text(
        json.dumps(
            {
                "dataset_id": snapshot.dataset_id,
                "version": snapshot.version,
                "license_name": snapshot.license_name,
                "source_uri": snapshot.source_uri,
                "snapshot_utc": snapshot.snapshot_utc,
                "content_hash": snapshot.content_hash,
                "schema_hash": snapshot.schema_hash,
                "extra": snapshot.extra,
            },
            ensure_ascii=True,
            indent=2,
        ),
        encoding="utf8",
    )

    if manifest.record_limit is not None:
        warnings.append("record_limit applied")

    report = IngestionReport(
        total_rows=int(data.shape[0]),
        normalized_rows=len(valid_items),
        warnings=warnings,
    )

    return IngestionResult(
        dataset_snapshot=snapshot,
        output_items_path=output_items_path,
        report=report,
    )
