from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from src.data.dataset_registry import DatasetRegistry, DatasetSnapshot

import pandas as pd


class CatalogRepository:
    def __init__(self, csv_path: Path, snapshot_path: Path | None = None) -> None:
        self.csv_path = csv_path
        self._dataset_registry = DatasetRegistry()
        self._dataset_snapshot: DatasetSnapshot | None = None
        if snapshot_path and snapshot_path.exists():
            self._dataset_snapshot = self._dataset_registry.register_from_json(snapshot_path)

    @lru_cache(maxsize=1)
    def load_catalog(self) -> pd.DataFrame:
        data = pd.read_csv(self.csv_path)
        data["movie_title"] = data["movie_title"].astype(str).str.lower()
        data["comb"] = data["comb"].astype(str)
        return data

    def titles(self) -> list[str]:
        data = self.load_catalog()
        return [title.capitalize() for title in data["movie_title"].tolist()]

    def dataset_snapshot(self) -> DatasetSnapshot | None:
        return self._dataset_snapshot
