from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import pandas as pd


class MusicCatalogRepository:
    def __init__(self, csv_path: Path) -> None:
        self.csv_path = csv_path

    @lru_cache(maxsize=1)
    def load_catalog(self) -> pd.DataFrame:
        data = pd.read_csv(self.csv_path)
        data["artist_name"] = data["artist_name"].astype(str).str.lower()
        data["comb"] = data["comb"].astype(str)
        return data

    def titles(self) -> list[str]:
        data = self.load_catalog()
        return [name.title() for name in data["artist_name"].tolist()]

    def search(self, query: str) -> pd.DataFrame:
        data = self.load_catalog()
        query_lower = query.lower().strip()
        exact = data.loc[data["artist_name"] == query_lower]
        if not exact.empty:
            return exact
        partial = data.loc[data["artist_name"].str.contains(query_lower, na=False)]
        return partial
