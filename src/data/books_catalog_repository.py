from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import pandas as pd


class BooksCatalogRepository:
    def __init__(self, csv_path: Path) -> None:
        self.csv_path = csv_path

    @lru_cache(maxsize=1)
    def load_catalog(self) -> pd.DataFrame:
        data = pd.read_csv(self.csv_path)
        data["book_title"] = data["book_title"].astype(str).str.lower()
        data["comb"] = data["comb"].astype(str)
        return data

    def titles(self) -> list[str]:
        data = self.load_catalog()
        return [title.title() for title in data["book_title"].tolist()]

    def authors(self) -> list[str]:
        data = self.load_catalog()
        authors_set = set()
        for row in data["authors"].tolist():
            if pd.isna(row):
                continue
            for a in str(row).split("|"):
                authors_set.add(a.strip())
        return list(authors_set)

    def search(self, query: str) -> pd.DataFrame:
        data = self.load_catalog()
        query_lower = query.lower().strip()
        exact = data.loc[data["book_title"] == query_lower]
        if not exact.empty:
            return exact
        partial = data.loc[data["book_title"].str.contains(query_lower, na=False)]
        return partial
