from __future__ import annotations

from typing import Any

import pandas as pd

from src.data.ingestion.base import BaseIngestionAdapter, IngestionConfig, stable_item_id


class BooksCatalogAdapter(BaseIngestionAdapter):
    name = "books_catalog"

    def load_raw(self, source_path, config: IngestionConfig | None = None) -> pd.DataFrame:
        data = pd.read_csv(source_path)
        data["book_title"] = data["book_title"].astype(str)
        data["comb"] = data["comb"].astype(str)
        return data

    def normalize(self, data: pd.DataFrame, config: IngestionConfig) -> list[dict[str, Any]]:
        items: list[dict[str, Any]] = []
        for index, row in data.iterrows():
            book_title = str(row.get("book_title", "")).strip()
            if not book_title:
                continue
            authors_raw = str(row.get("authors", ""))
            authors = [a.strip() for a in authors_raw.split("|") if a.strip()]
            year = str(row.get("first_publish_year", ""))
            subjects_raw = str(row.get("subjects", ""))
            subjects = [s.strip() for s in subjects_raw.split("|") if s.strip()]
            cover_id = str(row.get("cover_id", ""))
            key = str(row.get("key", ""))
            combined = str(row.get("comb", ""))

            item_id = stable_item_id(config.domain, book_title.lower())
            items.append(
                {
                    "item_id": item_id,
                    "domain": config.domain,
                    "locale": config.locale_default,
                    "modalities": {
                        "text": {
                            "title": book_title,
                            "authors": authors,
                            "first_publish_year": year,
                            "subjects": subjects,
                            "cover_id": cover_id,
                            "key": key,
                            "combined": combined,
                        },
                        "image": {},
                        "audio": {},
                        "video": {},
                        "sequence": {},
                    },
                    "assets": {"text": [], "image": [], "audio": [], "video": []},
                    "provenance": {
                        "dataset_id": config.dataset_id,
                        "row_index": int(index),
                        "source": "openlibrary",
                    },
                }
            )
        return items
