from __future__ import annotations

from typing import Any

import pandas as pd

from src.data.ingestion.base import BaseIngestionAdapter, IngestionConfig, stable_item_id


class MoviesCatalogAdapter(BaseIngestionAdapter):
    name = "movies_catalog"

    def load_raw(self, source_path, config: IngestionConfig | None = None) -> pd.DataFrame:
        data = pd.read_csv(source_path)
        data["movie_title"] = data["movie_title"].astype(str)
        data["comb"] = data["comb"].astype(str)
        return data

    def normalize(self, data: pd.DataFrame, config: IngestionConfig) -> list[dict[str, Any]]:
        items: list[dict[str, Any]] = []
        for index, row in data.iterrows():
            title = str(row.get("movie_title", "")).strip()
            if not title:
                continue
            combined = str(row.get("comb", "")).strip()
            item_id = stable_item_id(config.domain, title.lower())
            items.append(
                {
                    "item_id": item_id,
                    "domain": config.domain,
                    "locale": config.locale_default,
                    "modalities": {
                        "text": {"title": title, "combined": combined},
                        "image": {},
                        "audio": {},
                        "video": {},
                        "sequence": {},
                    },
                    "assets": {"text": [], "image": [], "audio": [], "video": []},
                    "provenance": {
                        "dataset_id": config.dataset_id,
                        "row_index": int(index),
                    },
                }
            )
        return items
