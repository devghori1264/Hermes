from __future__ import annotations

from typing import Any

import pandas as pd

from src.data.ingestion.base import BaseIngestionAdapter, IngestionConfig, stable_item_id


class MovieLensMoviesAdapter(BaseIngestionAdapter):
    name = "movielens_movies"

    def load_raw(self, source_path, config: IngestionConfig | None = None) -> pd.DataFrame:
        data = pd.read_csv(source_path)
        data["movieId"] = data["movieId"].astype(str)
        data["title"] = data["title"].astype(str)
        data["genres"] = data["genres"].astype(str)
        return data

    def normalize(self, data: pd.DataFrame, config: IngestionConfig) -> list[dict[str, Any]]:
        items: list[dict[str, Any]] = []
        for index, row in data.iterrows():
            movie_id = str(row.get("movieId", "")).strip()
            title = str(row.get("title", "")).strip()
            genres = str(row.get("genres", "")).strip()
            if not movie_id or not title:
                continue
            item_id = stable_item_id(config.domain, f"movielens:{movie_id}")
            combined = " ".join(part for part in [title, genres] if part).strip()
            items.append(
                {
                    "item_id": item_id,
                    "domain": config.domain,
                    "locale": config.locale_default,
                    "modalities": {
                        "text": {"title": title, "genres": genres, "combined": combined},
                        "image": {},
                        "audio": {},
                        "video": {},
                        "sequence": {},
                    },
                    "assets": {"text": [], "image": [], "audio": [], "video": []},
                    "provenance": {
                        "dataset_id": config.dataset_id,
                        "row_index": int(index),
                        "movie_id": movie_id,
                    },
                }
            )
        return items
