from __future__ import annotations

from typing import Any

import pandas as pd

from src.data.ingestion.base import BaseIngestionAdapter, IngestionConfig, stable_item_id


class MusicCatalogAdapter(BaseIngestionAdapter):
    name = "music_catalog"

    def load_raw(self, source_path, config: IngestionConfig | None = None) -> pd.DataFrame:
        data = pd.read_csv(source_path)
        data["artist_name"] = data["artist_name"].astype(str)
        data["comb"] = data["comb"].astype(str)
        return data

    def normalize(self, data: pd.DataFrame, config: IngestionConfig) -> list[dict[str, Any]]:
        items: list[dict[str, Any]] = []
        for index, row in data.iterrows():
            artist_name = str(row.get("artist_name", "")).strip()
            if not artist_name:
                continue
            entity_type = str(row.get("entity_type", ""))
            country = str(row.get("country", ""))
            genres_raw = str(row.get("genres", ""))
            genres = [g.strip() for g in genres_raw.split("|") if g.strip()]
            born = str(row.get("born", ""))
            disambiguation = str(row.get("disambiguation", ""))
            aliases_raw = str(row.get("aliases", ""))
            aliases = [a.strip() for a in aliases_raw.split("|") if a.strip()]
            combined = str(row.get("comb", ""))

            item_id = stable_item_id(config.domain, artist_name.lower())
            items.append(
                {
                    "item_id": item_id,
                    "domain": config.domain,
                    "locale": config.locale_default,
                    "modalities": {
                        "text": {
                            "title": artist_name,
                            "entity_type": entity_type,
                            "country": country,
                            "genres": genres,
                            "born": born,
                            "disambiguation": disambiguation,
                            "aliases": aliases,
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
                        "source": "musicbrainz",
                    },
                }
            )
        return items
