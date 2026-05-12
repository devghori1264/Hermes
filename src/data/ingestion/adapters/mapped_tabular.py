from __future__ import annotations

from typing import Any

import pandas as pd

from src.data.ingestion.base import BaseIngestionAdapter, IngestionConfig, stable_item_id


def _read_delimiter(config: IngestionConfig) -> str:
    delimiter = config.extras.get("delimiter", ",")
    if not isinstance(delimiter, str) or not delimiter:
        return ","
    return delimiter


def _read_mapping(config: IngestionConfig) -> dict[str, Any]:
    mapping = config.extras.get("mapping", {})
    if not isinstance(mapping, dict):
        return {}
    return mapping


class MappedTabularAdapter(BaseIngestionAdapter):
    name = "mapped_tabular"

    def load_raw(self, source_path, config: IngestionConfig | None = None) -> pd.DataFrame:
        delimiter = ","
        if config is not None:
            delimiter = _read_delimiter(config)
        return pd.read_csv(source_path, sep=delimiter)

    def normalize(self, data: pd.DataFrame, config: IngestionConfig) -> list[dict[str, Any]]:
        mapping = _read_mapping(config)
        id_column = str(mapping.get("id_column", "")).strip()
        title_column = str(mapping.get("title_column", "")).strip()
        raw_text_columns = mapping.get("text_columns", [])
        if not isinstance(raw_text_columns, list):
            raw_text_columns = []
        text_columns = [str(value).strip() for value in raw_text_columns]
        text_columns = [value for value in text_columns if value]

        if not id_column or not title_column:
            raise ValueError("mapping requires id_column and title_column")

        items: list[dict[str, Any]] = []
        for index, row in data.iterrows():
            raw_id = str(row.get(id_column, "")).strip()
            title = str(row.get(title_column, "")).strip()
            if not raw_id or not title:
                continue
            combined_parts = [title]
            for column in text_columns:
                value = str(row.get(column, "")).strip()
                if value:
                    combined_parts.append(value)
            combined = " ".join(combined_parts).strip()

            item_id = stable_item_id(config.domain, raw_id)
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
                        "source_id": raw_id,
                    },
                }
            )
        return items
