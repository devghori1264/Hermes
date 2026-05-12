from __future__ import annotations

import mimetypes
from typing import Any

import pandas as pd

from src.data.ingestion.base import BaseIngestionAdapter, IngestionConfig, stable_item_id


def _coerce_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return [item for item in value if item is not None]
    return [value]


def _coerce_str(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and pd.isna(value):
        return ""
    text = str(value).strip()
    if text.lower() == "nan":
        return ""
    return text


def _normalize_text_list(value: Any) -> list[str]:
    items = []
    for item in _coerce_list(value):
        if item is None:
            continue
        text = str(item).strip()
        if text:
            items.append(text)
    return items


def _guess_mime_type(url: str) -> str:
    mime_type, _ = mimetypes.guess_type(url)
    return mime_type or "application/octet-stream"


def _extract_image_urls(images_value: Any) -> list[str]:
    urls: list[str] = []
    for entry in _coerce_list(images_value):
        if isinstance(entry, dict):
            for key in ("hi_res", "large", "thumb", "medium", "small"):
                value = entry.get(key)
                if value:
                    urls.append(str(value))
        else:
            urls.append(str(entry))
    return [url for url in urls if url]


class AmazonReviews2023MetaAdapter(BaseIngestionAdapter):
    name = "amazon_reviews_2023_meta"

    def load_raw(self, source_path, config: IngestionConfig | None = None) -> pd.DataFrame:
        data = pd.read_json(source_path, lines=True)
        return data

    def normalize(self, data: pd.DataFrame, config: IngestionConfig) -> list[dict[str, Any]]:
        items: list[dict[str, Any]] = []
        for index, row in data.iterrows():
            parent_asin = _coerce_str(row.get("parent_asin"))
            asin = _coerce_str(row.get("asin"))
            identifier = parent_asin or asin
            title = _coerce_str(row.get("title"))
            if not identifier or not title:
                continue

            features = _normalize_text_list(row.get("features"))
            description = _normalize_text_list(row.get("description"))
            categories = _normalize_text_list(row.get("categories"))
            details = row.get("details") or {}
            store = _coerce_str(row.get("store"))

            combined_parts = [title] + features + description + categories
            combined = " ".join(part for part in combined_parts if part).strip()

            image_urls = _extract_image_urls(row.get("images"))
            image_assets = []
            for url in image_urls:
                image_assets.append({"uri": url, "mime_type": _guess_mime_type(url)})

            item_id = stable_item_id(config.domain, identifier)
            items.append(
                {
                    "item_id": item_id,
                    "domain": config.domain,
                    "locale": config.locale_default,
                    "modalities": {
                        "text": {
                            "title": title,
                            "features": features,
                            "description": description,
                            "categories": categories,
                            "store": store,
                            "details": details,
                            "combined": combined,
                        },
                        "image": {},
                        "audio": {},
                        "video": {},
                        "sequence": {},
                    },
                    "assets": {
                        "text": [],
                        "image": image_assets,
                        "audio": [],
                        "video": [],
                    },
                    "provenance": {
                        "dataset_id": config.dataset_id,
                        "row_index": int(index),
                        "parent_asin": parent_asin,
                        "asin": asin,
                    },
                }
            )
        return items
