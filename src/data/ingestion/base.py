from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
from pathlib import Path
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class IngestionConfig:
    dataset_id: str
    domain: str
    locale_default: str
    expected_columns: list[str]
    record_limit: int | None = None
    strict_validation: bool = True
    extras: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class IngestionReport:
    total_rows: int
    normalized_rows: int
    warnings: list[str]


class BaseIngestionAdapter:
    name: str

    def load_raw(self, source_path: Path, config: IngestionConfig | None = None) -> pd.DataFrame:
        raise NotImplementedError

    def normalize(self, data: pd.DataFrame, config: IngestionConfig) -> list[dict[str, Any]]:
        raise NotImplementedError

    def find_missing_columns(self, data: pd.DataFrame, expected: list[str]) -> list[str]:
        missing = [column for column in expected if column not in data.columns]
        return missing


def compute_content_hash(path: Path) -> str:
    payload = path.read_bytes()
    return hashlib.sha256(payload).hexdigest()


def compute_schema_hash(columns: list[str]) -> str:
    joined = "\n".join(columns)
    return hashlib.sha256(joined.encode("utf8")).hexdigest()


def stable_item_id(domain: str, title: str) -> str:
    payload = f"{domain}|{title}".encode("utf8")
    return hashlib.sha256(payload).hexdigest()
