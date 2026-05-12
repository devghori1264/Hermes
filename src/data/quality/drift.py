from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class DataProfile:
    row_count: int
    columns: list[str]
    missing_fraction: dict[str, float]
    numeric_summary: dict[str, dict[str, float]]
    unique_count: dict[str, int]


@dataclass(frozen=True)
class DriftThresholds:
    missing_fraction_delta: float = 0.05
    numeric_mean_delta: float = 0.25
    numeric_std_ratio: float = 0.5
    unique_count_delta: float = 0.3


@dataclass(frozen=True)
class DriftViolation:
    column: str
    metric: str
    baseline: float
    current: float
    threshold: float


@dataclass(frozen=True)
class DriftReport:
    is_drifted: bool
    violations: list[DriftViolation]

@dataclass(frozen=True)
class LeakageReport:
    has_leakage: bool
    leaked_columns: list[str]
    correlation_scores: dict[str, float]

class LeakageDetector:
    def __init__(self, target_column: str = "label", correlation_threshold: float = 0.95) -> None:
        self.target_column = target_column
        self.correlation_threshold = correlation_threshold

    def detect(self, frame: pd.DataFrame) -> LeakageReport:
        if self.target_column not in frame.columns:
            return LeakageReport(False, [], {})
        
        numeric_frame = frame.select_dtypes(include=["number"])
        if self.target_column not in numeric_frame.columns:
            return LeakageReport(False, [], {})

        correlations = numeric_frame.corr(method="pearson")[self.target_column].abs()
        
        leaked = []
        scores = {}
        for col, score in correlations.items():
            if col == self.target_column or pd.isna(score):
                continue
            scores[col] = float(score)
            if score >= self.correlation_threshold:
                leaked.append(col)
                
        try:
            from sklearn.ensemble import RandomForestClassifier
            X = numeric_frame.drop(columns=[self.target_column]).fillna(0)
            y = numeric_frame[self.target_column].fillna(0).astype(int)
            if not X.empty and len(y.unique()) > 1:
                clf = RandomForestClassifier(n_estimators=10, max_depth=3, random_state=42)
                clf.fit(X, y)
                importances = clf.feature_importances_
                for i, col in enumerate(X.columns):
                    imp = float(importances[i])
                    if imp > 0.8:
                        if col not in leaked:
                            leaked.append(col)
                        scores[f"{col}_importance"] = imp
        except ImportError:
            pass

        return LeakageReport(
            has_leakage=bool(leaked),
            leaked_columns=leaked,
            correlation_scores=scores
        )


def _numeric_stats(series: pd.Series, row_count: int) -> dict[str, float] | None:
    if not pd.api.types.is_numeric_dtype(series):
        return None
    if not row_count:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    return {
        "mean": float(series.mean()),
        "std": float(series.std(ddof=0)),
        "min": float(series.min()),
        "max": float(series.max()),
    }


def _profile_column(series: pd.Series, row_count: int) -> tuple[float, int, dict[str, float] | None]:
    missing = float(series.isna().mean()) if row_count else 0.0
    unique = int(series.nunique(dropna=True))
    numeric = _numeric_stats(series, row_count)
    return missing, unique, numeric


def profile_frame(frame: pd.DataFrame) -> DataProfile:
    row_count = int(frame.shape[0])
    columns = list(frame.columns)
    missing_fraction: dict[str, float] = {}
    numeric_summary: dict[str, dict[str, float]] = {}
    unique_count: dict[str, int] = {}

    for column in columns:
        missing, unique, numeric = _profile_column(frame[column], row_count)
        missing_fraction[column] = missing
        unique_count[column] = unique
        if numeric is not None:
            numeric_summary[column] = numeric

    return DataProfile(
        row_count=row_count,
        columns=columns,
        missing_fraction=missing_fraction,
        numeric_summary=numeric_summary,
        unique_count=unique_count,
    )


def _compare_missing(
    column: str,
    baseline: DataProfile,
    current: DataProfile,
    thresholds: DriftThresholds,
) -> list[DriftViolation]:
    base_missing = baseline.missing_fraction.get(column, 0.0)
    current_missing = current.missing_fraction.get(column, 0.0)
    missing_delta = abs(current_missing - base_missing)
    if missing_delta <= thresholds.missing_fraction_delta:
        return []
    return [
        DriftViolation(
            column=column,
            metric="missing_fraction",
            baseline=base_missing,
            current=current_missing,
            threshold=thresholds.missing_fraction_delta,
        )
    ]


def _compare_unique(
    column: str,
    baseline: DataProfile,
    current: DataProfile,
    thresholds: DriftThresholds,
) -> list[DriftViolation]:
    base_unique = baseline.unique_count.get(column, 0)
    current_unique = current.unique_count.get(column, 0)
    if not base_unique:
        return []
    unique_delta = abs(current_unique - base_unique) / float(base_unique)
    if unique_delta <= thresholds.unique_count_delta:
        return []
    return [
        DriftViolation(
            column=column,
            metric="unique_count_ratio",
            baseline=float(base_unique),
            current=float(current_unique),
            threshold=thresholds.unique_count_delta,
        )
    ]


def _compare_numeric(
    column: str,
    baseline: DataProfile,
    current: DataProfile,
    thresholds: DriftThresholds,
) -> list[DriftViolation]:
    if column not in baseline.numeric_summary or column not in current.numeric_summary:
        return []
    base_stats = baseline.numeric_summary[column]
    current_stats = current.numeric_summary[column]
    violations: list[DriftViolation] = []

    mean_delta = abs(current_stats["mean"] - base_stats["mean"])
    if mean_delta > thresholds.numeric_mean_delta:
        violations.append(
            DriftViolation(
                column=column,
                metric="mean_delta",
                baseline=base_stats["mean"],
                current=current_stats["mean"],
                threshold=thresholds.numeric_mean_delta,
            )
        )

    base_std = base_stats["std"]
    current_std = current_stats["std"]
    if base_std > 0.0:
        std_ratio = abs(current_std - base_std) / base_std
        if std_ratio > thresholds.numeric_std_ratio:
            violations.append(
                DriftViolation(
                    column=column,
                    metric="std_ratio",
                    baseline=base_std,
                    current=current_std,
                    threshold=thresholds.numeric_std_ratio,
                )
            )

    return violations


def compare_profiles(
    baseline: DataProfile, current: DataProfile, thresholds: DriftThresholds | None = None
) -> DriftReport:
    thresholds = thresholds or DriftThresholds()
    violations: list[DriftViolation] = []

    common_columns = set(baseline.columns) & set(current.columns)
    for column in sorted(common_columns):
        violations.extend(_compare_missing(column, baseline, current, thresholds))
        violations.extend(_compare_unique(column, baseline, current, thresholds))
        violations.extend(_compare_numeric(column, baseline, current, thresholds))

    return DriftReport(is_drifted=bool(violations), violations=violations)
