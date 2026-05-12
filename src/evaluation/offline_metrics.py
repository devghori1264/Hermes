from __future__ import annotations

import math
from typing import Iterable


def _discount(position: int) -> float:
    return 1.0 / math.log2(position + 2)


def ndcg_at_k(recommended: list[str], relevant: set[str], k: int) -> float:
    if not recommended or not relevant or k <= 0:
        return 0.0
    dcg = 0.0
    for index, item_id in enumerate(recommended[:k]):
        if item_id in relevant:
            dcg += _discount(index)
    ideal = 0.0
    for index in range(min(len(relevant), k)):
        ideal += _discount(index)
    return dcg / ideal if ideal > 0 else 0.0


def recall_at_k(recommended: list[str], relevant: set[str], k: int) -> float:
    if not recommended or not relevant or k <= 0:
        return 0.0
    hits = sum(1 for item_id in recommended[:k] if item_id in relevant)
    return hits / len(relevant)


def mrr_at_k(recommended: list[str], relevant: set[str], k: int) -> float:
    if not recommended or not relevant or k <= 0:
        return 0.0
    for index, item_id in enumerate(recommended[:k], start=1):
        if item_id in relevant:
            return 1.0 / index
    return 0.0


def map_at_k(recommended: list[str], relevant: set[str], k: int) -> float:
    if not recommended or not relevant or k <= 0:
        return 0.0
    hits = 0
    total = 0.0
    for index, item_id in enumerate(recommended[:k], start=1):
        if item_id in relevant:
            hits += 1
            total += hits / index
    return total / min(len(relevant), k)


def batch_metric(values: Iterable[float]) -> float:
    values_list = list(values)
    return sum(values_list) / len(values_list) if values_list else 0.0
