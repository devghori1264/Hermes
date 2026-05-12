from __future__ import annotations
from dataclasses import dataclass
import math
from typing import Iterable

@dataclass(frozen=True)
class LossResult:
    name: str
    value: float
    details: dict[str, float]

POINTWISE_LOGLOSS = "pointwise_logloss"
POINTWISE_FOCAL = "pointwise_focal"
PAIRWISE_HINGE = "pairwise_hinge"
LISTWISE_SOFTMAX = "listwise_softmax"
LISTWISE_LISTMLE = "listwise_listmle"
LISTWISE_LAMBDARANK = "listwise_lambdarank"

_OBJECTIVE_ALIASES = {
    "pointwise": POINTWISE_LOGLOSS,
    "focal": POINTWISE_FOCAL,
    "pairwise": PAIRWISE_HINGE,
    "listwise": LISTWISE_SOFTMAX,
    "listmle": LISTWISE_LISTMLE,
    "lambdarank": LISTWISE_LAMBDARANK,
}

def supported_objectives() -> tuple[str, ...]:
    return (POINTWISE_LOGLOSS, POINTWISE_FOCAL, PAIRWISE_HINGE, LISTWISE_SOFTMAX, LISTWISE_LISTMLE, LISTWISE_LAMBDARANK)

def normalize_objective_name(objective: str) -> str:
    normalized = objective.strip().lower()
    normalized = _OBJECTIVE_ALIASES.get(normalized, normalized)
    if normalized not in supported_objectives():
        raise ValueError(f"unsupported ranking objective: {objective}")
    return normalized

def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))

def pointwise_logloss(labels: Iterable[int], predictions: Iterable[float]) -> LossResult:
    label_list = [float(label) for label in labels]
    prediction_list = [float(prediction) for prediction in predictions]
    total = 0.0
    count = 0
    for label, prediction in zip(label_list, prediction_list):
        pred = _clamp(float(prediction), 1e-12, 1.0 - 1e-12)
        total += -float(label) * math.log(pred) - (1.0 - float(label)) * math.log(1.0 - pred)
        count += 1
    value = total / count if count else 0.0
    return LossResult(name=POINTWISE_LOGLOSS, value=value, details={"count": float(count)})

def pointwise_focal_loss(labels: Iterable[int], predictions: Iterable[float], gamma: float = 2.0, alpha: float = 0.25) -> LossResult:
    label_list = [float(label) for label in labels]
    prediction_list = [float(prediction) for prediction in predictions]
    total = 0.0
    count = 0
    for label, prediction in zip(label_list, prediction_list):
        pred = _clamp(float(prediction), 1e-12, 1.0 - 1e-12)
        if label > 0.0:
            total += -alpha * (1.0 - pred) ** gamma * math.log(pred)
        else:
            total += -(1.0 - alpha) * pred ** gamma * math.log(1.0 - pred)
        count += 1
    value = total / count if count else 0.0
    return LossResult(name=POINTWISE_FOCAL, value=value, details={"count": float(count), "gamma": gamma, "alpha": alpha})

def pairwise_hinge_loss(pairs: Iterable[tuple[float, float]], margin: float = 1.0) -> LossResult:
    total = 0.0
    count = 0
    for positive, negative in pairs:
        total += max(0.0, float(margin) - float(positive) + float(negative))
        count += 1
    value = total / count if count else 0.0
    return LossResult(name=PAIRWISE_HINGE, value=value, details={"count": float(count), "margin": float(margin)})

def listwise_softmax_loss(scores: list[float], labels: list[int]) -> LossResult:
    if not scores:
        return LossResult(name=LISTWISE_SOFTMAX, value=0.0, details={"count": 0.0})
    max_score = max(scores)
    exp_scores = [math.exp(value - max_score) for value in scores]
    denom = sum(exp_scores)
    probs = [value / denom for value in exp_scores]
    total = 0.0
    for prob, label in zip(probs, labels):
        total += -float(label) * math.log(_clamp(prob, 1e-12, 1.0))
    value = total / len(scores)
    return LossResult(name=LISTWISE_SOFTMAX, value=value, details={"count": float(len(scores))})

def listwise_listmle_loss(scores: list[float], labels: list[int]) -> LossResult:
    if not scores:
        return LossResult(name=LISTWISE_LISTMLE, value=0.0, details={"count": 0.0})
    paired = sorted(zip(scores, labels), key=lambda x: x[1], reverse=True)
    sorted_scores = [p[0] for p in paired]
    total_loss = 0.0
    n = len(sorted_scores)
    for i in range(n):
        current_score = sorted_scores[i]
        denom_sum = sum(math.exp(sorted_scores[j]) for j in range(i, n))
        prob = math.exp(current_score) / max(denom_sum, 1e-12)
        total_loss += -math.log(_clamp(prob, 1e-12, 1.0))
    value = total_loss / n
    return LossResult(name=LISTWISE_LISTMLE, value=value, details={"count": float(n)})

def dcg_at_k(labels: list[int], k: int) -> float:
    dcg = 0.0
    for i in range(min(k, len(labels))):
        dcg += (2**labels[i] - 1) / math.log2(i + 2)
    return dcg

def ndcg_at_k(labels: list[int], k: int) -> float:
    dcg = dcg_at_k(labels, k)
    ideal_labels = sorted(labels, reverse=True)
    idcg = dcg_at_k(ideal_labels, k)
    if idcg == 0.0:
        return 0.0
    return dcg / idcg

def lambdarank_loss(scores: list[float], labels: list[int], sigma: float = 1.0) -> LossResult:
    n = len(scores)
    if n < 2:
        return LossResult(name=LISTWISE_LAMBDARANK, value=0.0, details={"count": float(n)})
    paired = list(enumerate(zip(scores, labels)))
    paired.sort(key=lambda x: x[1][0], reverse=True)
    ranks = {original_idx: rank for rank, (original_idx, _) in enumerate(paired)}
    total_loss = 0.0
    pair_count = 0
    idcg = dcg_at_k(sorted(labels, reverse=True), n)
    if idcg == 0.0:
        return LossResult(name=LISTWISE_LAMBDARANK, value=0.0, details={"count": float(n)})
    for i in range(n):
        for j in range(n):
            if labels[i] > labels[j]:
                delta_ndcg = abs((2**labels[i] - 1) / math.log2(ranks[i] + 2) - (2**labels[j] - 1) / math.log2(ranks[j] + 2)) / idcg
                s_ij = scores[i] - scores[j]
                loss_ij = math.log(1.0 + math.exp(-sigma * s_ij))
                total_loss += delta_ndcg * loss_ij
                pair_count += 1
    value = total_loss / pair_count if pair_count else 0.0
    return LossResult(name=LISTWISE_LAMBDARANK, value=value, details={"count": float(pair_count)})
