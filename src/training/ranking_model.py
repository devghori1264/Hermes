from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Iterable
from src.model_registry.versioning import ModelVersionRegistry, VersionStage

import numpy as np
import pandas as pd

from src.domain.models import Candidate
from src.ranking.losses import (
    LISTWISE_SOFTMAX,
    LISTWISE_LISTMLE,
    LISTWISE_LAMBDARANK,
    PAIRWISE_HINGE,
    POINTWISE_LOGLOSS,
    POINTWISE_FOCAL,
    normalize_objective_name,
)


RANKING_SIGNAL_COLUMNS = ["base", "text", "multimodal", "popularity", "recency", "novelty"]


@dataclass(frozen=True)
class RankingModelConfig:
    epochs: int = 40
    learning_rate: float = 0.08
    l2: float = 1e-3
    pairwise_margin: float = 0.1
    listwise_temperature: float = 1.0
    objective: str = PAIRWISE_HINGE
    candidate_objectives: tuple[str, ...] = (PAIRWISE_HINGE, POINTWISE_LOGLOSS, LISTWISE_SOFTMAX, LISTWISE_LISTMLE, LISTWISE_LAMBDARANK)
    seed: int = 13


@dataclass(frozen=True)
class RankingModelArtifact:
    feature_names: list[str]
    weights: dict[str, float]
    bias: float
    feature_means: dict[str, float]
    feature_stds: dict[str, float]
    training_loss: float
    training_auc: float
    objective: str = PAIRWISE_HINGE


@dataclass(frozen=True)
class RankingTrainingResult:
    artifact: RankingModelArtifact
    output_dir: str


class LinearRankingModel:
    def __init__(
        self,
        feature_names: list[str],
        weights: dict[str, float],
        bias: float,
        feature_means: dict[str, float],
        feature_stds: dict[str, float],
        objective: str = PAIRWISE_HINGE,
    ) -> None:
        self.feature_names = feature_names
        self.weights = weights
        self.bias = bias
        self.feature_means = feature_means
        self.feature_stds = feature_stds
        self.objective = normalize_objective_name(objective)

    @classmethod
    def from_artifact(cls, artifact: RankingModelArtifact) -> "LinearRankingModel":
        return cls(
            feature_names=list(artifact.feature_names),
            weights=dict(artifact.weights),
            bias=float(artifact.bias),
            feature_means=dict(artifact.feature_means),
            feature_stds=dict(artifact.feature_stds),
            objective=artifact.objective,
        )

    def _scale(self, feature_name: str, value: float) -> float:
        mean = self.feature_means.get(feature_name, 0.0)
        std = self.feature_stds.get(feature_name, 1.0)
        if std <= 0.0:
            std = 1.0
        return (value - mean) / std

    def score_signals(self, signals: dict[str, float]) -> float:
        score = self.bias
        for name in self.feature_names:
            score += self.weights.get(name, 0.0) * self._scale(name, float(signals.get(name, 0.0)))
        return float(score)

    def score_candidate(self, candidate: Candidate) -> float:
        metadata = candidate.metadata or {}
        signals = metadata.get("signals", {}) if isinstance(metadata, dict) else {}
        if not isinstance(signals, dict):
            signals = {}
        return self.score_signals({name: float(signals.get(name, 0.0)) for name in self.feature_names})

    def rank(self, candidates: Iterable[Candidate]) -> list[Candidate]:
        ranked = list(candidates)
        ranked.sort(key=self.score_candidate, reverse=True)
        return ranked


def _extract_matrix(frame: pd.DataFrame, feature_names: list[str]) -> tuple[np.ndarray, np.ndarray]:
    if "label" not in frame.columns:
        raise ValueError("ranking training requires a label column")
    matrix = np.zeros((len(frame), len(feature_names)), dtype=np.float32)
    for col_index, feature_name in enumerate(feature_names):
        if feature_name in frame.columns:
            matrix[:, col_index] = frame[feature_name].fillna(0.0).astype(float).to_numpy()
    labels = frame["label"].fillna(0).astype(int).to_numpy()
    return matrix, labels


def _compute_stats(matrix: np.ndarray, feature_names: list[str]) -> tuple[dict[str, float], dict[str, float]]:
    means: dict[str, float] = {}
    stds: dict[str, float] = {}
    for index, feature_name in enumerate(feature_names):
        column = matrix[:, index]
        means[feature_name] = float(np.mean(column)) if len(column) else 0.0
        std = float(np.std(column)) if len(column) else 0.0
        stds[feature_name] = std if std > 0.0 else 1.0
    return means, stds


def _scale_matrix(matrix: np.ndarray, means: dict[str, float], stds: dict[str, float], feature_names: list[str]) -> np.ndarray:
    scaled = matrix.astype(np.float32).copy()
    for index, feature_name in enumerate(feature_names):
        std = stds.get(feature_name, 1.0)
        if std <= 0.0:
            std = 1.0
        scaled[:, index] = (scaled[:, index] - means.get(feature_name, 0.0)) / std
    return scaled


def _clamp_probability(value: float) -> float:
    return max(1e-12, min(1.0 - 1e-12, float(value)))


def _sigmoid(value: float) -> float:
    if value >= 0:
        z = np.exp(-value)
        return float(1.0 / (1.0 + z))
    z = np.exp(value)
    return float(z / (1.0 + z))


def _softmax(values: np.ndarray, temperature: float) -> np.ndarray:
    temp = temperature if temperature > 0.0 else 1.0
    scaled = values / temp
    shifted = scaled - float(np.max(scaled))
    exp_scores = np.exp(shifted)
    denom = float(np.sum(exp_scores))
    if denom <= 0.0:
        return np.full_like(exp_scores, 1.0 / len(exp_scores))
    return exp_scores / denom


def _pairwise_training_step(
    weights: np.ndarray,
    bias: float,
    pos_features: np.ndarray,
    neg_features: np.ndarray,
    learning_rate: float,
    l2: float,
    margin: float,
) -> tuple[np.ndarray, float, float]:
    delta = float(np.dot(weights, pos_features - neg_features) + bias)
    loss = max(0.0, margin - delta)
    if loss <= 0.0:
        return weights, bias, 0.0
    gradient = -(pos_features - neg_features)
    weights = weights - learning_rate * (gradient + l2 * weights)
    bias = bias + learning_rate
    return weights, bias, loss


def _prepare_query_groups(training_frame: pd.DataFrame) -> list[pd.DataFrame]:
    if "query_id" in training_frame.columns:
        grouped = training_frame.copy()
        grouped["_row_index"] = range(len(grouped))
        return [group for _, group in grouped.groupby("query_id", dropna=False)]
    return [training_frame.assign(_row_index=range(len(training_frame)))]


def _group_training_pairs(group: pd.DataFrame, scaled: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    positive_rows = group[group["label"].astype(int) > 0]
    negative_rows = group[group["label"].astype(int) <= 0]
    if positive_rows.empty or negative_rows.empty:
        return []

    pairs: list[tuple[np.ndarray, np.ndarray]] = []
    for _, pos_row in positive_rows.iterrows():
        pos_features = scaled[int(pos_row["_row_index"])]
        for _, neg_row in negative_rows.iterrows():
            neg_features = scaled[int(neg_row["_row_index"])]
            pairs.append((pos_features, neg_features))
    return pairs


def _fit_pairwise_model(
    scaled: np.ndarray,
    query_groups: list[pd.DataFrame],
    cfg: RankingModelConfig,
) -> tuple[np.ndarray, float, float]:
    rng = np.random.default_rng(cfg.seed)
    weights = rng.normal(0.0, 0.02, size=scaled.shape[1]).astype(np.float32)
    bias = 0.0
    final_loss = 0.0

    for _ in range(cfg.epochs):
        epoch_loss = 0.0
        epoch_pairs = 0
        for group in query_groups:
            for pos_features, neg_features in _group_training_pairs(group, scaled):
                weights, bias, loss = _pairwise_training_step(
                    weights=weights,
                    bias=bias,
                    pos_features=pos_features,
                    neg_features=neg_features,
                    learning_rate=cfg.learning_rate,
                    l2=cfg.l2,
                    margin=cfg.pairwise_margin,
                )
                epoch_loss += loss
                epoch_pairs += 1
        if epoch_pairs == 0:
            break
        final_loss = epoch_loss / epoch_pairs

    return weights, bias, final_loss


def _fit_pointwise_model(
    scaled: np.ndarray,
    labels: np.ndarray,
    cfg: RankingModelConfig,
) -> tuple[np.ndarray, float, float]:
    rng = np.random.default_rng(cfg.seed)
    weights = rng.normal(0.0, 0.02, size=scaled.shape[1]).astype(np.float32)
    bias = 0.0
    final_loss = 0.0

    if len(scaled) == 0:
        return weights, bias, final_loss

    for _ in range(cfg.epochs):
        epoch_loss = 0.0
        for row_index in range(len(scaled)):
            features = scaled[row_index]
            label = float(labels[row_index])
            score = float(np.dot(weights, features) + bias)
            probability = _sigmoid(score)
            probability = _clamp_probability(probability)
            epoch_loss += -label * np.log(probability) - (1.0 - label) * np.log(1.0 - probability)
            gradient_error = probability - label
            weights = weights - cfg.learning_rate * (gradient_error * features + cfg.l2 * weights)
            bias = bias - cfg.learning_rate * gradient_error
        final_loss = epoch_loss / len(scaled)

    return weights, bias, float(final_loss)


def _fit_listwise_model(
    scaled: np.ndarray,
    query_groups: list[pd.DataFrame],
    cfg: RankingModelConfig,
) -> tuple[np.ndarray, float, float]:
    rng = np.random.default_rng(cfg.seed)
    weights = rng.normal(0.0, 0.02, size=scaled.shape[1]).astype(np.float32)
    bias = 0.0
    final_loss = 0.0

    for _ in range(cfg.epochs):
        epoch_loss = 0.0
        epoch_groups = 0
        for group in query_groups:
            group_rows = group["_row_index"].astype(int).to_list()
            if not group_rows:
                continue
            raw_labels = group["label"].astype(float).clip(lower=0.0).to_numpy()
            total_label_weight = float(np.sum(raw_labels))
            if total_label_weight <= 0.0:
                continue

            targets = raw_labels / total_label_weight
            group_matrix = scaled[group_rows]
            scores = group_matrix @ weights + bias
            probabilities = _softmax(scores, cfg.listwise_temperature)
            safe_probs = np.clip(probabilities, 1e-12, 1.0)
            epoch_loss += float(-np.sum(targets * np.log(safe_probs)))

            grad_scores = probabilities - targets
            grad_weights = (group_matrix.T @ grad_scores) / len(group_rows)
            grad_weights = grad_weights + cfg.l2 * weights
            grad_bias = float(np.mean(grad_scores))
            weights = weights - cfg.learning_rate * grad_weights
            bias = bias - cfg.learning_rate * grad_bias
            epoch_groups += 1

        if epoch_groups == 0:
            break
        final_loss = epoch_loss / epoch_groups

    return weights, bias, float(final_loss)

def _fit_listmle_model(
    scaled: np.ndarray,
    query_groups: list[pd.DataFrame],
    cfg: RankingModelConfig,
) -> tuple[np.ndarray, float, float]:
    rng = np.random.default_rng(cfg.seed)
    weights = rng.normal(0.0, 0.02, size=scaled.shape[1]).astype(np.float32)
    bias = 0.0
    final_loss = 0.0

    for _ in range(cfg.epochs):
        epoch_loss = 0.0
        epoch_groups = 0
        for group in query_groups:
            group_rows = group["_row_index"].astype(int).to_list()
            if not group_rows or len(group_rows) < 2:
                continue
            
            raw_labels = group["label"].astype(float).to_numpy()
            sort_indices = np.argsort(raw_labels)[::-1]
            sorted_rows = [group_rows[i] for i in sort_indices]
            
            group_matrix = scaled[sorted_rows]
            scores = group_matrix @ weights + bias
            
            group_loss = 0.0
            grad_scores = np.zeros_like(scores)
            
            for i in range(len(scores)):
                denom_sum = np.sum(np.exp(scores[i:]))
                prob = np.exp(scores[i]) / max(denom_sum, 1e-12)
                group_loss += -np.log(max(prob, 1e-12))
                
                grad_scores[i] -= 1.0
                for j in range(i, len(scores)):
                    grad_scores[j] += np.exp(scores[j]) / max(denom_sum, 1e-12)
            
            epoch_loss += float(group_loss / len(scores))
            
            grad_weights = (group_matrix.T @ grad_scores) / len(scores)
            grad_weights = grad_weights + cfg.l2 * weights
            grad_bias = float(np.mean(grad_scores))
            
            weights = weights - cfg.learning_rate * grad_weights
            bias = bias - cfg.learning_rate * grad_bias
            epoch_groups += 1

        if epoch_groups == 0:
            break
        final_loss = epoch_loss / epoch_groups

    return weights, bias, float(final_loss)

def _fit_lambdarank_model(
    scaled: np.ndarray,
    query_groups: list[pd.DataFrame],
    cfg: RankingModelConfig,
) -> tuple[np.ndarray, float, float]:
    rng = np.random.default_rng(cfg.seed)
    weights = rng.normal(0.0, 0.02, size=scaled.shape[1]).astype(np.float32)
    bias = 0.0
    final_loss = 0.0

    for _ in range(cfg.epochs):
        epoch_loss = 0.0
        epoch_pairs = 0
        for group in query_groups:
            group_rows = group["_row_index"].astype(int).to_list()
            if not group_rows or len(group_rows) < 2:
                continue
            
            raw_labels = group["label"].astype(float).to_numpy()
            group_matrix = scaled[group_rows]
            scores = group_matrix @ weights + bias
            
            sort_indices = np.argsort(scores)[::-1]
            ranks = {original_idx: rank for rank, original_idx in enumerate(sort_indices)}
            
            idcg = 0.0
            ideal_labels = sorted(raw_labels, reverse=True)
            for i in range(min(10, len(ideal_labels))):
                idcg += (2**ideal_labels[i] - 1) / np.log2(i + 2)
            if idcg == 0.0:
                idcg = 1.0

            grad_scores = np.zeros_like(scores)
            for i in range(len(scores)):
                for j in range(len(scores)):
                    if raw_labels[i] > raw_labels[j]:
                        delta_ndcg = abs((2**raw_labels[i] - 1) / np.log2(ranks[i] + 2) - (2**raw_labels[j] - 1) / np.log2(ranks[j] + 2)) / idcg
                        s_ij = scores[i] - scores[j]
                        loss_ij = np.log(1.0 + np.exp(-s_ij))
                        epoch_loss += delta_ndcg * loss_ij
                        epoch_pairs += 1
                        
                        lambda_ij = delta_ndcg * (-1.0 / (1.0 + np.exp(s_ij)))
                        grad_scores[i] += lambda_ij
                        grad_scores[j] -= lambda_ij
            
            grad_weights = (group_matrix.T @ grad_scores) / len(scores)
            grad_weights = grad_weights + cfg.l2 * weights
            grad_bias = float(np.mean(grad_scores))
            
            weights = weights - cfg.learning_rate * grad_weights
            bias = bias - cfg.learning_rate * grad_bias

        if epoch_pairs == 0:
            break
        final_loss = epoch_loss / epoch_pairs

    return weights, bias, float(final_loss)


def _compute_auc(scores: list[tuple[float, float]]) -> float:
    positives = [score for label, score in scores if label > 0]
    negatives = [score for label, score in scores if label <= 0]
    if not positives or not negatives:
        return 0.5
    comparisons = 0
    wins = 0.0
    for pos_score in positives:
        for neg_score in negatives:
            comparisons += 1
            if pos_score > neg_score:
                wins += 1.0
            elif pos_score == neg_score:
                wins += 0.5
    return wins / comparisons if comparisons else 0.5


def train_ranking_model(
    training_frame: pd.DataFrame,
    output_dir: Path,
    config: RankingModelConfig | None = None,
    model_registry: ModelVersionRegistry | None = None,
) -> RankingTrainingResult:
    cfg = config or RankingModelConfig()
    objective = normalize_objective_name(cfg.objective)

    feature_names = [name for name in RANKING_SIGNAL_COLUMNS if name in training_frame.columns]
    if not feature_names:
        raise ValueError("no ranking signal columns found")

    matrix, labels = _extract_matrix(training_frame, feature_names)
    feature_means, feature_stds = _compute_stats(matrix, feature_names)
    scaled = _scale_matrix(matrix, feature_means, feature_stds, feature_names)
    query_groups = _prepare_query_groups(training_frame)

    best_auc = -1.0
    best_artifact = None

    for candidate_objective in cfg.candidate_objectives:
        if candidate_objective == PAIRWISE_HINGE:
            weights, bias, total_loss = _fit_pairwise_model(scaled, query_groups, cfg)
        elif candidate_objective in [POINTWISE_LOGLOSS, POINTWISE_FOCAL]:
            weights, bias, total_loss = _fit_pointwise_model(scaled, labels, cfg)
        elif candidate_objective == LISTWISE_SOFTMAX:
            weights, bias, total_loss = _fit_listwise_model(scaled, query_groups, cfg)
        elif candidate_objective == LISTWISE_LISTMLE:
            weights, bias, total_loss = _fit_listmle_model(scaled, query_groups, cfg)
        elif candidate_objective == LISTWISE_LAMBDARANK:
            weights, bias, total_loss = _fit_lambdarank_model(scaled, query_groups, cfg)
        else:
            continue

        model = LinearRankingModel(
            feature_names=feature_names,
            weights={name: float(weights[index]) for index, name in enumerate(feature_names)},
            bias=float(bias),
            feature_means=feature_means,
            feature_stds=feature_stds,
            objective=candidate_objective,
        )

        ordered_scores = []
        for _, row in training_frame.iterrows():
            signals = {name: float(row.get(name, 0.0)) for name in feature_names}
            ordered_scores.append((float(row.get("label", 0.0)), model.score_signals(signals)))

        auc = _compute_auc(ordered_scores)
        
        if auc > best_auc:
            best_auc = auc
            best_artifact = RankingModelArtifact(
                feature_names=feature_names,
                weights=model.weights,
                bias=model.bias,
                feature_means=model.feature_means,
                feature_stds=model.feature_stds,
                training_loss=float(total_loss),
                training_auc=float(auc),
                objective=candidate_objective,
            )

    if best_artifact is None:
        raise ValueError("champion selection failed, no objective produced a valid model")

    output_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = output_dir / "ranking_model.json"
    artifact_path.write_text(json.dumps(best_artifact.__dict__, ensure_ascii=True, indent=2), encoding="utf8")
    
    if model_registry is not None:
        record = model_registry.register(artifact_path, "ranking", metadata={"objective": best_artifact.objective, "auc": best_artifact.training_auc})
        if best_artifact.training_auc > 0.5:
            model_registry.promote(record.version_id, VersionStage.PRODUCTION)
            
    return RankingTrainingResult(artifact=best_artifact, output_dir=str(output_dir))


def load_ranking_model(path: Path) -> LinearRankingModel:
    payload = json.loads(path.read_text(encoding="utf8"))
    artifact = RankingModelArtifact(
        feature_names=list(payload["feature_names"]),
        weights={str(key): float(value) for key, value in payload["weights"].items()},
        bias=float(payload["bias"]),
        feature_means={str(key): float(value) for key, value in payload["feature_means"].items()},
        feature_stds={str(key): float(value) for key, value in payload["feature_stds"].items()},
        training_loss=float(payload["training_loss"]),
        training_auc=float(payload["training_auc"]),
        objective=normalize_objective_name(str(payload.get("objective", PAIRWISE_HINGE))),
    )
    return LinearRankingModel.from_artifact(artifact)
