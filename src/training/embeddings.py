from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import random

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class EmbeddingTrainingConfig:
    embedding_dim: int = 64
    epochs: int = 8
    learning_rate: float = 0.03
    l2: float = 1e-4
    negative_samples: int = 3
    seed: int = 7


@dataclass(frozen=True)
class EmbeddingTrainingResult:
    user_count: int
    item_count: int
    embedding_dim: int
    final_loss: float
    output_dir: str


def _sigmoid(x: float) -> float:
    if x >= 0:
        z = np.exp(-x)
        return float(1.0 / (1.0 + z))
    z = np.exp(x)
    return float(z / (1.0 + z))


def _prepare_interactions(interactions: pd.DataFrame) -> tuple[list[tuple[str, str]], dict[str, set[str]]]:
    required = {"user_id", "item_id"}
    missing = required.difference(set(interactions.columns))
    if missing:
        raise ValueError(f"missing required columns: {sorted(missing)}")
    pairs = [(str(row["user_id"]), str(row["item_id"])) for _, row in interactions.iterrows()]
    user_history: dict[str, set[str]] = {}
    for user_id, item_id in pairs:
        user_history.setdefault(user_id, set()).add(item_id)
    return pairs, user_history


def train_user_item_embeddings(
    interactions: pd.DataFrame,
    output_dir: Path,
    config: EmbeddingTrainingConfig | None = None,
) -> EmbeddingTrainingResult:
    cfg = config or EmbeddingTrainingConfig()
    rng = random.Random(cfg.seed)
    np_rng = np.random.default_rng(cfg.seed)

    pairs, user_history = _prepare_interactions(interactions)
    user_ids = sorted(user_history.keys())
    item_ids = sorted({item_id for _, item_id in pairs})
    if not user_ids or not item_ids:
        raise ValueError("interactions are empty")

    user_to_idx = {user_id: index for index, user_id in enumerate(user_ids)}
    item_to_idx = {item_id: index for index, item_id in enumerate(item_ids)}

    user_matrix = (np_rng.standard_normal((len(user_ids), cfg.embedding_dim)).astype(np.float32) * 0.03)
    item_matrix = (np_rng.standard_normal((len(item_ids), cfg.embedding_dim)).astype(np.float32) * 0.03)

    all_items = list(item_ids)
    final_loss = 0.0

    for _ in range(cfg.epochs):
        rng.shuffle(pairs)
        epoch_loss = 0.0
        for user_id, pos_item_id in pairs:
            user_index = user_to_idx[user_id]
            pos_index = item_to_idx[pos_item_id]
            for _ in range(cfg.negative_samples):
                neg_item_id = rng.choice(all_items)
                if neg_item_id in user_history[user_id]:
                    continue
                neg_index = item_to_idx[neg_item_id]

                user_vec = user_matrix[user_index]
                pos_vec = item_matrix[pos_index]
                neg_vec = item_matrix[neg_index]
                x_uij = float(np.dot(user_vec, pos_vec - neg_vec))
                sigma = _sigmoid(x_uij)
                grad = 1.0 - sigma

                user_grad = grad * (pos_vec - neg_vec) - (cfg.l2 * user_vec)
                pos_grad = grad * user_vec - (cfg.l2 * pos_vec)
                neg_grad = (-grad * user_vec) - (cfg.l2 * neg_vec)

                user_matrix[user_index] = user_vec + (cfg.learning_rate * user_grad)
                item_matrix[pos_index] = pos_vec + (cfg.learning_rate * pos_grad)
                item_matrix[neg_index] = neg_vec + (cfg.learning_rate * neg_grad)
                epoch_loss += -np.log(max(sigma, 1e-8))

        final_loss = float(epoch_loss / max(len(pairs), 1))

    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "user_embeddings.npy", user_matrix)
    np.save(output_dir / "item_embeddings.npy", item_matrix)
    (output_dir / "user_index.json").write_text(json.dumps(user_to_idx, ensure_ascii=True), encoding="utf8")
    (output_dir / "item_index.json").write_text(json.dumps(item_to_idx, ensure_ascii=True), encoding="utf8")
    metadata = {
        "embedding_dim": cfg.embedding_dim,
        "epochs": cfg.epochs,
        "learning_rate": cfg.learning_rate,
        "l2": cfg.l2,
        "negative_samples": cfg.negative_samples,
        "final_loss": final_loss,
        "user_count": len(user_ids),
        "item_count": len(item_ids),
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=True), encoding="utf8")

    return EmbeddingTrainingResult(
        user_count=len(user_ids),
        item_count=len(item_ids),
        embedding_dim=cfg.embedding_dim,
        final_loss=final_loss,
        output_dir=str(output_dir),
    )
