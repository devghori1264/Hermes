from pathlib import Path

import pandas as pd

from src.training.embeddings import EmbeddingTrainingConfig, train_user_item_embeddings
from src.training.retrieval import (
    build_candidate_retrieval_index,
    evaluate_retrieval_recall_at_k,
)


def _sample_interactions() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"user_id": "u1", "item_id": "i1"},
            {"user_id": "u1", "item_id": "i2"},
            {"user_id": "u2", "item_id": "i2"},
            {"user_id": "u2", "item_id": "i3"},
            {"user_id": "u3", "item_id": "i1"},
            {"user_id": "u3", "item_id": "i3"},
        ]
    )


def test_embedding_training_writes_artifacts(tmp_path) -> None:
    interactions = _sample_interactions()
    output_dir = Path(tmp_path / "embeddings")
    result = train_user_item_embeddings(
        interactions,
        output_dir,
        config=EmbeddingTrainingConfig(embedding_dim=16, epochs=2, negative_samples=1, seed=11),
    )
    assert result.user_count == 3
    assert result.item_count == 3
    assert result.embedding_dim == 16
    assert (output_dir / "user_embeddings.npy").exists()
    assert (output_dir / "item_embeddings.npy").exists()
    assert (output_dir / "user_index.json").exists()
    assert (output_dir / "item_index.json").exists()


def test_retrieval_training_and_eval(tmp_path) -> None:
    interactions = _sample_interactions()
    output_dir = Path(tmp_path / "embeddings")
    train_user_item_embeddings(
        interactions,
        output_dir,
        config=EmbeddingTrainingConfig(embedding_dim=16, epochs=2, negative_samples=1, seed=3),
    )
    index_path = Path(tmp_path / "retrieval" / "index.json")
    retrieval = build_candidate_retrieval_index(output_dir, index_path)
    assert retrieval.item_count == 3
    assert index_path.exists()
    recall = evaluate_retrieval_recall_at_k(output_dir, interactions, k=2)
    assert 0.0 <= recall <= 1.0
