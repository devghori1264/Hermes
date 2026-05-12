from pathlib import Path

import pandas as pd
import pytest

from src.domain.models import Candidate
from src.training.ranking_model import RankingModelConfig, load_ranking_model, train_ranking_model


def _training_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"query_id": "q1", "label": 1, "base": 1.0, "text": 0.8, "multimodal": 0.4, "popularity": 0.3, "recency": 0.1, "novelty": 0.2},
            {"query_id": "q1", "label": 0, "base": 0.2, "text": 0.1, "multimodal": 0.0, "popularity": 0.0, "recency": 0.0, "novelty": 0.1},
            {"query_id": "q2", "label": 1, "base": 0.9, "text": 0.7, "multimodal": 0.5, "popularity": 0.4, "recency": 0.2, "novelty": 0.2},
            {"query_id": "q2", "label": 0, "base": 0.1, "text": 0.2, "multimodal": 0.1, "popularity": 0.0, "recency": 0.0, "novelty": 0.0},
        ]
    )


@pytest.mark.parametrize(
    "objective",
    ["pairwise_hinge", "pointwise_logloss", "listwise_softmax"],
)
def test_ranker_training_writes_model(tmp_path: Path, objective: str) -> None:
    result = train_ranking_model(
        _training_frame(),
        tmp_path,
        config=RankingModelConfig(epochs=16, learning_rate=0.08, seed=21, objective=objective),
    )

    model_path = tmp_path / "ranking_model.json"
    assert model_path.exists()
    assert result.artifact.training_auc >= 0.5
    assert result.artifact.objective == objective

    model = load_ranking_model(model_path)
    assert model.objective == objective
    positive = Candidate(item_id="p", title="Positive", score=0.0, channel="ranker", metadata={"signals": {"base": 1.0, "text": 0.8, "multimodal": 0.4, "popularity": 0.3, "recency": 0.1, "novelty": 0.2}})
    negative = Candidate(item_id="n", title="Negative", score=0.0, channel="ranker", metadata={"signals": {"base": 0.1, "text": 0.1, "multimodal": 0.0, "popularity": 0.0, "recency": 0.0, "novelty": 0.0}})

    ranked = model.rank([negative, positive])
    assert ranked[0].item_id == "p"


def test_ranker_training_normalizes_objective_alias(tmp_path: Path) -> None:
    result = train_ranking_model(
        _training_frame(),
        tmp_path,
        config=RankingModelConfig(epochs=8, learning_rate=0.08, seed=21, objective="pairwise"),
    )
    assert result.artifact.objective == "pairwise_hinge"


def test_ranker_training_rejects_unknown_objective(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unsupported ranking objective"):
        train_ranking_model(
            _training_frame(),
            tmp_path,
            config=RankingModelConfig(epochs=8, learning_rate=0.08, seed=21, objective="unknown"),
        )
