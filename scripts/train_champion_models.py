from __future__ import annotations
import pandas as pd
from pathlib import Path
from src.training.ranking_model import RankingModelConfig, train_ranking_model
from src.ranking.losses import LISTWISE_LISTMLE, LISTWISE_LAMBDARANK, POINTWISE_FOCAL, PAIRWISE_HINGE, LISTWISE_SOFTMAX

def main() -> None:
    df = pd.DataFrame([
        {"query_id": "q1", "label": 1.0, "base": 0.5, "text": 0.8, "popularity": 0.9, "multimodal": 0.4},
        {"query_id": "q1", "label": 0.0, "base": 0.2, "text": 0.3, "popularity": 0.1, "multimodal": 0.1},
        {"query_id": "q2", "label": 1.0, "base": 0.9, "text": 0.7, "popularity": 0.8, "multimodal": 0.9},
        {"query_id": "q2", "label": 0.0, "base": 0.1, "text": 0.2, "popularity": 0.1, "multimodal": 0.2}
    ])
    
    cfg = RankingModelConfig(
        epochs=10,
        learning_rate=0.05,
        candidate_objectives=(LISTWISE_LAMBDARANK, LISTWISE_LISTMLE, POINTWISE_FOCAL, PAIRWISE_HINGE, LISTWISE_SOFTMAX)
    )
    
    output_dir = Path("models/champion")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    result = train_ranking_model(training_frame=df, output_dir=output_dir, config=cfg)
    
    print(f"Champion selected: {result.artifact.objective} with AUC {result.artifact.training_auc}")

if __name__ == "__main__":
    main()
