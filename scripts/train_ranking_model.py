from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.ranking.losses import supported_objectives
from src.training.ranking_model import RankingModelConfig, train_ranking_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a linear ranking model from labeled candidate data")
    parser.add_argument("--training-frame", required=True, help="CSV path with label and signal columns")
    parser.add_argument("--output-dir", required=True, help="Directory for ranking model artifacts")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--learning-rate", type=float, default=0.08)
    parser.add_argument("--l2", type=float, default=1e-3)
    parser.add_argument("--pairwise-margin", type=float, default=0.1)
    parser.add_argument("--listwise-temperature", type=float, default=1.0)
    parser.add_argument("--objective", default="pairwise_hinge", choices=supported_objectives())
    parser.add_argument("--seed", type=int, default=13)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frame = pd.read_csv(args.training_frame)
    result = train_ranking_model(
        frame,
        Path(args.output_dir),
        config=RankingModelConfig(
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            l2=args.l2,
            pairwise_margin=args.pairwise_margin,
            listwise_temperature=args.listwise_temperature,
            objective=args.objective,
            seed=args.seed,
        ),
    )
    print(
        f"loss={result.artifact.training_loss:.6f} auc={result.artifact.training_auc:.6f} "
        f"output_dir={result.output_dir}"
    )


if __name__ == "__main__":
    main()
