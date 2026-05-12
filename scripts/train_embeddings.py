from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.training.embeddings import EmbeddingTrainingConfig, train_user_item_embeddings


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train user and item embeddings from interactions")
    parser.add_argument("--interactions", required=True, help="CSV path with user_id and item_id columns")
    parser.add_argument("--output-dir", required=True, help="Directory to write embedding artifacts")
    parser.add_argument("--embedding-dim", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=0.03)
    parser.add_argument("--l2", type=float, default=1e-4)
    parser.add_argument("--negative-samples", type=int, default=3)
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    interactions = pd.read_csv(args.interactions)
    config = EmbeddingTrainingConfig(
        embedding_dim=args.embedding_dim,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        l2=args.l2,
        negative_samples=args.negative_samples,
        seed=args.seed,
    )
    result = train_user_item_embeddings(interactions, Path(args.output_dir), config=config)
    print(
        f"user_count={result.user_count} item_count={result.item_count} "
        f"dim={result.embedding_dim} final_loss={result.final_loss:.6f}"
    )


if __name__ == "__main__":
    main()
