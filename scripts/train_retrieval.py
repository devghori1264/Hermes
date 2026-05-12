from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.training.retrieval import (
    build_candidate_retrieval_index,
    evaluate_retrieval_recall_at_k,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build and evaluate candidate retrieval artifacts")
    parser.add_argument("--embedding-dir", required=True, help="Directory containing trained embedding artifacts")
    parser.add_argument("--index-output", required=True, help="Output path for serialized retrieval index")
    parser.add_argument("--interactions", required=False, help="Optional interactions csv for recall evaluation")
    parser.add_argument("--recall-k", type=int, default=20)
    parser.add_argument("--domain", default="movies")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = build_candidate_retrieval_index(
        embedding_dir=Path(args.embedding_dir),
        output_path=Path(args.index_output),
        domain=args.domain,
    )
    print(f"item_count={result.item_count} dimension={result.dimension} index_path={result.index_path}")

    if args.interactions:
        interactions = pd.read_csv(args.interactions)
        recall = evaluate_retrieval_recall_at_k(
            embedding_dir=Path(args.embedding_dir),
            interactions=interactions,
            k=args.recall_k,
            domain=args.domain,
        )
        print(f"recall_at_{args.recall_k}={recall:.6f}")


if __name__ == "__main__":
    main()
