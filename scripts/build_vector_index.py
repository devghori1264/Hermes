from __future__ import annotations

import argparse
from pathlib import Path

from src.services.vector_index_service import VectorIndexService


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect a trained vector index artifact")
    parser.add_argument("--index-path", required=True, help="Path to the serialized vector index json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    service = VectorIndexService(Path(args.index_path))
    snapshot = service.snapshot()
    print(
        f"domain={snapshot.domain} dimension={snapshot.dimension} item_count={snapshot.item_count} "
        f"source_path={snapshot.source_path}"
    )


if __name__ == "__main__":
    main()
