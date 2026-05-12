from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.retrieval.index import InMemoryVectorIndex
from src.retrieval.types import RetrievalQuery, VectorItem


@dataclass(frozen=True)
class RetrievalTrainingResult:
    item_count: int
    dimension: int
    recall_at_k: float
    index_path: str


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf8"))


def build_candidate_retrieval_index(
    embedding_dir: Path,
    output_path: Path,
    domain: str = "movies",
) -> RetrievalTrainingResult:
    item_embeddings = np.load(embedding_dir / "item_embeddings.npy")
    item_to_idx = _load_json(embedding_dir / "item_index.json")
    index = InMemoryVectorIndex(dimension=int(item_embeddings.shape[1]), name="trained_item_index")

    for item_id, idx in item_to_idx.items():
        index.add(
            VectorItem(
                item_id=str(item_id),
                vector=item_embeddings[int(idx)].astype(float).tolist(),
                domain=domain,
                metadata={"trained_embedding": True},
            )
        )

    payload = {
        "domain": domain,
        "dimension": int(item_embeddings.shape[1]),
        "items": [
            {"item_id": item_id, "vector": item_embeddings[int(idx)].astype(float).tolist()}
            for item_id, idx in item_to_idx.items()
        ],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=True), encoding="utf8")
    return RetrievalTrainingResult(
        item_count=len(item_to_idx),
        dimension=int(item_embeddings.shape[1]),
        recall_at_k=0.0,
        index_path=str(output_path),
    )


def evaluate_retrieval_recall_at_k(
    embedding_dir: Path,
    interactions: pd.DataFrame,
    k: int = 20,
    domain: str = "movies",
) -> float:
    user_embeddings = np.load(embedding_dir / "user_embeddings.npy")
    item_embeddings = np.load(embedding_dir / "item_embeddings.npy")
    user_to_idx = _load_json(embedding_dir / "user_index.json")
    item_to_idx = _load_json(embedding_dir / "item_index.json")

    index = InMemoryVectorIndex(dimension=int(item_embeddings.shape[1]), name="trained_item_index")
    for item_id, idx in item_to_idx.items():
        index.add(VectorItem(item_id=str(item_id), vector=item_embeddings[int(idx)].astype(float).tolist(), domain=domain))

    hits = 0
    total = 0
    for _, row in interactions.iterrows():
        user_id = str(row["user_id"])
        item_id = str(row["item_id"])
        user_index = user_to_idx.get(user_id)
        if user_index is None:
            continue
        query_vector = user_embeddings[int(user_index)].astype(float).tolist()
        candidates = index.search(RetrievalQuery(embedding=query_vector, top_k=k))
        predicted = {candidate.item_id for candidate in candidates}
        if item_id in predicted:
            hits += 1
        total += 1
    if total == 0:
        return 0.0
    return float(hits / total)
