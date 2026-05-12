from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable

from src.retrieval.types import RetrievalCandidate, RetrievalFilter, RetrievalQuery, VectorItem


def _dot(left: list[float], right: list[float]) -> float:
    total = 0.0
    for value, other in zip(left, right):
        total += float(value) * float(other)
    return total


def _norm(vector: list[float]) -> float:
    return math.sqrt(_dot(vector, vector))


@dataclass(frozen=True)
class IndexStats:
    dimension: int
    count: int


class VectorIndex:
    def add(self, item: VectorItem) -> None:
        raise NotImplementedError

    def add_many(self, items: Iterable[VectorItem]) -> None:
        for item in items:
            self.add(item)

    def search(self, query: RetrievalQuery) -> list[RetrievalCandidate]:
        raise NotImplementedError

    def stats(self) -> IndexStats:
        raise NotImplementedError


class InMemoryVectorIndex(VectorIndex):
    def __init__(self, dimension: int | None = None, name: str = "in_memory_index") -> None:
        self._dimension = dimension
        self._name = name
        self._items: list[VectorItem] = []

    def add(self, item: VectorItem) -> None:
        if not item.vector:
            raise ValueError("empty vector")
        if self._dimension is None:
            self._dimension = len(item.vector)
        if len(item.vector) != self._dimension:
            raise ValueError("dimension mismatch")
        self._items.append(item)

    def search(self, query: RetrievalQuery) -> list[RetrievalCandidate]:
        if not query.embedding or not self._items:
            return []
        if self._dimension is not None and len(query.embedding) != self._dimension:
            raise ValueError("dimension mismatch")
        filters = query.filters or RetrievalFilter()
        query_norm = _norm(query.embedding)
        if query_norm == 0:
            return []
        scored: list[RetrievalCandidate] = []
        for item in self._items:
            if filters.allowed_item_ids is not None and item.item_id not in filters.allowed_item_ids:
                continue
            if filters.domain and item.domain != filters.domain:
                continue
            denominator = _norm(item.vector) * query_norm
            score = 0.0 if denominator == 0 else _dot(item.vector, query.embedding) / denominator
            scored.append(
                RetrievalCandidate(
                    item_id=item.item_id,
                    score=float(score),
                    source=self._name,
                    metadata=dict(item.metadata),
                )
            )
        scored.sort(key=lambda candidate: candidate.score, reverse=True)
        return scored[: query.top_k]

    def stats(self) -> IndexStats:
        return IndexStats(dimension=self._dimension or 0, count=len(self._items))


class FaissVectorIndex(VectorIndex):
    def __init__(self, dimension: int | None = None, name: str = "faiss_index", index_type: str = "Flat") -> None:
        self._dimension = dimension
        self._name = name
        self._index_type = index_type
        self._faiss_index = None
        self._id_map: list[str] = []
        self._metadata: dict[str, dict] = {}
        self._domains: dict[str, str | None] = {}
        self._id_to_int: dict[str, int] = {}

    def _ensure_index(self, dim: int) -> None:
        if self._faiss_index is not None:
            return
        import faiss
        self._dimension = dim
        if self._index_type == "Flat":
            self._faiss_index = faiss.IndexFlatIP(dim)
        elif self._index_type == "HNSW":
            self._faiss_index = faiss.IndexHNSWFlat(dim, 32)
        else:
            self._faiss_index = faiss.IndexFlatIP(dim)

    def add(self, item: VectorItem) -> None:
        if not item.vector:
            raise ValueError("empty vector")
        
        self._ensure_index(len(item.vector))
        
        if len(item.vector) != self._dimension:
            raise ValueError("dimension mismatch")

        import numpy as np
        vec = np.array(item.vector, dtype=np.float32)
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
            
        self._faiss_index.add(np.expand_dims(vec, axis=0))
        
        idx = len(self._id_map)
        self._id_map.append(item.item_id)
        self._id_to_int[item.item_id] = idx
        self._metadata[item.item_id] = dict(item.metadata)
        self._domains[item.item_id] = item.domain

    def search(self, query: RetrievalQuery) -> list[RetrievalCandidate]:
        if not query.embedding or self._faiss_index is None:
            return []
            
        if self._dimension is not None and len(query.embedding) != self._dimension:
            raise ValueError("dimension mismatch")

        import numpy as np
        q_vec = np.array(query.embedding, dtype=np.float32)
        q_norm = np.linalg.norm(q_vec)
        if q_norm == 0:
            return []
        q_vec = q_vec / q_norm
        
        filters = query.filters or RetrievalFilter()
        
        k = query.top_k
        if filters.allowed_item_ids is not None or filters.domain is not None:
            k = min(self._faiss_index.ntotal, k * 10)
            
        distances, indices = self._faiss_index.search(np.expand_dims(q_vec, axis=0), k)
        
        candidates: list[RetrievalCandidate] = []
        for i in range(len(indices[0])):
            idx = indices[0][i]
            if idx < 0 or idx >= len(self._id_map):
                continue
                
            item_id = self._id_map[idx]
            score = float(distances[0][i])
            
            if filters.allowed_item_ids is not None and item_id not in filters.allowed_item_ids:
                continue
            if filters.domain and self._domains.get(item_id) != filters.domain:
                continue
                
            candidates.append(
                RetrievalCandidate(
                    item_id=item_id,
                    score=score,
                    source=self._name,
                    metadata=dict(self._metadata.get(item_id, {})),
                )
            )
            
            if len(candidates) >= query.top_k:
                break
                
        return candidates

    def stats(self) -> IndexStats:
        if self._dimension is None or self._faiss_index is None:
            return IndexStats(dimension=0, count=0)
        return IndexStats(dimension=self._dimension, count=self._faiss_index.ntotal)


class CrossDomainAdapter:
    def __init__(self, source_domain: str, target_domain: str, adapter_matrix: list[list[float]] | None = None) -> None:
        self.source_domain = source_domain
        self.target_domain = target_domain
        self._matrix = adapter_matrix

    def adapt(self, query: RetrievalQuery) -> RetrievalQuery:
        if not self._matrix or not query.embedding:
            return query
            
        import numpy as np
        vec = np.array(query.embedding, dtype=np.float32)
        mat = np.array(self._matrix, dtype=np.float32)
        adapted_vec = vec @ mat
        
        return RetrievalQuery(
            embedding=adapted_vec.tolist(),
            top_k=query.top_k,
            filters=query.filters,
            user_id=query.user_id,
        )
