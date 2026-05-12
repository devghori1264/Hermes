from __future__ import annotations

from dataclasses import dataclass
import math
from collections import Counter

import pandas as pd
import numpy as np

try:
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
except ModuleNotFoundError:  # pragma: no cover - exercised when sklearn is unavailable
    CountVectorizer = None
    cosine_similarity = None


@dataclass(frozen=True)
class TextFeatureArtifacts:
    matrix: object
    similarity: object


class TextFeatureService:
    def __init__(self) -> None:
        self._cache: dict[tuple[str, tuple[str, ...]], TextFeatureArtifacts] = {}

    def _fallback_similarity(self, texts: list[str]) -> tuple[object, object]:
        tokenized = [text.lower().split() for text in texts]
        vocabulary = sorted({token for doc in tokenized for token in doc})
        matrix = []
        for doc in tokenized:
            counts = Counter(doc)
            row = [float(counts.get(token, 0)) for token in vocabulary]
            matrix.append(row)
        if not matrix:
            return [], []
        count_matrix = np.asarray(matrix, dtype=np.float32)
        norms = np.linalg.norm(count_matrix, axis=1, keepdims=True)
        norms = np.where(np.isclose(norms, 0.0), 1.0, norms)
        normalized = count_matrix / norms
        similarity = normalized @ normalized.T
        return count_matrix, similarity.tolist()

    def build_similarity(self, data_csv_fingerprint: str, data: pd.DataFrame) -> TextFeatureArtifacts:
        texts = tuple(str(value) for value in data["comb"].fillna("").tolist())
        cache_key = (data_csv_fingerprint, texts)
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached

        if CountVectorizer is None or cosine_similarity is None:
            count_matrix, similarity = self._fallback_similarity(list(texts))
            artifacts = TextFeatureArtifacts(matrix=count_matrix, similarity=similarity)
            self._cache[cache_key] = artifacts
            return artifacts

        vectorizer = CountVectorizer()
        count_matrix = vectorizer.fit_transform(list(texts))
        similarity = cosine_similarity(count_matrix)
        artifacts = TextFeatureArtifacts(matrix=count_matrix, similarity=similarity)
        self._cache[cache_key] = artifacts
        return artifacts
