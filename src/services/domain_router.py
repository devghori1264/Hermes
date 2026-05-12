from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class DomainClassification:
    domain: str
    confidence: float
    method: str
    scores: dict[str, float]


SUPPORTED_DOMAINS = ("movies", "music", "books")

DOMAIN_ANCHOR_PHRASES: dict[str, list[str]] = {
    "movies": [
        "movie film cinema director actor actress screenplay box office",
        "blockbuster thriller drama comedy horror action adventure sci-fi",
        "oscar nomination premiere sequel franchise trailer scene plot",
        "hollywood bollywood cinematography cast crew rating imdb",
        "watch streaming release date runtime poster genre",
    ],
    "music": [
        "song track album artist singer rapper band musician genre",
        "concert tour live performance lyrics melody beat tempo",
        "hip hop rock pop jazz classical electronic metal country",
        "grammy award platinum record label studio recording",
        "playlist stream listen release single EP discography",
    ],
    "books": [
        "book novel author writer publisher publication edition pages",
        "fiction nonfiction literature literary genre bestseller",
        "chapter story narrative plot character setting theme",
        "paperback hardcover ebook kindle library isbn reading",
        "biography memoir poetry anthology series volume trilogy",
    ],
}

KEYWORD_WEIGHTS: dict[str, dict[str, float]] = {
    "movies": {
        "movie": 3.0, "film": 3.0, "cinema": 2.5, "director": 2.5,
        "actor": 2.0, "actress": 2.0, "cast": 1.5, "screenplay": 2.0,
        "blockbuster": 2.0, "oscar": 2.0, "imdb": 2.0, "trailer": 1.5,
        "sequel": 1.5, "franchise": 1.5, "directed": 2.5, "starring": 2.0,
        "watch": 1.0, "scene": 1.0, "plot": 1.0, "hollywood": 2.0,
    },
    "music": {
        "song": 3.0, "album": 3.0, "artist": 2.0, "singer": 3.0,
        "rapper": 3.0, "band": 3.0, "musician": 3.0, "concert": 2.5,
        "track": 2.0, "lyrics": 2.5, "melody": 2.0, "beat": 1.5,
        "genre": 1.0, "grammy": 2.5, "rap": 2.5, "rock": 1.5,
        "pop": 1.5, "jazz": 2.0, "hip hop": 2.5, "listen": 1.0,
        "playlist": 2.0, "tour": 1.5, "guitar": 2.0, "vocal": 2.0,
    },
    "books": {
        "book": 3.0, "novel": 3.0, "author": 2.5, "writer": 2.5,
        "wrote": 2.0, "written": 2.0, "publish": 2.0, "edition": 2.0,
        "read": 1.5, "fiction": 2.0, "nonfiction": 2.0, "literature": 2.5,
        "chapter": 2.0, "bestseller": 2.0, "paperback": 2.0, "hardcover": 2.0,
        "library": 1.5, "isbn": 2.5, "page": 1.0, "trilogy": 2.0,
        "memoir": 2.5, "biography": 2.5, "poetry": 2.5, "narrative": 1.5,
    },
}

CONFIDENCE_THRESHOLD = 0.15


class SemanticDomainRouter:
    def __init__(self, model_id: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
        self._model = None
        self._centroids: dict[str, Any] = {}
        self._model_id = model_id
        self._model_available = False
        self._initialized = False

    def _initialize(self) -> None:
        if self._initialized:
            return
        self._initialized = True
        try:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self._model_id)
            self._model_available = True
            self._build_centroids()
        except Exception:
            self._model_available = False

    def _build_centroids(self) -> None:
        import numpy as np
        for domain, phrases in DOMAIN_ANCHOR_PHRASES.items():
            embeddings = self._model.encode(phrases)
            centroid = np.mean(embeddings, axis=0)
            norm = np.linalg.norm(centroid)
            if norm > 0:
                centroid = centroid / norm
            self._centroids[domain] = centroid

    def _semantic_classify(self, query: str) -> DomainClassification:
        import numpy as np
        query_embedding = self._model.encode([query])[0]
        query_norm = np.linalg.norm(query_embedding)
        if query_norm > 0:
            query_embedding = query_embedding / query_norm

        scores: dict[str, float] = {}
        for domain, centroid in self._centroids.items():
            similarity = float(np.dot(query_embedding, centroid))
            scores[domain] = max(0.0, similarity)

        best_domain = max(scores, key=scores.get)
        best_score = scores[best_domain]

        sorted_scores = sorted(scores.values(), reverse=True)
        margin = sorted_scores[0] - sorted_scores[1] if len(sorted_scores) > 1 else sorted_scores[0]

        if margin < CONFIDENCE_THRESHOLD:
            return DomainClassification(
                domain="unknown",
                confidence=margin,
                method="semantic",
                scores=scores,
            )

        return DomainClassification(
            domain=best_domain,
            confidence=best_score,
            method="semantic",
            scores=scores,
        )

    def _keyword_classify(self, query: str) -> DomainClassification:
        query_lower = query.lower()
        tokens = set(re.findall(r"[a-z]+", query_lower))

        scores: dict[str, float] = {}
        for domain, weights in KEYWORD_WEIGHTS.items():
            total = 0.0
            for keyword, weight in weights.items():
                if " " in keyword:
                    if keyword in query_lower:
                        total += weight
                elif keyword in tokens:
                    total += weight
            scores[domain] = total

        total_score = sum(scores.values())
        if total_score > 0:
            for domain in scores:
                scores[domain] = scores[domain] / total_score

        best_domain = max(scores, key=scores.get)
        best_score = scores[best_domain]

        if best_score < 0.4 or total_score == 0:
            return DomainClassification(
                domain="unknown",
                confidence=best_score,
                method="keyword",
                scores=scores,
            )

        return DomainClassification(
            domain=best_domain,
            confidence=best_score,
            method="keyword",
            scores=scores,
        )

    def classify(self, query: str, override_domain: str | None = None) -> DomainClassification:
        if override_domain and override_domain in SUPPORTED_DOMAINS:
            return DomainClassification(
                domain=override_domain,
                confidence=1.0,
                method="user_override",
                scores={d: 1.0 if d == override_domain else 0.0 for d in SUPPORTED_DOMAINS},
            )

        self._initialize()

        if self._model_available:
            return self._semantic_classify(query)

        return self._keyword_classify(query)
