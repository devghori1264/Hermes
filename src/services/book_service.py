from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import requests

from src.domain.models import Candidate
from src.data.ingestion.base import stable_item_id


@dataclass(frozen=True)
class BookFact:
    title: str
    authors: list[str]
    first_publish_year: int | None
    subjects: list[str]
    cover_id: int | None
    key: str
    truth_file: str


@dataclass(frozen=True)
class BookSearchResult:
    facts: list[BookFact]
    candidates: list[Candidate]
    truth_context: list[str]


USER_AGENT = "Hermes/1.0 (Universal Intelligence Platform)"
BASE_URL = "https://openlibrary.org"
MIN_REQUEST_INTERVAL = 1.1


class BookService:
    def __init__(self) -> None:
        self._cache: dict[str, BookSearchResult] = {}
        self._last_request_time: float = 0.0

    def _throttled_get(self, url: str) -> dict:
        elapsed = time.time() - self._last_request_time
        if elapsed < MIN_REQUEST_INTERVAL:
            time.sleep(MIN_REQUEST_INTERVAL - elapsed)

        headers = {"User-Agent": USER_AGENT}
        response = requests.get(url, headers=headers, timeout=8)
        self._last_request_time = time.time()
        response.raise_for_status()
        return response.json()

    def _search_books(self, query: str, limit: int = 10) -> list[dict]:
        encoded = requests.utils.quote(query)
        url = (
            f"{BASE_URL}/search.json"
            f"?q={encoded}"
            f"&fields=key,title,author_name,first_publish_year,subject,cover_i"
            f"&limit={limit}"
        )
        try:
            data = self._throttled_get(url)
            return data.get("docs", [])
        except Exception:
            return []

    def _extract_book_fact(self, doc: dict) -> BookFact:
        title = doc.get("title", "")
        authors = doc.get("author_name", [])
        first_publish_year = doc.get("first_publish_year")
        subjects_raw = doc.get("subject", [])
        subjects = [s for s in subjects_raw if s and not s.startswith("nyt:")][:8]
        cover_id = doc.get("cover_i")
        key = doc.get("key", "")

        parts = [f"Book: {title}"]
        if authors:
            parts.append(f"Author: {', '.join(authors[:3])}")
        if first_publish_year:
            parts.append(f"Published: {first_publish_year}")
        if subjects:
            parts.append(f"Subjects: {', '.join(subjects[:5])}")

        truth_file = "Truth File: " + " | ".join(parts)

        return BookFact(
            title=title,
            authors=authors[:3],
            first_publish_year=first_publish_year,
            subjects=subjects,
            cover_id=cover_id,
            key=key,
            truth_file=truth_file,
        )

    def _fact_to_candidate(self, fact: BookFact, rank: int) -> Candidate:
        item_id = stable_item_id("books", fact.title.lower())
        score = max(0.1, 1.0 - (rank * 0.1))
        author_str = ", ".join(fact.authors) if fact.authors else "Unknown"

        return Candidate(
            item_id=item_id,
            title=fact.title,
            score=score,
            channel="openlibrary_api",
            metadata={
                "signals": {
                    "text": score * 0.6,
                    "multimodal": 0.0,
                    "popularity": score * 0.3,
                    "recency": 0.0,
                    "novelty": score * 0.1,
                },
                "domain": "books",
                "authors": fact.authors,
                "first_publish_year": fact.first_publish_year,
                "subjects": fact.subjects,
                "cover_url": f"https://covers.openlibrary.org/b/id/{fact.cover_id}-M.jpg" if fact.cover_id else None,
            },
        )

    def search_and_embed(self, query: str) -> BookSearchResult:
        cache_key = query.lower().strip()
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached

        docs = self._search_books(query)
        facts: list[BookFact] = []
        candidates: list[Candidate] = []
        truth_context: list[str] = []

        for rank, doc in enumerate(docs):
            title = doc.get("title", "")
            if not title:
                continue
            fact = self._extract_book_fact(doc)
            facts.append(fact)
            truth_context.append(fact.truth_file)
            candidates.append(self._fact_to_candidate(fact, rank))

        result = BookSearchResult(
            facts=facts,
            candidates=candidates,
            truth_context=truth_context,
        )
        self._cache[cache_key] = result
        return result
