from __future__ import annotations

import time
import hashlib
from dataclasses import dataclass, field
from typing import Any

import requests

from src.domain.models import Candidate
from src.data.ingestion.base import stable_item_id


@dataclass(frozen=True)
class MusicFact:
    name: str
    entity_type: str
    country: str
    genres: list[str]
    born: str
    disambiguation: str
    aliases: list[str]
    truth_file: str


@dataclass(frozen=True)
class MusicSearchResult:
    facts: list[MusicFact]
    candidates: list[Candidate]
    truth_context: list[str]


USER_AGENT = "Hermes/1.0 (Universal Intelligence Platform)"
BASE_URL = "https://musicbrainz.org/ws/2"
MIN_REQUEST_INTERVAL = 1.1


class MusicService:
    def __init__(self) -> None:
        self._cache: dict[str, MusicSearchResult] = {}
        self._last_request_time: float = 0.0

    def _throttled_get(self, url: str) -> dict:
        elapsed = time.time() - self._last_request_time
        if elapsed < MIN_REQUEST_INTERVAL:
            time.sleep(MIN_REQUEST_INTERVAL - elapsed)

        headers = {"User-Agent": USER_AGENT, "Accept": "application/json"}
        response = requests.get(url, headers=headers, timeout=8)
        self._last_request_time = time.time()
        response.raise_for_status()
        return response.json()

    def _search_artists(self, query: str, limit: int = 5) -> list[dict]:
        url = f"{BASE_URL}/artist/?query={requests.utils.quote(query)}&fmt=json&limit={limit}"
        try:
            data = self._throttled_get(url)
            return data.get("artists", [])
        except Exception:
            return []

    def _search_release_groups(self, query: str, limit: int = 5) -> list[dict]:
        url = f"{BASE_URL}/release-group/?query={requests.utils.quote(query)}&fmt=json&limit={limit}"
        try:
            data = self._throttled_get(url)
            return data.get("release-groups", [])
        except Exception:
            return []

    def _extract_artist_fact(self, artist: dict) -> MusicFact:
        name = artist.get("name", "")
        entity_type = artist.get("type", "Unknown")
        country = artist.get("country", "Unknown")
        disambiguation = artist.get("disambiguation", "")

        tags = artist.get("tags", [])
        genres = []
        for tag in sorted(tags, key=lambda t: t.get("count", 0), reverse=True):
            tag_name = tag.get("name", "")
            if tag_name and tag.get("count", 0) >= 0:
                genres.append(tag_name)
        genres = genres[:5]

        life_span = artist.get("life-span", {})
        born = life_span.get("begin", "")

        aliases_raw = artist.get("aliases", [])
        aliases = [a.get("name", "") for a in aliases_raw if a.get("name")][:3]

        parts = [f"Artist: {name}"]
        if entity_type and entity_type != "Unknown":
            parts.append(f"Type: {entity_type}")
        if country and country != "Unknown":
            parts.append(f"Country: {country}")
        if genres:
            parts.append(f"Genres: {', '.join(genres)}")
        if born:
            parts.append(f"Born: {born}")
        if disambiguation:
            parts.append(f"About: {disambiguation}")

        truth_file = "Truth File: " + " | ".join(parts)

        return MusicFact(
            name=name,
            entity_type=entity_type,
            country=country,
            genres=genres,
            born=born,
            disambiguation=disambiguation,
            aliases=aliases,
            truth_file=truth_file,
        )

    def _fact_to_candidate(self, fact: MusicFact, rank: int) -> Candidate:
        combined = f"{fact.name} {' '.join(fact.genres)} {fact.disambiguation}"
        item_id = stable_item_id("music", fact.name.lower())
        score = max(0.1, 1.0 - (rank * 0.15))

        return Candidate(
            item_id=item_id,
            title=fact.name,
            score=score,
            channel="musicbrainz_api",
            metadata={
                "signals": {
                    "text": score * 0.7,
                    "multimodal": 0.0,
                    "popularity": score * 0.2,
                    "recency": 0.0,
                    "novelty": score * 0.1,
                },
                "domain": "music",
                "entity_type": fact.entity_type,
                "country": fact.country,
                "genres": fact.genres,
            },
        )

    def search_and_embed(self, query: str) -> MusicSearchResult:
        cache_key = query.lower().strip()
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached

        artists = self._search_artists(query)
        facts: list[MusicFact] = []
        candidates: list[Candidate] = []
        truth_context: list[str] = []

        for rank, artist in enumerate(artists):
            score = artist.get("score", 0)
            if score < 50:
                continue
            fact = self._extract_artist_fact(artist)
            facts.append(fact)
            truth_context.append(fact.truth_file)
            candidates.append(self._fact_to_candidate(fact, rank))

        if not facts:
            release_groups = self._search_release_groups(query)
            for rank, rg in enumerate(release_groups):
                score_val = rg.get("score", 0)
                if score_val < 50:
                    continue
                title = rg.get("title", "")
                primary_type = rg.get("primary-type", "Album")
                artist_credit = rg.get("artist-credit", [])
                artist_names = [ac.get("name", "") for ac in artist_credit if ac.get("name")]
                artist_str = ", ".join(artist_names) if artist_names else "Unknown"

                parts = [f"Artist: {title}", f"Type: {primary_type}", f"By: {artist_str}"]
                truth_str = "Truth File: " + " | ".join(parts)
                truth_context.append(truth_str)

                fact = MusicFact(
                    name=title,
                    entity_type=primary_type,
                    country="",
                    genres=[],
                    born="",
                    disambiguation="",
                    aliases=[],
                    truth_file=truth_str,
                )
                facts.append(fact)
                candidates.append(self._fact_to_candidate(fact, rank))

        result = MusicSearchResult(
            facts=facts,
            candidates=candidates,
            truth_context=truth_context,
        )
        self._cache[cache_key] = result
        return result
