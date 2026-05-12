from __future__ import annotations

from dataclasses import dataclass
import requests


@dataclass(frozen=True)
class TmdbConfig:
    api_key: str
    timeout_seconds: int


class TmdbService:
    def __init__(self, config: TmdbConfig) -> None:
        self.config = config

    def _get(self, url: str) -> dict:
        if not self.config.api_key:
            return {}
        response = requests.get(url, timeout=self.config.timeout_seconds)
        response.raise_for_status()
        return response.json()

    def search_movie(self, query: str) -> dict:
        url = f"https://api.themoviedb.org/3/search/movie?api_key={self.config.api_key}&query={query}"
        return self._get(url)

    def movie_details(self, movie_id: int) -> dict:
        url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key={self.config.api_key}"
        return self._get(url)

    def movie_credits(self, movie_id: int) -> dict:
        url = f"https://api.themoviedb.org/3/movie/{movie_id}/credits?api_key={self.config.api_key}"
        return self._get(url)

    def person_details(self, person_id: int) -> dict:
        url = f"https://api.themoviedb.org/3/person/{person_id}?api_key={self.config.api_key}"
        return self._get(url)
