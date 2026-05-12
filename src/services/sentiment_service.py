from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import pickle
import requests
from bs4 import BeautifulSoup
import numpy as np


@dataclass(frozen=True)
class ReviewSentiment:
    review: str
    label: str


class SentimentService:
    def __init__(self, model_path: Path, vectorizer_path: Path, timeout_seconds: int = 8) -> None:
        self._clf = pickle.load(open(model_path, "rb"))
        self._vectorizer = pickle.load(open(vectorizer_path, "rb"))
        self._timeout_seconds = timeout_seconds

    def imdb_review_sentiment(self, imdb_id: str) -> list[ReviewSentiment]:
        url = f"https://www.imdb.com/title/{imdb_id}/reviews/?ref_=tt_ov_rt"
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0 Safari/537.36"
            )
        }
        response = requests.get(url, headers=headers, timeout=self._timeout_seconds)
        if response.status_code != 200:
            return []
        soup = BeautifulSoup(response.content, "lxml")
        nodes = soup.find_all("div", {"class": "ipc-html-content-inner-div"})
        output: list[ReviewSentiment] = []
        for node in nodes:
            if not node.string:
                continue
            text = node.string
            vector = self._vectorizer.transform(np.array([text]))
            pred = self._clf.predict(vector)
            output.append(ReviewSentiment(review=text, label="Good" if pred else "Bad"))
        return output
