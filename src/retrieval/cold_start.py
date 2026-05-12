from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import math
import re
from typing import Any, Iterable

from src.domain.models import Candidate


_TOKEN_PATTERN = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> set[str]:
    return set(_TOKEN_PATTERN.findall(text.lower()))


def _jaccard(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    overlap = len(left & right)
    union = len(left | right)
    if union == 0:
        return 0.0
    return overlap / union


def _stable_noise(seed: str) -> float:
    digest = hashlib.sha256(seed.encode("utf8")).hexdigest()
    value = int(digest[:8], 16)
    return value / 0xFFFFFFFF


def _popularity_decay(rank_position: int, total_items: int, half_life: float) -> float:
    """Compute an exponential popularity decay weight based on rank position.

    Items near the top of the catalog (lower rank_position) receive weights
    close to 1.0.  The weight halves every ``half_life`` positions, producing
    a smooth decay curve that avoids the cliff effect of simple linear priors.
    """
    if total_items <= 1 or half_life <= 0.0:
        return 1.0
    decay_rate = math.log(2.0) / half_life
    return math.exp(-decay_rate * rank_position)


@dataclass(frozen=True)
class MetaFeaturePrior:
    """Contextual meta features used to bootstrap preference estimation
    for cold start users who have no interaction history.

    Each field carries a normalized weight in the range [0, 1].  When
    all fields are zero the cold start scorer falls back to pure lexical
    and popularity signals, which is the safe default behavior.
    """
    genre_affinity: float = 0.0
    recency_preference: float = 0.0
    popularity_preference: float = 0.5
    locale_boost: float = 0.0
    device_factor: float = 0.0

    def combined_weight(self) -> float:
        total = (
            self.genre_affinity
            + self.recency_preference
            + self.popularity_preference
            + self.locale_boost
            + self.device_factor
        )
        return min(max(total / 5.0, 0.0), 1.0)


@dataclass(frozen=True)
class ColdStartDecision:
    """An auditable record of every decision the cold start strategy made
    for a single candidate, capturing the scoring breakdown and the
    reason the candidate was included or excluded."""
    item_id: str
    title: str
    mode: str
    lexical_score: float
    prior_score: float
    personalization_score: float
    meta_prior_score: float
    popularity_decay_score: float
    final_score: float
    included: bool
    reason: str


@dataclass(frozen=True)
class ColdStartConfig:
    top_k: int = 200
    lexical_weight: float = 0.40
    prior_weight: float = 0.15
    personalization_weight: float = 0.05
    meta_prior_weight: float = 0.20
    popularity_decay_weight: float = 0.20
    popularity_half_life: float = 50.0
    minimum_score_threshold: float = 0.0


class ColdStartRetrievalStrategy:
    """Retrieval strategy for cold start scenarios where there is
    insufficient interaction history to power collaborative or
    embedding based retrieval.

    Supports two modes:

    1. ``new_user``:  the user has no history.  Uses lexical match,
       catalog popularity prior, meta feature priors (genre affinity,
       locale, device), and popularity decay.

    2. ``new_item``:  the item has no embedding.  Uses lexical match
       against the query, personalization noise seeded on user id,
       and popularity decay to surface fresh content.

    Both modes produce fully auditable ``ColdStartDecision`` records
    so that downstream systems can explain why each candidate was
    selected.
    """

    def __init__(self, config: ColdStartConfig | None = None) -> None:
        self._config = config or ColdStartConfig()

    @property
    def config(self) -> ColdStartConfig:
        return self._config

    def for_new_user(
        self,
        query_title: str,
        catalog,
        *,
        meta_prior: MetaFeaturePrior | None = None,
    ) -> list[Candidate]:
        candidates, _ = self._rank_candidates(
            query_title=query_title,
            catalog=catalog,
            user_id=None,
            mode="new_user",
            meta_prior=meta_prior,
        )
        return candidates

    def for_new_item(
        self,
        query_title: str,
        catalog,
        user_id: str | None = None,
        *,
        meta_prior: MetaFeaturePrior | None = None,
    ) -> list[Candidate]:
        candidates, _ = self._rank_candidates(
            query_title=query_title,
            catalog=catalog,
            user_id=user_id,
            mode="new_item",
            meta_prior=meta_prior,
        )
        return candidates

    def for_new_user_with_decisions(
        self,
        query_title: str,
        catalog,
        *,
        meta_prior: MetaFeaturePrior | None = None,
    ) -> tuple[list[Candidate], list[ColdStartDecision]]:
        return self._rank_candidates(
            query_title=query_title,
            catalog=catalog,
            user_id=None,
            mode="new_user",
            meta_prior=meta_prior,
        )

    def for_new_item_with_decisions(
        self,
        query_title: str,
        catalog,
        user_id: str | None = None,
        *,
        meta_prior: MetaFeaturePrior | None = None,
    ) -> tuple[list[Candidate], list[ColdStartDecision]]:
        return self._rank_candidates(
            query_title=query_title,
            catalog=catalog,
            user_id=user_id,
            mode="new_item",
            meta_prior=meta_prior,
        )

    def _rank_candidates(
        self,
        *,
        query_title: str,
        catalog,
        user_id: str | None,
        mode: str,
        meta_prior: MetaFeaturePrior | None,
    ) -> tuple[list[Candidate], list[ColdStartDecision]]:
        if catalog.empty:
            return [], []

        total_items = len(catalog)
        query_tokens = _tokenize(query_title)
        resolved_prior = meta_prior or MetaFeaturePrior()
        scored: list[tuple[Candidate, ColdStartDecision]] = []

        for row_index, row in catalog.reset_index(drop=True).iterrows():
            title = str(row["movie_title"])
            title_tokens = _tokenize(title)

            lexical = _jaccard(query_tokens, title_tokens)
            prior = 1.0 - (row_index / max(1, total_items - 1))
            personalization = 0.0
            if user_id:
                personalization = _stable_noise(f"{user_id}:{title}")

            meta_score = resolved_prior.combined_weight()
            decay_score = _popularity_decay(
                int(row_index),
                total_items,
                self._config.popularity_half_life,
            )

            score = (
                lexical * self._config.lexical_weight
                + prior * self._config.prior_weight
                + personalization * self._config.personalization_weight
                + meta_score * self._config.meta_prior_weight
                + decay_score * self._config.popularity_decay_weight
            )

            included = score >= self._config.minimum_score_threshold
            reason = f"scored_{mode}" if included else f"below_threshold_{mode}"

            signals: dict[str, float] = {
                "text": float(lexical),
                "multimodal": 0.0,
                "popularity": float(prior),
                "recency": 0.0,
                "novelty": float(1.0 - prior),
            }
            candidate = Candidate(
                item_id=str(row_index),
                title=title,
                score=float(score),
                channel=f"cold_start_{mode}",
                metadata={
                    "signals": signals,
                    "cold_start": {
                        "mode": mode,
                        "lexical_score": float(lexical),
                        "prior_score": float(prior),
                        "personalization_score": float(personalization),
                        "meta_prior_score": float(meta_score),
                        "popularity_decay_score": float(decay_score),
                    },
                },
            )
            decision = ColdStartDecision(
                item_id=str(row_index),
                title=title,
                mode=mode,
                lexical_score=float(lexical),
                prior_score=float(prior),
                personalization_score=float(personalization),
                meta_prior_score=float(meta_score),
                popularity_decay_score=float(decay_score),
                final_score=float(score),
                included=included,
                reason=reason,
            )
            scored.append((candidate, decision))

        scored.sort(key=lambda pair: pair[0].score, reverse=True)
        top = scored[: self._config.top_k]
        candidates = [pair[0] for pair in top]
        decisions = [pair[1] for pair in scored]
        return candidates, decisions

    def top_up(
        self,
        *,
        existing_item_ids: set[str],
        query_title: str,
        catalog,
        user_id: str | None,
        top_k: int,
        meta_prior: MetaFeaturePrior | None = None,
    ) -> list[Candidate]:
        if user_id:
            base = self.for_new_item(query_title, catalog, user_id=user_id, meta_prior=meta_prior)
        else:
            base = self.for_new_user(query_title, catalog, meta_prior=meta_prior)
        topped: list[Candidate] = []
        for candidate in base:
            if candidate.item_id in existing_item_ids:
                continue
            topped.append(candidate)
            if len(topped) >= top_k:
                break
        return topped

    def detect_cold_start(
        self,
        *,
        user_interaction_count: int,
        item_interaction_count: int,
        user_threshold: int = 5,
        item_threshold: int = 3,
    ) -> str:
        """Determine which cold start mode applies, if any.

        Returns one of:
            ``"warm"``     :  enough history exists on both sides.
            ``"new_user"`` :  the user has too few interactions.
            ``"new_item"`` :  the item has too few interactions.
            ``"full_cold"``:  both user and item lack history.
        """
        user_cold = user_interaction_count < user_threshold
        item_cold = item_interaction_count < item_threshold
        if user_cold and item_cold:
            return "full_cold"
        if user_cold:
            return "new_user"
        if item_cold:
            return "new_item"
        return "warm"
