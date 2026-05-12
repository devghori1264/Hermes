from __future__ import annotations

import pandas as pd
import pytest

from src.retrieval.cold_start import (
    ColdStartConfig,
    ColdStartDecision,
    ColdStartRetrievalStrategy,
    MetaFeaturePrior,
    _jaccard,
    _popularity_decay,
    _stable_noise,
    _tokenize,
)


def _small_catalog() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"movie_title": "the matrix"},
            {"movie_title": "the godfather"},
            {"movie_title": "inception"},
            {"movie_title": "interstellar"},
            {"movie_title": "the dark knight"},
            {"movie_title": "pulp fiction"},
            {"movie_title": "the matrix reloaded"},
            {"movie_title": "the matrix revolutions"},
        ]
    )


class TestTokenizer:
    def test_basic_tokenization(self) -> None:
        tokens = _tokenize("The Matrix 1999")
        assert tokens == {"the", "matrix", "1999"}

    def test_empty_string(self) -> None:
        assert _tokenize("") == set()

    def test_special_characters_ignored(self) -> None:
        tokens = _tokenize("hello!! world??")
        assert tokens == {"hello", "world"}


class TestJaccard:
    def test_identical_sets(self) -> None:
        assert _jaccard({"a", "b"}, {"a", "b"}) == 1.0

    def test_disjoint_sets(self) -> None:
        assert _jaccard({"a"}, {"b"}) == 0.0

    def test_empty_left(self) -> None:
        assert _jaccard(set(), {"a"}) == 0.0

    def test_empty_right(self) -> None:
        assert _jaccard({"a"}, set()) == 0.0

    def test_partial_overlap(self) -> None:
        result = _jaccard({"a", "b", "c"}, {"b", "c", "d"})
        assert abs(result - 0.5) < 1e-9


class TestStableNoise:
    def test_determinism(self) -> None:
        a = _stable_noise("user123:title")
        b = _stable_noise("user123:title")
        assert a == b

    def test_different_seeds_differ(self) -> None:
        a = _stable_noise("seed_a")
        b = _stable_noise("seed_b")
        assert a != b

    def test_range(self) -> None:
        value = _stable_noise("range_check")
        assert 0.0 <= value <= 1.0


class TestPopularityDecay:
    def test_top_item_has_weight_near_one(self) -> None:
        weight = _popularity_decay(0, 100, 50.0)
        assert abs(weight - 1.0) < 1e-9

    def test_half_life_item_has_weight_near_half(self) -> None:
        weight = _popularity_decay(50, 100, 50.0)
        assert abs(weight - 0.5) < 1e-6

    def test_single_item_catalog(self) -> None:
        weight = _popularity_decay(0, 1, 50.0)
        assert weight == 1.0

    def test_zero_half_life_returns_one(self) -> None:
        weight = _popularity_decay(10, 100, 0.0)
        assert weight == 1.0


class TestMetaFeaturePrior:
    def test_default_combined_weight(self) -> None:
        prior = MetaFeaturePrior()
        assert 0.0 <= prior.combined_weight() <= 1.0

    def test_all_zeros(self) -> None:
        prior = MetaFeaturePrior(
            genre_affinity=0.0,
            recency_preference=0.0,
            popularity_preference=0.0,
            locale_boost=0.0,
            device_factor=0.0,
        )
        assert prior.combined_weight() == 0.0

    def test_all_ones(self) -> None:
        prior = MetaFeaturePrior(
            genre_affinity=1.0,
            recency_preference=1.0,
            popularity_preference=1.0,
            locale_boost=1.0,
            device_factor=1.0,
        )
        assert prior.combined_weight() == 1.0


class TestColdStartRetrievalStrategy:
    def test_new_user_returns_candidates(self) -> None:
        strategy = ColdStartRetrievalStrategy()
        catalog = _small_catalog()
        candidates = strategy.for_new_user("the matrix", catalog)
        assert len(candidates) > 0
        assert all(c.channel == "cold_start_new_user" for c in candidates)

    def test_new_item_returns_candidates(self) -> None:
        strategy = ColdStartRetrievalStrategy()
        catalog = _small_catalog()
        candidates = strategy.for_new_item("inception", catalog, user_id="u42")
        assert len(candidates) > 0
        assert all(c.channel == "cold_start_new_item" for c in candidates)

    def test_empty_catalog_returns_empty(self) -> None:
        strategy = ColdStartRetrievalStrategy()
        empty = pd.DataFrame(columns=["movie_title"])
        assert strategy.for_new_user("anything", empty) == []
        assert strategy.for_new_item("anything", empty) == []

    def test_top_k_limits_output(self) -> None:
        config = ColdStartConfig(top_k=3)
        strategy = ColdStartRetrievalStrategy(config)
        catalog = _small_catalog()
        candidates = strategy.for_new_user("matrix", catalog)
        assert len(candidates) <= 3

    def test_lexical_match_ranks_higher(self) -> None:
        strategy = ColdStartRetrievalStrategy()
        catalog = _small_catalog()
        candidates = strategy.for_new_user("the matrix", catalog)
        titles = [c.title for c in candidates]
        matrix_indices = [i for i, t in enumerate(titles) if "matrix" in t.lower()]
        non_matrix_indices = [i for i, t in enumerate(titles) if "matrix" not in t.lower()]
        if matrix_indices and non_matrix_indices:
            assert min(matrix_indices) < max(non_matrix_indices)

    def test_new_user_decisions_are_auditable(self) -> None:
        strategy = ColdStartRetrievalStrategy()
        catalog = _small_catalog()
        candidates, decisions = strategy.for_new_user_with_decisions("the matrix", catalog)
        assert len(decisions) == len(catalog)
        assert all(isinstance(d, ColdStartDecision) for d in decisions)
        assert all(d.mode == "new_user" for d in decisions)
        for d in decisions:
            assert d.lexical_score >= 0.0
            assert d.prior_score >= 0.0
            assert d.final_score >= 0.0

    def test_new_item_decisions_are_auditable(self) -> None:
        strategy = ColdStartRetrievalStrategy()
        catalog = _small_catalog()
        candidates, decisions = strategy.for_new_item_with_decisions("inception", catalog, user_id="u1")
        assert len(decisions) == len(catalog)
        assert all(d.mode == "new_item" for d in decisions)

    def test_meta_prior_influences_score(self) -> None:
        catalog = _small_catalog()
        base_strategy = ColdStartRetrievalStrategy()
        boosted_strategy = ColdStartRetrievalStrategy()
        base_candidates = base_strategy.for_new_user("matrix", catalog, meta_prior=MetaFeaturePrior())
        high_prior = MetaFeaturePrior(
            genre_affinity=1.0,
            recency_preference=1.0,
            popularity_preference=1.0,
            locale_boost=1.0,
            device_factor=1.0,
        )
        boosted_candidates = boosted_strategy.for_new_user("matrix", catalog, meta_prior=high_prior)
        base_total = sum(c.score for c in base_candidates)
        boosted_total = sum(c.score for c in boosted_candidates)
        assert boosted_total > base_total

    def test_top_up_excludes_existing_ids(self) -> None:
        strategy = ColdStartRetrievalStrategy()
        catalog = _small_catalog()
        all_candidates = strategy.for_new_user("the matrix", catalog)
        existing = {all_candidates[0].item_id}
        topped = strategy.top_up(
            existing_item_ids=existing,
            query_title="the matrix",
            catalog=catalog,
            user_id=None,
            top_k=5,
        )
        for c in topped:
            assert c.item_id not in existing

    def test_top_up_respects_top_k(self) -> None:
        strategy = ColdStartRetrievalStrategy()
        catalog = _small_catalog()
        topped = strategy.top_up(
            existing_item_ids=set(),
            query_title="the matrix",
            catalog=catalog,
            user_id=None,
            top_k=2,
        )
        assert len(topped) <= 2

    def test_candidate_metadata_contains_signals(self) -> None:
        strategy = ColdStartRetrievalStrategy()
        catalog = _small_catalog()
        candidates = strategy.for_new_user("the matrix", catalog)
        for c in candidates:
            assert "signals" in c.metadata
            signals = c.metadata["signals"]
            assert "text" in signals
            assert "popularity" in signals
            assert "novelty" in signals

    def test_candidate_metadata_contains_cold_start_breakdown(self) -> None:
        strategy = ColdStartRetrievalStrategy()
        catalog = _small_catalog()
        candidates = strategy.for_new_user("the matrix", catalog)
        for c in candidates:
            cs = c.metadata["cold_start"]
            assert cs["mode"] == "new_user"
            assert "lexical_score" in cs
            assert "prior_score" in cs
            assert "meta_prior_score" in cs
            assert "popularity_decay_score" in cs

    def test_personalization_only_active_with_user_id(self) -> None:
        strategy = ColdStartRetrievalStrategy()
        catalog = _small_catalog()
        no_user = strategy.for_new_user("the matrix", catalog)
        with_user = strategy.for_new_item("the matrix", catalog, user_id="u1")
        for c in no_user:
            assert c.metadata["cold_start"]["personalization_score"] == 0.0
        has_nonzero = any(c.metadata["cold_start"]["personalization_score"] > 0.0 for c in with_user)
        assert has_nonzero


class TestDetectColdStart:
    def test_warm(self) -> None:
        strategy = ColdStartRetrievalStrategy()
        assert strategy.detect_cold_start(user_interaction_count=10, item_interaction_count=5) == "warm"

    def test_new_user(self) -> None:
        strategy = ColdStartRetrievalStrategy()
        assert strategy.detect_cold_start(user_interaction_count=2, item_interaction_count=5) == "new_user"

    def test_new_item(self) -> None:
        strategy = ColdStartRetrievalStrategy()
        assert strategy.detect_cold_start(user_interaction_count=10, item_interaction_count=1) == "new_item"

    def test_full_cold(self) -> None:
        strategy = ColdStartRetrievalStrategy()
        assert strategy.detect_cold_start(user_interaction_count=0, item_interaction_count=0) == "full_cold"

    def test_custom_thresholds(self) -> None:
        strategy = ColdStartRetrievalStrategy()
        result = strategy.detect_cold_start(
            user_interaction_count=10,
            item_interaction_count=10,
            user_threshold=20,
            item_threshold=20,
        )
        assert result == "full_cold"
