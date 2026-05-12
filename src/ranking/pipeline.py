from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from src.domain.models import Candidate, RankedItem
from src.ranking.scoring import ScoreWeights, compute_score
import math


@dataclass(frozen=True)
class RankingBudget:
    max_candidates: int = 200
    pre_rank_k: int = 50
    final_k: int = 10


@dataclass(frozen=True)
class RankingConfig:
    pre_rank_weights: ScoreWeights = ScoreWeights()
    fine_rank_weights: ScoreWeights = ScoreWeights()


class RankingPipeline:
    def __init__(self, budget: RankingBudget, config: RankingConfig | None = None) -> None:
        self.budget = budget
        self.config = config or RankingConfig()

    def pre_rank(self, candidates: Iterable[Candidate]) -> list[Candidate]:
        ranked = sorted(
            candidates,
            key=lambda candidate: compute_score(candidate, self.config.pre_rank_weights).total,
            reverse=True,
        )
        return ranked[: self.budget.pre_rank_k]

    def fine_rank(self, candidates: Iterable[Candidate]) -> list[RankedItem]:
        ranked = sorted(
            candidates,
            key=lambda candidate: compute_score(candidate, self.config.fine_rank_weights).total,
            reverse=True,
        )
        top = ranked[: self.budget.final_k]
        return [
            self._ranked_item(candidate)
            for candidate in top
        ]

    def _ranked_item(self, candidate: Candidate) -> RankedItem:
        breakdown = compute_score(candidate, self.config.fine_rank_weights)
        explanation = (
            f"channel={candidate.channel}"
            f"|score={breakdown.total:.6f}"
            f"|base={breakdown.base:.6f}"
            f"|text={breakdown.text:.6f}"
            f"|multimodal={breakdown.multimodal:.6f}"
            f"|popularity={breakdown.popularity:.6f}"
            f"|recency={breakdown.recency:.6f}"
            f"|novelty={breakdown.novelty:.6f}"
        )
        score_total = float(breakdown.total)
        calibration_prob = 1.0 / (1.0 + math.exp(-score_total))
        uncertainty = 1.0 - abs(2.0 * calibration_prob - 1.0)

        original_signals = {}
        if isinstance(candidate.metadata, dict):
            original_signals = candidate.metadata.get("signals", {})

        return RankedItem(
            item_id=candidate.item_id,
            title=candidate.title,
            score=score_total,
            explanation=explanation,
            metadata={
                "score_breakdown": breakdown.to_metadata(),
                "calibrated_probability": calibration_prob,
                "uncertainty_score": uncertainty,
                "signals": original_signals,
            },
        )
