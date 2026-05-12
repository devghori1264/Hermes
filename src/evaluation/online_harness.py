from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

from src.evaluation.offline_metrics import map_at_k, mrr_at_k, ndcg_at_k, recall_at_k
from src.evaluation.simulator import BehaviorModel, SimulatedOutcome
from src.domain.models import RankedItem
from src.evaluation.fairness import FairnessAuditEngine, FairnessAuditReport
from src.evaluation.calibration import CalibrationEngine, CalibrationResult


@dataclass(frozen=True)
class OnlineImpression:
    impression_id: str
    recommended_item_ids: list[str]
    clicked_item_ids: set[str]
    scores: list[float]
    reward: float
    recommended_items: list[RankedItem] = field(default_factory=list)


@dataclass(frozen=True)
class OnlineEvaluationSummary:
    impression_count: int
    click_count: int
    ctr: float
    mean_reward: float
    coverage: float
    ndcg_at_10: float
    mrr_at_10: float
    map_at_10: float
    recall_at_10: float
    unique_items: int
    fairness_report: FairnessAuditReport | None = None
    calibration_result: CalibrationResult | None = None


class OnlineEvaluationHarness:
    def __init__(
        self,
        behavior_model: BehaviorModel | None = None,
        fairness_engine: FairnessAuditEngine | None = None,
        calibration_engine: CalibrationEngine | None = None,
        catalog_cohort_counts: dict[str, int] | None = None,
        catalog_popularity_scores: dict[str, float] | None = None,
        catalog_provider_counts: dict[str, int] | None = None,
    ) -> None:
        self.behavior_model = behavior_model or BehaviorModel()
        self.fairness_engine = fairness_engine
        self.calibration_engine = calibration_engine
        self.catalog_cohort_counts = catalog_cohort_counts
        self.catalog_popularity_scores = catalog_popularity_scores
        self.catalog_provider_counts = catalog_provider_counts
        self._impressions: list[OnlineImpression] = []

    def record_impression(
        self,
        impression_id: str,
        recommended_items: Iterable[RankedItem],
        clicked_item_ids: Iterable[str],
        reward: float | None = None,
    ) -> OnlineImpression:
        items = list(recommended_items)
        recommended_item_ids = [item.item_id for item in items]
        clicked_set = {str(item_id) for item_id in clicked_item_ids}
        if reward is None:
            reward = float(sum(1.0 for item_id in recommended_item_ids if item_id in clicked_set))
        impression = OnlineImpression(
            impression_id=impression_id,
            recommended_item_ids=recommended_item_ids,
            clicked_item_ids=clicked_set,
            scores=[float(item.score) for item in items],
            reward=float(reward),
            recommended_items=items,
        )
        self._impressions.append(impression)
        return impression

    def simulate_impression(self, impression_id: str, recommended_items: Iterable[RankedItem]) -> OnlineImpression:
        items = list(recommended_items)
        outcomes: list[SimulatedOutcome] = self.behavior_model.simulate(
            [(item.item_id, item.score) for item in items]
        )
        clicked = {outcome.item_id for outcome in outcomes if outcome.clicked}
        reward = float(sum(outcome.reward for outcome in outcomes))
        return self.record_impression(impression_id, items, clicked, reward=reward)

    def summary(self) -> OnlineEvaluationSummary:
        impression_count = len(self._impressions)
        if impression_count == 0:
            return OnlineEvaluationSummary(0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0)

        click_count = sum(len(impression.clicked_item_ids) for impression in self._impressions)
        total_recommendations = sum(len(impression.recommended_item_ids) for impression in self._impressions)
        unique_items = {item_id for impression in self._impressions for item_id in impression.recommended_item_ids}

        ndcg = []
        mrr = []
        map_scores = []
        recall_scores = []
        for impression in self._impressions:
            relevant = set(impression.clicked_item_ids)
            ndcg.append(ndcg_at_k(impression.recommended_item_ids, relevant, 10))
            mrr.append(mrr_at_k(impression.recommended_item_ids, relevant, 10))
            map_scores.append(map_at_k(impression.recommended_item_ids, relevant, 10))
            recall_scores.append(recall_at_k(impression.recommended_item_ids, relevant, 10))

        coverage = len(unique_items) / total_recommendations if total_recommendations else 0.0
        ctr = click_count / total_recommendations if total_recommendations else 0.0
        mean_reward = sum(impression.reward for impression in self._impressions) / impression_count
        
        fairness_report = None
        if self.fairness_engine is not None and self.catalog_cohort_counts is not None and self.catalog_popularity_scores is not None:
            all_items = [item for imp in self._impressions for item in imp.recommended_items]
            fairness_report = self.fairness_engine.audit(
                recommended_items=all_items,
                catalog_cohort_counts=self.catalog_cohort_counts,
                catalog_popularity_scores=self.catalog_popularity_scores,
                catalog_provider_counts=self.catalog_provider_counts
            )

        calibration_result = None
        if self.calibration_engine is not None:
            preds = []
            targets = []
            for imp in self._impressions:
                for item in imp.recommended_items:
                    preds.append(item.score)
                    targets.append(1 if item.item_id in imp.clicked_item_ids else 0)
            calibration_result = self.calibration_engine.evaluate(preds, targets)

        return OnlineEvaluationSummary(
            impression_count=impression_count,
            click_count=click_count,
            ctr=float(ctr),
            mean_reward=float(mean_reward),
            coverage=float(coverage),
            ndcg_at_10=float(sum(ndcg) / len(ndcg)),
            mrr_at_10=float(sum(mrr) / len(mrr)),
            map_at_10=float(sum(map_scores) / len(map_scores)),
            recall_at_10=float(sum(recall_scores) / len(recall_scores)),
            unique_items=len(unique_items),
            fairness_report=fairness_report,
            calibration_result=calibration_result,
        )
