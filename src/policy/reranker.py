from __future__ import annotations
from collections import defaultdict
from src.domain.models import RankedItem


class PolicyReranker:
    def __init__(self, max_exposure_per_cohort: int = 3) -> None:
        self.max_exposure_per_cohort = max_exposure_per_cohort

    def apply(self, ranked_items: list[RankedItem]) -> list[RankedItem]:
        cohort_exposure = defaultdict(int)
        output: list[RankedItem] = []

        for item in ranked_items:
            cohort = item.metadata.get("cohort") if item.metadata else None

            penalty = 0.0
            if cohort is not None:
                exposure_count = cohort_exposure[cohort]
                if exposure_count >= self.max_exposure_per_cohort:
                    penalty = 0.1 * (exposure_count - self.max_exposure_per_cohort + 1)
                cohort_exposure[cohort] += 1

            penalized_item = RankedItem(
                item_id=item.item_id,
                title=item.title,
                score=item.score - penalty,
                explanation=item.explanation,
                metadata=item.metadata
            )
            output.append(penalized_item)

        output.sort(key=lambda x: x.score, reverse=True)
        return output
