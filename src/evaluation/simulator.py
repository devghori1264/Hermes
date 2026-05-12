from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Iterable


@dataclass(frozen=True)
class SimulatedOutcome:
    item_id: str
    clicked: bool
    reward: float


class BehaviorModel:
    def __init__(self, base_click_rate: float = 0.05, seed: int | None = None) -> None:
        self._base_click_rate = base_click_rate
        self._rng = random.Random(seed)

    def click_probability(self, score: float) -> float:
        return min(max(self._base_click_rate + score * 0.1, 0.0), 1.0)

    def simulate(self, ranked_items: Iterable[tuple[str, float]]) -> list[SimulatedOutcome]:
        outcomes: list[SimulatedOutcome] = []
        for item_id, score in ranked_items:
            prob = self.click_probability(score)
            clicked = self._rng.random() < prob
            reward = 1.0 if clicked else 0.0
            outcomes.append(SimulatedOutcome(item_id=item_id, clicked=clicked, reward=reward))
        return outcomes
