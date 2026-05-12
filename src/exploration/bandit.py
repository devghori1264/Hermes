from __future__ import annotations

from dataclasses import dataclass, field
import random
from typing import Any


@dataclass(frozen=True)
class BanditAction:
    item_id: str
    score: float
    metadata: dict[str, Any] = field(default_factory=dict)


class BanditPolicy:
    name: str

    def select(self, actions: list[BanditAction], top_k: int) -> list[BanditAction]:
        raise NotImplementedError


class EpsilonGreedyPolicy(BanditPolicy):
    def __init__(self, epsilon: float = 0.05, seed: int | None = None) -> None:
        self.name = "epsilon_greedy"
        self._epsilon = epsilon
        self._rng = random.Random(seed)

    def select(self, actions: list[BanditAction], top_k: int) -> list[BanditAction]:
        if not actions or top_k <= 0:
            return []
        ranked = sorted(actions, key=lambda action: action.score, reverse=True)
        if self._rng.random() > self._epsilon:
            return ranked[:top_k]
        choices = list(actions)
        self._rng.shuffle(choices)
        return choices[:top_k]
