from __future__ import annotations
from dataclasses import dataclass
from typing import Sequence

@dataclass(frozen=True)
class CalibrationResult:
    ece: float
    brier_score: float

class CalibrationEngine:
    def __init__(self, n_bins: int = 10):
        self.n_bins = n_bins

    def evaluate(self, predictions: Sequence[float], targets: Sequence[int]) -> CalibrationResult:
        if not predictions or len(predictions) != len(targets):
            return CalibrationResult(ece=0.0, brier_score=0.0)
            
        bins = [[] for _ in range(self.n_bins)]
        for p, t in zip(predictions, targets):
            # Clamp p just in case
            p_clamped = max(0.0, min(1.0, p))
            bin_idx = min(int(p_clamped * self.n_bins), self.n_bins - 1)
            bins[bin_idx].append((p_clamped, t))
            
        ece = 0.0
        brier = 0.0
        n = len(predictions)
        
        for b in bins:
            if not b:
                continue
            bin_preds = [x[0] for x in b]
            bin_targets = [x[1] for x in b]
            avg_pred = sum(bin_preds) / len(b)
            avg_target = sum(bin_targets) / len(b)
            ece += len(b) * abs(avg_pred - avg_target)
            
        for p, t in zip(predictions, targets):
            brier += (p - t) ** 2
            
        return CalibrationResult(
            ece=ece / n,
            brier_score=brier / n
        )
