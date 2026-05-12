from __future__ import annotations
from dataclasses import dataclass
from typing import Sequence

@dataclass(frozen=True)
class CausalEstimationResult:
    ate: float
    confidence_lower: float
    confidence_upper: float
    variance: float

class CausalEstimator:
    def estimate_treatment_effect(
        self,
        treatments: Sequence[int],
        outcomes: Sequence[float],
        propensities: Sequence[float]
    ) -> CausalEstimationResult:
        if not treatments or len(treatments) != len(outcomes) or len(treatments) != len(propensities):
            return CausalEstimationResult(ate=0.0, confidence_lower=0.0, confidence_upper=0.0, variance=0.0)

        n = len(treatments)
        ipw_estimates = []
        for w, y, p in zip(treatments, outcomes, propensities):
            prob = max(1e-6, min(1.0 - 1e-6, p))
            if w == 1:
                ipw_estimates.append(y / prob)
            else:
                ipw_estimates.append(-y / (1.0 - prob))

        ate = sum(ipw_estimates) / n
        mean = ate
        variance = sum((x - mean) ** 2 for x in ipw_estimates) / (n - 1) if n > 1 else 0.0
        import math
        std_err = math.sqrt(variance / n) if n > 0 else 0.0
        z_score = 1.96

        return CausalEstimationResult(
            ate=ate,
            confidence_lower=ate - z_score * std_err,
            confidence_upper=ate + z_score * std_err,
            variance=variance
        )

    def doubly_robust_estimate(
        self,
        treatments: Sequence[int],
        outcomes: Sequence[float],
        propensities: Sequence[float],
        predicted_outcomes_0: Sequence[float],
        predicted_outcomes_1: Sequence[float]
    ) -> CausalEstimationResult:
        if not treatments or len(treatments) != len(outcomes) or len(treatments) != len(propensities):
            return CausalEstimationResult(ate=0.0, confidence_lower=0.0, confidence_upper=0.0, variance=0.0)

        n = len(treatments)
        dr_estimates = []
        for w, y, p, y0_hat, y1_hat in zip(treatments, outcomes, propensities, predicted_outcomes_0, predicted_outcomes_1):
            prob = max(1e-6, min(1.0 - 1e-6, p))
            if w == 1:
                dr_1 = y1_hat + (y - y1_hat) / prob
                dr_0 = y0_hat
            else:
                dr_1 = y1_hat
                dr_0 = y0_hat + (y - y0_hat) / (1.0 - prob)
            dr_estimates.append(dr_1 - dr_0)

        ate = sum(dr_estimates) / n
        mean = ate
        variance = sum((x - mean) ** 2 for x in dr_estimates) / (n - 1) if n > 1 else 0.0
        import math
        std_err = math.sqrt(variance / n) if n > 0 else 0.0
        z_score = 1.96

        return CausalEstimationResult(
            ate=ate,
            confidence_lower=ate - z_score * std_err,
            confidence_upper=ate + z_score * std_err,
            variance=variance
        )
