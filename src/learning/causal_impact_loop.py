from dataclasses import dataclass
from src.learning.causal import CausalEstimator, CausalEstimationResult

@dataclass(frozen=True)
class GuardrailResult:
    passed: bool
    reason: str
    ate: float
    confidence_lower: float

class CausalGuardrail:
    def __init__(self, min_ate: float = 0.01, min_confidence_lower: float = 0.0) -> None:
        self.min_ate = min_ate
        self.min_confidence_lower = min_confidence_lower
        self.estimator = CausalEstimator()

    def evaluate_experiment(
        self,
        treatments: list[int],
        outcomes: list[float],
        propensities: list[float],
        predicted_outcomes_0: list[float],
        predicted_outcomes_1: list[float]
    ) -> GuardrailResult:
        result = self.estimator.doubly_robust_estimate(
            treatments=treatments,
            outcomes=outcomes,
            propensities=propensities,
            predicted_outcomes_0=predicted_outcomes_0,
            predicted_outcomes_1=predicted_outcomes_1
        )
        
        passed = True
        reason = "Experiment passed causal guardrails."
        
        if result.ate < self.min_ate:
            passed = False
            reason = f"ATE {result.ate:.4f} is below minimum threshold {self.min_ate:.4f}."
        elif result.confidence_lower < self.min_confidence_lower:
            passed = False
            reason = f"Lower confidence bound {result.confidence_lower:.4f} is below zero, effect not statistically significant."
            
        return GuardrailResult(
            passed=passed,
            reason=reason,
            ate=result.ate,
            confidence_lower=result.confidence_lower
        )
