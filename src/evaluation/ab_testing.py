from __future__ import annotations
from dataclasses import dataclass
import math
from typing import Sequence

@dataclass(frozen=True)
class ABTestVariant:
    name: str
    allocation_fraction: float

@dataclass(frozen=True)
class ABTestResult:
    variant_a: str
    variant_b: str
    lift: float
    p_value: float
    significant: bool
    gate_passed: bool

class SequentialABTester:
    def __init__(self, alpha: float = 0.05, power: float = 0.8) -> None:
        self.alpha = alpha
        self.power = power

    def assign_variant(self, user_id: str, variants: list[ABTestVariant]) -> str:
        import hashlib
        hash_val = int(hashlib.md5(user_id.encode("utf8")).hexdigest(), 16)
        normalized = (hash_val % 10000) / 10000.0
        cumulative = 0.0
        for variant in variants:
            cumulative += variant.allocation_fraction
            if normalized <= cumulative:
                return variant.name
        return variants[-1].name if variants else "control"

    def evaluate(self, metrics_a: Sequence[float], metrics_b: Sequence[float], mde: float = 0.05) -> ABTestResult:
        n_a = len(metrics_a)
        n_b = len(metrics_b)
        if n_a < 30 or n_b < 30:
            return ABTestResult("control", "treatment", 0.0, 1.0, False, False)

        mean_a = sum(metrics_a) / n_a
        mean_b = sum(metrics_b) / n_b
        var_a = sum((x - mean_a) ** 2 for x in metrics_a) / (n_a - 1)
        var_b = sum((x - mean_b) ** 2 for x in metrics_b) / (n_b - 1)

        lift = (mean_b - mean_a) / mean_a if mean_a > 0 else 0.0
        pooled_se = math.sqrt((var_a / n_a) + (var_b / n_b))
        
        if pooled_se == 0:
            return ABTestResult("control", "treatment", lift, 1.0, False, False)

        z_score = (mean_b - mean_a) / pooled_se
        import scipy.stats
        p_value = 2.0 * (1.0 - scipy.stats.norm.cdf(abs(z_score)))

        significant = bool(p_value < self.alpha)
        gate_passed = significant and (lift >= mde)

        return ABTestResult(
            variant_a="control",
            variant_b="treatment",
            lift=lift,
            p_value=p_value,
            significant=significant,
            gate_passed=gate_passed
        )

class ReleaseGateAutomator:
    def check_gates(self, test_result: ABTestResult, fairness_passed: bool, causal_passed: bool) -> bool:
        return test_result.gate_passed and fairness_passed and causal_passed
