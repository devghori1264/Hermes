from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable


@dataclass(frozen=True)
class LoggedEvent:
    """A single logged interaction from a production policy.

    ``reward``: the observed outcome (click, conversion, etc.).
    ``logging_propensity``: the probability assigned by the logging
    policy to the action that was taken.
    ``target_propensity``: the probability that the target (evaluation)
    policy would have assigned to the same action.
    ``direct_reward_estimate``: an optional reward model prediction
    used by the doubly robust estimator.  When absent the doubly
    robust estimator falls back to standard IPS behavior.
    """
    reward: float
    logging_propensity: float
    target_propensity: float
    direct_reward_estimate: float | None = None


@dataclass(frozen=True)
class OffPolicyEstimate:
    """The result of an off policy evaluation run.

    ``name``: estimator identifier (ips, snips, doubly_robust, etc.).
    ``value``: the point estimate of the target policy's expected reward.
    ``count``: the number of valid events used in the computation.
    ``variance``: the sample variance of the per event importance
    weighted rewards, useful for constructing confidence intervals.
    ``confidence_lower``: lower bound of a 95 percent normal
    approximation confidence interval.
    ``confidence_upper``: upper bound of the same interval.
    """
    name: str
    value: float
    count: int
    variance: float = 0.0
    confidence_lower: float = 0.0
    confidence_upper: float = 0.0


_Z_95 = 1.96


def _safe_propensity_ratio(
    target: float,
    logging: float,
    clip_min: float,
    clip_max: float,
) -> float | None:
    """Compute the importance weight target/logging with optional clipping.

    Returns None if either propensity is non positive, which signals
    that the event should be skipped.
    """
    if logging <= 0.0 or target <= 0.0:
        return None
    ratio = target / logging
    return max(clip_min, min(clip_max, ratio))


def _variance_and_ci(
    weighted_rewards: list[float],
    mean: float,
    count: int,
) -> tuple[float, float, float]:
    """Compute sample variance and a 95 percent confidence interval
    from a list of per event weighted reward values."""
    if count < 2:
        return 0.0, mean, mean
    variance = sum((w - mean) ** 2 for w in weighted_rewards) / (count - 1)
    std_error = math.sqrt(variance / count)
    lower = mean - _Z_95 * std_error
    upper = mean + _Z_95 * std_error
    return float(variance), float(lower), float(upper)


def inverse_propensity_score(
    events: Iterable[LoggedEvent],
    *,
    clip_min: float = 0.0,
    clip_max: float = float("inf"),
) -> OffPolicyEstimate:
    """Standard Inverse Propensity Scoring (IPS) estimator.

    Each event contributes reward * (target_propensity / logging_propensity)
    to the estimate.  The final value is the mean of these weighted rewards.

    Propensity ratio clipping is applied when ``clip_min`` or ``clip_max``
    are provided, which reduces variance at the cost of introducing bias.
    This tradeoff is well documented in Bottou et al. (2013) and
    Swaminathan and Joachims (2015).
    """
    weighted: list[float] = []
    for event in events:
        ratio = _safe_propensity_ratio(event.target_propensity, event.logging_propensity, clip_min, clip_max)
        if ratio is None:
            continue
        weighted.append(ratio * event.reward)
    count = len(weighted)
    mean = sum(weighted) / count if count > 0 else 0.0
    variance, lower, upper = _variance_and_ci(weighted, mean, count)
    return OffPolicyEstimate(
        name="ips",
        value=float(mean),
        count=count,
        variance=variance,
        confidence_lower=lower,
        confidence_upper=upper,
    )


def self_normalized_ips(
    events: Iterable[LoggedEvent],
    *,
    clip_min: float = 0.0,
    clip_max: float = float("inf"),
) -> OffPolicyEstimate:
    """Self Normalized Inverse Propensity Scoring (SNIPS) estimator.

    Normalizes the importance weighted rewards by the sum of the
    importance weights, which reduces variance compared to standard
    IPS especially when the importance weights have high variance.

    Reference: Swaminathan and Joachims (2015), "The Self Normalized
    Estimator for Counterfactual Learning".
    """
    weighted_rewards: list[float] = []
    weights: list[float] = []
    for event in events:
        ratio = _safe_propensity_ratio(event.target_propensity, event.logging_propensity, clip_min, clip_max)
        if ratio is None:
            continue
        weighted_rewards.append(ratio * event.reward)
        weights.append(ratio)
    count = len(weighted_rewards)
    total_weight = sum(weights)
    mean = sum(weighted_rewards) / total_weight if total_weight > 0 else 0.0
    variance, lower, upper = _variance_and_ci(weighted_rewards, mean, count)
    return OffPolicyEstimate(
        name="snips",
        value=float(mean),
        count=count,
        variance=variance,
        confidence_lower=lower,
        confidence_upper=upper,
    )


def doubly_robust(
    events: Iterable[LoggedEvent],
    *,
    clip_min: float = 0.0,
    clip_max: float = float("inf"),
) -> OffPolicyEstimate:
    """Doubly Robust (DR) off policy estimator.

    Combines a direct reward model estimate with importance weighted
    corrections to produce an estimate that is consistent when either
    the reward model or the propensity model is correct.

    For each event the contribution is:

        direct_estimate + ratio * (reward - direct_estimate)

    When ``direct_reward_estimate`` is missing from an event the
    estimator treats the direct estimate as zero, which causes it
    to fall back to standard IPS behavior for that event.

    Reference: Dudik, Langford, and Li (2011), "Doubly Robust Policy
    Evaluation and Learning".
    """
    adjusted: list[float] = []
    for event in events:
        ratio = _safe_propensity_ratio(event.target_propensity, event.logging_propensity, clip_min, clip_max)
        if ratio is None:
            continue
        direct = event.direct_reward_estimate if event.direct_reward_estimate is not None else 0.0
        value = direct + ratio * (event.reward - direct)
        adjusted.append(value)
    count = len(adjusted)
    mean = sum(adjusted) / count if count > 0 else 0.0
    variance, lower, upper = _variance_and_ci(adjusted, mean, count)
    return OffPolicyEstimate(
        name="doubly_robust",
        value=float(mean),
        count=count,
        variance=variance,
        confidence_lower=lower,
        confidence_upper=upper,
    )


def sensitivity_analysis(
    events: Iterable[LoggedEvent],
    *,
    perturbation_factors: tuple[float, ...] = (0.8, 0.9, 1.0, 1.1, 1.2),
    clip_min: float = 0.0,
    clip_max: float = float("inf"),
) -> list[OffPolicyEstimate]:
    """Run IPS estimation under multiple propensity perturbation factors.

    This tests how sensitive the off policy estimate is to misspecification
    of the logging propensity.  Each factor multiplies every event's
    ``logging_propensity`` before computing the IPS estimate.

    A robust estimate should show small variation across perturbation
    factors.  Large swings indicate that the estimate depends heavily
    on accurate propensity modeling.

    Reference: Kallus and Zhou (2020), "Confounding Robust Policy
    Evaluation in Infinite Horizon Reinforcement Learning".
    """
    event_list = list(events)
    results: list[OffPolicyEstimate] = []
    for factor in perturbation_factors:
        perturbed = [
            LoggedEvent(
                reward=e.reward,
                logging_propensity=e.logging_propensity * factor,
                target_propensity=e.target_propensity,
                direct_reward_estimate=e.direct_reward_estimate,
            )
            for e in event_list
        ]
        estimate = inverse_propensity_score(perturbed, clip_min=clip_min, clip_max=clip_max)
        results.append(
            OffPolicyEstimate(
                name=f"ips_perturbed_{factor:.2f}",
                value=estimate.value,
                count=estimate.count,
                variance=estimate.variance,
                confidence_lower=estimate.confidence_lower,
                confidence_upper=estimate.confidence_upper,
            )
        )
    return results


def batch_off_policy_report(
    events: Iterable[LoggedEvent],
    *,
    clip_min: float = 0.0,
    clip_max: float = float("inf"),
) -> dict[str, OffPolicyEstimate]:
    """Run all available off policy estimators on the same event stream
    and return a dictionary keyed by estimator name.

    This is the standard entry point for generating a comprehensive
    off policy evaluation report that can be stored alongside experiment
    artifacts.
    """
    event_list = list(events)
    
    report = {
        "ips": inverse_propensity_score(event_list, clip_min=clip_min, clip_max=clip_max),
        "snips": self_normalized_ips(event_list, clip_min=clip_min, clip_max=clip_max),
        "doubly_robust": doubly_robust(event_list, clip_min=clip_min, clip_max=clip_max),
    }

    try:
        from src.learning.causal import CausalEstimator
        estimator = CausalEstimator()
        treatments = [1 if e.target_propensity > e.logging_propensity else 0 for e in event_list]
        outcomes = [e.reward for e in event_list]
        propensities = [e.logging_propensity for e in event_list]
        predicted_outcomes_0 = [e.direct_reward_estimate if e.direct_reward_estimate else 0.0 for e in event_list]
        predicted_outcomes_1 = [e.direct_reward_estimate if e.direct_reward_estimate else 0.0 for e in event_list]
        
        causal_result = estimator.doubly_robust_estimate(
            treatments=treatments,
            outcomes=outcomes,
            propensities=propensities,
            predicted_outcomes_0=predicted_outcomes_0,
            predicted_outcomes_1=predicted_outcomes_1
        )
        
        report["causal_ate"] = OffPolicyEstimate(
            name="causal_ate",
            value=causal_result.ate,
            count=len(treatments),
            variance=causal_result.variance,
            confidence_lower=causal_result.confidence_lower,
            confidence_upper=causal_result.confidence_upper
        )
    except ImportError:
        pass

    return report
