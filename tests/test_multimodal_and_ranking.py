from src.features.multimodal_features import MultimodalFeatureService
from src.policy.reranker import PolicyReranker
from src.domain.models import RankedItem


def test_multimodal_fallback_produces_provenance() -> None:
    svc = MultimodalFeatureService()
    vector = svc.extract(title="Avatar The Way of Water", overview="A marine world epic")
    assert vector.provenance["mode"] == "deterministic_hash"
    assert len(vector.image_attributes) > 0


def test_policy_reranker_limits_same_prefix_items() -> None:
    reranker = PolicyReranker()
    items = [
        RankedItem(item_id=str(i), title=f"a_title_{i}", score=1.0 - i * 0.1, explanation="x")
        for i in range(4)
    ]
    out = reranker.apply(items)
    assert len(out) == 4
    assert out[0].score >= out[1].score


def test_policy_reranker_penalizes_cohort_overexposure() -> None:
    reranker = PolicyReranker(max_exposure_per_cohort=2)
    items = [
        RankedItem(item_id=str(i), title=f"title_{i}", score=1.0, explanation="x", metadata={"cohort": "action"})
        for i in range(4)
    ]
    out = reranker.apply(items)
    assert len(out) == 4
    assert out[2].score < out[0].score

