from src.domain.models import RankedItem
from src.evaluation.online_harness import OnlineEvaluationHarness
from src.policy.diversity import DiversityConfig, DiversityOptimizer


def test_diversity_optimizer_spreads_groups() -> None:
    optimizer = DiversityOptimizer(DiversityConfig(max_per_group=1))
    items = [
        RankedItem(item_id="1", title="Alpha One", score=0.9, explanation="x", metadata={"category": "alpha"}),
        RankedItem(item_id="2", title="Alpha Two", score=0.8, explanation="x", metadata={"category": "alpha"}),
        RankedItem(item_id="3", title="Beta One", score=0.7, explanation="x", metadata={"category": "beta"}),
    ]

    out = optimizer.rerank(items)
    assert out[0].metadata["category"] == "alpha"
    assert out[1].metadata["category"] == "beta"


def test_online_evaluation_harness_summary() -> None:
    harness = OnlineEvaluationHarness()
    harness.record_impression(
        "imp1",
        [
            RankedItem(item_id="1", title="Alpha", score=0.9, explanation="x"),
            RankedItem(item_id="2", title="Beta", score=0.8, explanation="x"),
        ],
        clicked_item_ids={"1"},
        reward=1.0,
    )
    harness.record_impression(
        "imp2",
        [
            RankedItem(item_id="3", title="Gamma", score=0.7, explanation="x"),
            RankedItem(item_id="4", title="Delta", score=0.6, explanation="x"),
        ],
        clicked_item_ids=set(),
        reward=0.0,
    )

    summary = harness.summary()
    assert summary.impression_count == 2
    assert summary.click_count == 1
    assert 0.0 <= summary.ctr <= 1.0
    assert summary.unique_items == 4
