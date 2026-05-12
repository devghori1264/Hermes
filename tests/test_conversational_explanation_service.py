from src.domain.models import RankedItem
from src.services.conversational_explanation_service import ConversationalExplanationService


def test_explanation_bundle_contains_summary_and_signal_context() -> None:
    service = ConversationalExplanationService()
    ranked = [
        RankedItem(
            item_id="1",
            title="inception",
            score=0.93,
            explanation="ranked",
            metadata={"signals": {"text": 0.74, "multimodal": 0.19, "novelty": 0.03}},
        ),
        RankedItem(
            item_id="2",
            title="interstellar",
            score=0.91,
            explanation="ranked",
            metadata={"signals": {"text": 0.68, "multimodal": 0.18}},
        ),
    ]

    bundle = service.build("avatar", ranked)

    assert bundle.query_title == "avatar"
    assert "Mean confidence score" in bundle.summary
    assert len(bundle.items) == 2
    assert bundle.items[0].title == "inception"
    assert "Semantic Match" in bundle.items[0].primary_signals
    assert "avatar" in bundle.items[0].rationale


def test_explanation_bundle_handles_empty_rankings() -> None:
    service = ConversationalExplanationService()
    bundle = service.build("unknown", [])

    assert bundle.query_title == "unknown"
    assert bundle.items == ()
    assert "No confident recommendations" in bundle.summary
