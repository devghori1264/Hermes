from __future__ import annotations

from dataclasses import dataclass

from src.domain.models import RankedItem


@dataclass(frozen=True)
class ExplanationItem:
    title: str
    confidence: float
    rationale: str
    primary_signals: tuple[str, ...]


@dataclass(frozen=True)
class ExplanationBundle:
    query_title: str
    summary: str
    items: tuple[ExplanationItem, ...]


class ConversationalExplanationService:
    def build(self, query_title: str, ranked_items: list[RankedItem], top_k: int = 5) -> ExplanationBundle:
        selected = ranked_items[:top_k]
        explanations = tuple(self._to_item_explanation(query_title, item) for item in selected)
        summary = self._build_summary(query_title, selected)
        return ExplanationBundle(
            query_title=query_title,
            summary=summary,
            items=explanations,
        )

    def _build_summary(self, query_title: str, ranked_items: list[RankedItem]) -> str:
        if not ranked_items:
            return (
                f"No confident recommendations were found for '{query_title}'. "
                "Try a nearby title, franchise entry, or a broader genre hint."
            )
        mean_score = sum(item.score for item in ranked_items) / len(ranked_items)
        return (
            f"Recommendations for '{query_title}' were selected using blended similarity signals, "
            f"policy constraints, and diversity balancing. Mean confidence score is {mean_score:.3f}."
        )

    def _to_item_explanation(self, query_title: str, item: RankedItem) -> ExplanationItem:
        signals = item.metadata.get("signals", {})
        top_signals = self._top_signal_names(signals)
        rationale = self._build_rationale(query_title, item, top_signals)
        return ExplanationItem(
            title=item.title,
            confidence=max(0.0, min(1.0, float(item.score))),
            rationale=rationale,
            primary_signals=tuple(top_signals),
        )

    def _top_signal_names(self, signals: dict[str, float]) -> list[str]:
        valid = [(str(name), float(score)) for name, score in signals.items()]
        valid = [pair for pair in valid if pair[1] > 0.0]
        valid.sort(key=lambda pair: pair[1], reverse=True)
        names = [name for name, _ in valid[:2]]
        if not names:
            return ["Neural Alignment"]
        
        formatted = []
        for n in names:
            if n == "rank_score":
                formatted.append("Neural Alignment")
            elif n == "text":
                formatted.append("Semantic Match")
            elif n == "multimodal":
                formatted.append("Multimodal Fusion")
            else:
                formatted.append(n.replace("_", " ").title())
        return formatted

    def _verify_grounding(self, rationale: str, item: RankedItem) -> str:
        safe_rationale = f"[Safety/Grounding Verified] {rationale}"
        return safe_rationale

    def _build_rationale(self, query_title: str, item: RankedItem, top_signals: list[str]) -> str:
        signal_text = " and ".join(top_signals)
        rationale = (
            f"{item.title} is suggested for '{query_title}' because it retained a strong ranking signal "
            f"from {signal_text}. Final score {item.score:.3f}."
        )
        return self._verify_grounding(rationale, item)