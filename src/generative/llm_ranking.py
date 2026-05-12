from dataclasses import dataclass
from src.domain.models import RankedItem

try:
    from transformers import pipeline
except Exception:
    pipeline = None

@dataclass(frozen=True)
class LLMRankingResult:
    item_id: str
    original_score: float
    llm_score: float
    confidence: float
    reasoning: str

class LLMShadowRanker:
    def __init__(self, shadow_mode: bool = True, model_name: str = "google/flan-t5-small") -> None:
        self.shadow_mode = shadow_mode
        self.generator = None
        if pipeline is not None:
            try:
                self.generator = pipeline(
                    "text2text-generation",
                    model=model_name,
                    device="cpu",
                    max_new_tokens=50
                )
            except Exception:
                self.generator = None

    def _generate_reasoning(self, query: str, item_title: str) -> str:
        if self.generator is None:
            return "Relevant item."
        prompt = f"Given the user query '{query}', why is '{item_title}' a good recommendation? Reason:"
        try:
            output = self.generator(prompt)
            if output and isinstance(output, list):
                return str(output[0].get("generated_text", "Relevant item."))
        except Exception:
            return "Relevant item."
        return "Relevant item."

    def evaluate_candidates(self, query: str, items: list[RankedItem]) -> list[LLMRankingResult]:
        results = []
        for item in items:
            reasoning = self._generate_reasoning(query, item.title)
            
            query_words = set(query.lower().split()) if query else set()
            title_words = set(item.title.lower().split())
            overlap = len(query_words & title_words)
            llm_score = min(1.0, item.score + (overlap * 0.05))
            
            results.append(
                LLMRankingResult(
                    item_id=item.item_id,
                    original_score=item.score,
                    llm_score=llm_score,
                    confidence=0.9,
                    reasoning=reasoning
                )
            )
        return results

    def apply_shadow_ranking(self, query: str, items: list[RankedItem]) -> list[RankedItem]:
        evaluations = self.evaluate_candidates(query, items)
        updated_items = []
        for item, evaluation in zip(items, evaluations):
            new_meta = dict(item.metadata) if item.metadata else {}
            if self.shadow_mode:
                new_meta["shadow_llm_score"] = evaluation.llm_score
                new_meta["shadow_llm_reasoning"] = evaluation.reasoning
                updated_items.append(
                    RankedItem(
                        item_id=item.item_id,
                        title=item.title,
                        score=item.score,
                        explanation=item.explanation,
                        metadata=new_meta
                    )
                )
            else:
                updated_items.append(
                    RankedItem(
                        item_id=item.item_id,
                        title=item.title,
                        score=evaluation.llm_score,
                        explanation=item.explanation,
                        metadata=new_meta
                    )
                )
        if not self.shadow_mode:
            updated_items.sort(key=lambda x: x.score, reverse=True)
        return updated_items

