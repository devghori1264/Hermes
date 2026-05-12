from src.retrieval.blender import BlendInput, CandidateBlender
from src.retrieval.types import RetrievalCandidate


def test_blender_merges_sources() -> None:
    blender = CandidateBlender()
    left = [RetrievalCandidate(item_id="a", score=0.9, source="left")]
    right = [RetrievalCandidate(item_id="a", score=0.5, source="right")]

    result = blender.blend(
        [
            BlendInput(name="left", candidates=left, weight=1.0),
            BlendInput(name="right", candidates=right, weight=1.0),
        ],
        top_k=10,
    )

    assert len(result.candidates) == 1
    assert result.candidates[0].metadata.get("sources")
