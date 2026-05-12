from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from src.services.recommendation_service import RecommendationService
from pathlib import Path

app = FastAPI(title="Hermes Retrieval Service")
base_path = Path(__file__).resolve().parent.parent
recommender = RecommendationService(base_path / "main_data.csv")

class RetrievalRequest(BaseModel):
    query: str
    domain: str = "movies"
    top_k: int = 50

@app.post("/api/v1/retrieve")
def retrieve_candidates(req: RetrievalRequest):
    try:
        from src.domain.models import RecommendationContext
        ctx = RecommendationContext(query_item_title=req.query)
        candidates, _ = recommender.cold_start.for_new_user_with_decisions(req.query, recommender.repo.load_catalog())
        return {"status": "success", "candidates": [{"id": c.item_id, "score": c.score} for c in candidates[:req.top_k]]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
