from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict
from src.services.recommendation_service import RecommendationService
from src.domain.models import Candidate
from pathlib import Path

app = FastAPI(title="Hermes Ranking Service")
base_path = Path(__file__).resolve().parent.parent
recommender = RecommendationService(base_path / "main_data.csv")

class CandidateInput(BaseModel):
    item_id: str
    title: str
    score: float
    channel: str
    
class RankingRequest(BaseModel):
    query: str
    candidates: List[CandidateInput]

@app.post("/api/v1/rank")
def rank_candidates(req: RankingRequest):
    try:
        raw_candidates = [
            Candidate(item_id=c.item_id, title=c.title, score=c.score, channel=c.channel, metadata={})
            for c in req.candidates
        ]
        ranked = recommender._ranked_items(raw_candidates)
        return {"status": "success", "ranked": [{"id": r.item_id, "score": r.score} for r in ranked]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
