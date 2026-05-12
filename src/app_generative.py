from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
from src.generative.conversational_agent import ConversationalAgent

app = FastAPI(title="Hermes Generative Service")
agent = ConversationalAgent()

class ChatRequest(BaseModel):
    query: str
    context_items: List[str]

@app.post("/api/v1/chat")
def chat_endpoint(req: ChatRequest):
    try:
        response = agent.chat(req.query, req.context_items)
        return {"status": "success", "response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
