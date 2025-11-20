from fastapi import FastAPI
from pydantic import BaseModel

from app.guardian_model import guardian_analyzer


app = FastAPI(title="Safyrix Guardian AI", version="1.0.0")


class AnalyzeRequest(BaseModel):
    message: str
    user_id: str = "anonymous"


class AnalyzeResponse(BaseModel):
    risk_level: str
    confidence: float
    explanation_for_user: str
    ml_proba: dict


@app.post("/analyze", response_model=AnalyzeResponse)
def analyze(req: AnalyzeRequest):
    result = guardian_analyzer.analyze(req.message)
    return AnalyzeResponse(**result)
