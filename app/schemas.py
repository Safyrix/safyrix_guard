from pydantic import BaseModel

class AnalyzeRequest(BaseModel):
    message: str
    user_id: str

class AnalyzeResponse(BaseModel):
    risk_level: str
    categories: list
    confidence: int
    red_flags: list
    explanation_for_user: str
    recommended_actions: list
