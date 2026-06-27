from datetime import date, datetime
from pydantic import BaseModel


class TopOpportunityRead(BaseModel):
    prediction_rank: int
    symbol: str
    name: str | None
    sector: str | None
    industry: str | None
    prediction_date: date
    horizon_days: int
    model_name: str
    predicted_return: float
    prediction_score: float
    risk_score: float
    final_score: float
    created_at: datetime

    model_config = {
        "from_attributes": True
    }