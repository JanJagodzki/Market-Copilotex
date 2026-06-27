from fastapi import APIRouter, Depends, Query
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.api.deps import get_db
from app.models.asset import Asset
from app.models.model_prediction_daily import ModelPredictionDaily


router = APIRouter(prefix="/dashboard", tags=["dashboard"])


@router.get("/top-opportunities")
def get_top_opportunities(
    db: Session = Depends(get_db),
    horizon_days: int = Query(default=30, ge=1, le=365),
    limit: int = Query(default=20, ge=1, le=100),
) -> list[dict]:
    latest_date_query = (
        select(ModelPredictionDaily.date)
        .where(ModelPredictionDaily.horizon_days == horizon_days)
        .order_by(ModelPredictionDaily.date.desc())
        .limit(1)
    )

    latest_date = db.execute(latest_date_query).scalar_one_or_none()

    if latest_date is None:
        return []

    query = (
        select(ModelPredictionDaily, Asset)
        .join(Asset, Asset.id == ModelPredictionDaily.asset_id)
        .where(ModelPredictionDaily.date == latest_date)
        .where(ModelPredictionDaily.horizon_days == horizon_days)
        .order_by(ModelPredictionDaily.prediction_rank.asc())
        .limit(limit)
    )

    rows = db.execute(query).all()

    return [
        {
            "prediction_rank": prediction.prediction_rank,
            "symbol": asset.symbol,
            "name": asset.name,
            "sector": asset.sector,
            "industry": asset.industry,
            "prediction_date": prediction.date,
            "horizon_days": prediction.horizon_days,
            "model_name": prediction.model_name,
            "predicted_return": prediction.predicted_return,
            "prediction_score": prediction.prediction_score,
            "risk_score": prediction.risk_score,
            "final_score": prediction.final_score,
            "created_at": prediction.created_at,
        }
        for prediction, asset in rows
    ]