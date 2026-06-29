from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import func, or_, select, text
from sqlalchemy.orm import Session

from app.api.deps import get_db
from app.models.asset import Asset
from app.models.journal_entry import JournalEntry
from app.models.market_price_daily import MarketPriceDaily


router = APIRouter(prefix="/companies", tags=["companies"])


def format_model_display_name(model_name: str) -> str:
    if "lightgbm" in model_name:
        return "LightGBM"
    if "xgboost" in model_name:
        return "XGBoost"
    if "catboost" in model_name:
        return "CatBoost"
    if "extra_trees" in model_name:
        return "Extra Trees"
    if "hist_gradient" in model_name:
        return "HistGradientBoosting"

    return model_name


def get_asset_by_symbol(db: Session, symbol: str) -> Asset:
    asset = db.execute(
        select(Asset).where(func.upper(Asset.symbol) == symbol.upper())
    ).scalar_one_or_none()

    if asset is None:
        raise HTTPException(status_code=404, detail=f"Company {symbol.upper()} not found")

    return asset


def serialize_journal_entry(entry: JournalEntry, asset: Asset) -> dict:
    return {
        "id": entry.id,
        "asset_id": entry.asset_id,
        "symbol": asset.symbol,
        "name": asset.name,
        "horizon_days": entry.horizon_days,
        "decision": entry.decision,
        "status": entry.status,
        "title": entry.title,
        "thesis": entry.thesis,
        "plan": entry.plan,
        "notes": entry.notes,
        "entry_price": entry.entry_price,
        "stop_loss": entry.stop_loss,
        "take_profit": entry.take_profit,
        "position_size": entry.position_size,
        "emotion": entry.emotion,
        "confidence": entry.confidence,
        "tags": entry.tags,
        "created_at": entry.created_at,
        "updated_at": entry.updated_at,
    }


@router.get("/search")
def search_companies(
    q: str = Query(..., min_length=1),
    limit: int = Query(default=20, ge=1, le=100),
    db: Session = Depends(get_db),
) -> list[dict]:
    pattern = f"%{q.upper()}%"

    query = (
        select(Asset)
        .where(Asset.is_active.is_(True))
        .where(
            or_(
                func.upper(Asset.symbol).like(pattern),
                func.upper(Asset.name).like(pattern),
                func.upper(Asset.sector).like(pattern),
                func.upper(Asset.industry).like(pattern),
            )
        )
        .order_by(Asset.universe_rank.asc().nulls_last(), Asset.symbol.asc())
        .limit(limit)
    )

    assets = db.execute(query).scalars().all()

    return [
        {
            "id": asset.id,
            "symbol": asset.symbol,
            "name": asset.name,
            "exchange": asset.exchange,
            "currency": asset.currency,
            "sector": asset.sector,
            "industry": asset.industry,
            "country": asset.country,
            "market_cap": asset.market_cap,
            "universe_name": asset.universe_name,
            "universe_rank": asset.universe_rank,
        }
        for asset in assets
    ]


@router.get("/{symbol}/overview")
def get_company_overview(
    symbol: str,
    db: Session = Depends(get_db),
) -> dict:
    asset = get_asset_by_symbol(db, symbol)

    latest_price = db.execute(
        select(MarketPriceDaily)
        .where(MarketPriceDaily.asset_id == asset.id)
        .where(
            or_(
                MarketPriceDaily.close.is_not(None),
                MarketPriceDaily.adjusted_close.is_not(None),
            )
        )
        .order_by(MarketPriceDaily.date.desc())
        .limit(1)
    ).scalar_one_or_none()

    return {
        "id": asset.id,
        "symbol": asset.symbol,
        "name": asset.name,
        "exchange": asset.exchange,
        "currency": asset.currency,
        "sector": asset.sector,
        "industry": asset.industry,
        "country": asset.country,
        "market_cap": asset.market_cap,
        "universe_name": asset.universe_name,
        "universe_rank": asset.universe_rank,
        "latest_price": None
        if latest_price is None
        else {
            "date": latest_price.date,
            "open": latest_price.open,
            "high": latest_price.high,
            "low": latest_price.low,
            "close": latest_price.close,
            "adjusted_close": latest_price.adjusted_close,
            "volume": latest_price.volume,
        },
    }


@router.get("/{symbol}/predictions")
def get_company_predictions(
    symbol: str,
    db: Session = Depends(get_db),
) -> list[dict]:
    asset = get_asset_by_symbol(db, symbol)

    query = text(
        """
        SELECT DISTINCT ON (mp.horizon_days)
            mp.horizon_days,
            mp.date AS prediction_date,
            mp.model_name,
            mp.predicted_return,
            mp.prediction_score,
            mp.risk_score,
            mp.final_score,
            mp.prediction_rank,
            mp.created_at,
            mp.updated_at
        FROM model_predictions_daily mp
        WHERE mp.asset_id = :asset_id
        ORDER BY mp.horizon_days ASC, mp.date DESC, mp.created_at DESC
        """
    )

    rows = db.execute(query, {"asset_id": asset.id}).mappings().all()

    return [
        {
            "symbol": asset.symbol,
            "horizon_days": row["horizon_days"],
            "horizon_label": f"{row['horizon_days']}D",
            "prediction_date": row["prediction_date"],
            "model_name": row["model_name"],
            "model_display_name": format_model_display_name(row["model_name"]),
            "predicted_return": row["predicted_return"],
            "predicted_return_percent": None
            if row["predicted_return"] is None
            else row["predicted_return"] * 100.0,
            "prediction_score": row["prediction_score"],
            "risk_score": row["risk_score"],
            "final_score": row["final_score"],
            "prediction_rank": row["prediction_rank"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
        }
        for row in rows
    ]


@router.get("/{symbol}/journal")
def get_company_journal_entries(
    symbol: str,
    status: str | None = Query(default=None),
    limit: int = Query(default=20, ge=1, le=100),
    db: Session = Depends(get_db),
) -> list[dict]:
    asset = get_asset_by_symbol(db, symbol)

    query = (
        select(JournalEntry)
        .where(JournalEntry.asset_id == asset.id)
        .order_by(JournalEntry.created_at.desc())
        .limit(limit)
    )

    if status is not None:
        query = query.where(JournalEntry.status == status.lower())

    entries = db.execute(query).scalars().all()

    return [
        serialize_journal_entry(entry=entry, asset=asset)
        for entry in entries
    ]


@router.get("/{symbol}/analysis")
def get_company_analysis(
    symbol: str,
    db: Session = Depends(get_db),
) -> dict:
    overview = get_company_overview(symbol=symbol, db=db)
    predictions = get_company_predictions(symbol=symbol, db=db)
    journal_entries = get_company_journal_entries(symbol=symbol, status=None, limit=20, db=db)

    return {
        "overview": overview,
        "predictions": predictions,
        "journal": {
            "entries": journal_entries,
            "count": len(journal_entries),
        },
        "news_ai": {
            "status": "not_implemented_yet",
            "message": "News AI analysis will be added later.",
        },
    }
