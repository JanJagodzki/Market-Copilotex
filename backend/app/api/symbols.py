from fastapi import APIRouter, Depends, HTTPException, Query

from backend.app.data.yahoo_intraday import sync_intraday_prices
from backend.app.db.database import SessionLocal
from backend.app.db.models import (
    DailyFeature,
    DailyPrice,
    IntradayPrice,
    Symbol,
)
from backend.app.ml.prediction_service import (
    PredictionError,
    predict_for_rows,
)


router = APIRouter(prefix="/api")


DAILY_PERIOD_LIMITS = {
    "1m": 23,
    "3m": 66,
    "1y": 252,
    "5y": 1260,
    "max": None,
}


def get_db():
    db = SessionLocal()

    try:
        yield db
    finally:
        db.close()


def find_symbol(db, ticker):
    return (
        db.query(Symbol)
        .filter(
            Symbol.ticker == ticker.upper(),
            Symbol.active.is_(True),
        )
        .first()
    )


@router.get("/symbols")
def search_symbols(
    search: str = "",
    limit: int = Query(default=20, ge=1, le=100),
    db=Depends(get_db),
):
    query = db.query(Symbol).filter(
        Symbol.active.is_(True)
    )

    if search:
        value = f"%{search.strip()}%"

        query = query.filter(
            Symbol.ticker.ilike(value)
            | Symbol.name.ilike(value)
        )

    symbols = (
        query
        .order_by(Symbol.ticker)
        .limit(limit)
        .all()
    )

    return [
        {
            "ticker": symbol.ticker,
            "name": symbol.name,
            "exchange": symbol.primary_exchange,
        }
        for symbol in symbols
    ]


@router.post("/symbols/{ticker}/prices/sync")
def sync_symbol_prices(
    ticker: str,
    days: int = Query(default=5, ge=1, le=59),
):
    ticker = ticker.upper()

    result = sync_intraday_prices(
        tickers=[ticker],
        days=days,
        batch_size=1,
    )

    if result["missing_symbols"] > 0:
        raise HTTPException(
            status_code=404,
            detail="Symbol not found",
        )

    if result["failed_symbols"] > 0:
        raise HTTPException(
            status_code=502,
            detail="Yahoo did not return intraday data",
        )

    return result


@router.get("/symbols/{ticker}/prices")
def get_symbol_prices(
    ticker: str,
    interval: str = "15m",
    limit: int = Query(default=200, ge=1, le=2000),
    db=Depends(get_db),
):
    if interval != "15m":
        raise HTTPException(
            status_code=400,
            detail="Only the 15m interval is available",
        )

    symbol = find_symbol(db, ticker)

    if symbol is None:
        raise HTTPException(
            status_code=404,
            detail="Symbol not found",
        )

    prices = (
        db.query(IntradayPrice)
        .filter(
            IntradayPrice.symbol_id == symbol.id,
            IntradayPrice.interval == interval,
        )
        .order_by(IntradayPrice.timestamp.desc())
        .limit(limit)
        .all()
    )

    prices.reverse()

    return {
        "ticker": symbol.ticker,
        "name": symbol.name,
        "interval": interval,
        "count": len(prices),
        "prices": [
            {
                "timestamp": price.timestamp.isoformat(),
                "open": price.open,
                "high": price.high,
                "low": price.low,
                "close": price.close,
                "volume": price.volume,
            }
            for price in prices
        ],
    }


@router.get("/symbols/{ticker}/daily-prices")
def get_daily_prices(
    ticker: str,
    period: str = "1y",
    db=Depends(get_db),
):
    period = period.lower()

    if period not in DAILY_PERIOD_LIMITS:
        raise HTTPException(
            status_code=400,
            detail="Unknown daily period",
        )

    symbol = find_symbol(db, ticker)

    if symbol is None:
        raise HTTPException(
            status_code=404,
            detail="Symbol not found",
        )

    query = (
        db.query(DailyPrice)
        .filter(DailyPrice.symbol_id == symbol.id)
        .order_by(DailyPrice.date.desc())
    )

    limit = DAILY_PERIOD_LIMITS[period]

    if limit is not None:
        query = query.limit(limit)

    prices = query.all()
    prices.reverse()

    return {
        "ticker": symbol.ticker,
        "name": symbol.name,
        "interval": "1d",
        "period": period,
        "count": len(prices),
        "prices": [
            {
                "timestamp": price.date.isoformat(),
                "open": price.open,
                "high": price.high,
                "low": price.low,
                "close": price.close,
                "volume": price.volume,
            }
            for price in prices
        ],
    }


@router.get(
    "/symbols/{ticker}/ai-predictions"
)
def get_ai_predictions(
    ticker: str,
    db=Depends(get_db),
):
    symbol = find_symbol(db, ticker)

    if symbol is None:
        raise HTTPException(
            status_code=404,
            detail="Symbol not found",
        )

    feature_rows = (
        db.query(DailyFeature)
        .filter(
            DailyFeature.symbol_id
            == symbol.id
        )
        .order_by(DailyFeature.date.desc())
        .limit(60)
        .all()
    )

    if len(feature_rows) < 60:
        raise HTTPException(
            status_code=422,
            detail=(
                "At least 60 daily feature rows "
                "are required"
            ),
        )

    feature_rows.reverse()
    data_date = feature_rows[-1].date

    price = (
        db.query(DailyPrice)
        .filter(
            DailyPrice.symbol_id
            == symbol.id,
            DailyPrice.date == data_date,
        )
        .first()
    )

    if price is None:
        raise HTTPException(
            status_code=422,
            detail=(
                "Reference price is missing "
                "for the latest feature date"
            ),
        )

    try:
        predictions = predict_for_rows(
            feature_rows
        )
    except PredictionError as error:
        raise HTTPException(
            status_code=503,
            detail=str(error),
        ) from error

    return {
        "ticker": symbol.ticker,
        "name": symbol.name,
        "data_date": data_date.isoformat(),
        "reference_price": price.close,
        "predictions": predictions,
        "warning": (
            "Experimental model output, not "
            "investment advice. Short horizons "
            "have weak evaluation results."
        ),
    }
