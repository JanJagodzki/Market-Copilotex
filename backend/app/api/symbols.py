from fastapi import APIRouter, Depends, HTTPException, Query

from backend.app.data.yahoo_intraday import sync_intraday_prices
from backend.app.db.database import SessionLocal
from backend.app.db.models import IntradayPrice, Symbol


router = APIRouter(prefix="/api")


def get_db():
    db = SessionLocal()

    try:
        yield db
    finally:
        db.close()


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

    ticker = ticker.upper()

    symbol = (
        db.query(Symbol)
        .filter(
            Symbol.ticker == ticker,
            Symbol.active.is_(True),
        )
        .first()
    )

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
