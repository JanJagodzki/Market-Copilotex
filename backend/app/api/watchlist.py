from fastapi import APIRouter, Depends, HTTPException

from backend.app.db.database import SessionLocal
from backend.app.db.models import Symbol, WatchlistItem
from backend.app.services.watchlist_sync import (
    get_watchlist_sync_status,
)


router = APIRouter(
    prefix="/api/watchlist"
)


def get_db():
    db = SessionLocal()

    try:
        yield db
    finally:
        db.close()


def find_active_symbol(db, ticker):
    return (
        db.query(Symbol)
        .filter(
            Symbol.ticker == ticker.upper(),
            Symbol.active.is_(True),
        )
        .first()
    )


def symbol_response(symbol):
    return {
        "ticker": symbol.ticker,
        "name": symbol.name,
        "exchange": symbol.primary_exchange,
    }


@router.get("")
def get_watchlist(
    db=Depends(get_db),
):
    symbols = (
        db.query(Symbol)
        .join(
            WatchlistItem,
            WatchlistItem.symbol_id
            == Symbol.id,
        )
        .order_by(Symbol.ticker)
        .all()
    )

    return [
        symbol_response(symbol)
        for symbol in symbols
    ]


@router.post("/{ticker}")
def add_to_watchlist(
    ticker: str,
    db=Depends(get_db),
):
    symbol = find_active_symbol(
        db,
        ticker,
    )

    if symbol is None:
        raise HTTPException(
            status_code=404,
            detail="Symbol not found",
        )

    existing = (
        db.query(WatchlistItem)
        .filter(
            WatchlistItem.symbol_id
            == symbol.id
        )
        .first()
    )

    if existing is None:
        db.add(
            WatchlistItem(
                symbol_id=symbol.id
            )
        )
        db.commit()

    return symbol_response(symbol)


@router.delete("/{ticker}")
def remove_from_watchlist(
    ticker: str,
    db=Depends(get_db),
):
    symbol = find_active_symbol(
        db,
        ticker,
    )

    if symbol is None:
        raise HTTPException(
            status_code=404,
            detail="Symbol not found",
        )

    item = (
        db.query(WatchlistItem)
        .filter(
            WatchlistItem.symbol_id
            == symbol.id
        )
        .first()
    )

    if item is not None:
        db.delete(item)
        db.commit()

    return {
        "ticker": symbol.ticker,
        "removed": item is not None,
    }


@router.get("/status/current")
def watchlist_sync_status():
    return get_watchlist_sync_status()
