from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from backend.app.db.database import SessionLocal
from backend.app.db.models import (
    DailyPrice,
    IntradayPrice,
    PaperTrade,
    Symbol,
)


router = APIRouter(prefix="/api/paper-trades")


class TradeInput(BaseModel):
    side: str
    quantity: float = Field(gt=0)
    price: float = Field(gt=0)
    note: str | None = Field(
        default=None,
        max_length=500,
    )


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


def calculate_position(trades, current_price=None):
    shares = 0.0
    average_price = 0.0
    realized_profit_loss = 0.0

    for trade in trades:
        if trade.side == "BUY":
            old_value = shares * average_price
            new_value = trade.quantity * trade.price
            shares += trade.quantity
            average_price = (
                old_value + new_value
            ) / shares
        elif trade.side == "SELL":
            if trade.quantity > shares + 0.000001:
                raise ValueError(
                    "A sell trade cannot exceed owned shares"
                )

            realized_profit_loss += (
                trade.price - average_price
            ) * trade.quantity
            shares -= trade.quantity

            if shares < 0.000001:
                shares = 0.0
                average_price = 0.0

    market_value = None
    unrealized_profit_loss = None

    if current_price is not None:
        market_value = shares * current_price
        unrealized_profit_loss = (
            current_price - average_price
        ) * shares

    total_profit_loss = realized_profit_loss

    if unrealized_profit_loss is not None:
        total_profit_loss += unrealized_profit_loss

    return {
        "shares": round(shares, 6),
        "average_buy_price": (
            round(average_price, 6)
            if shares > 0
            else None
        ),
        "market_value": (
            round(market_value, 2)
            if market_value is not None
            else None
        ),
        "realized_profit_loss": round(
            realized_profit_loss,
            2,
        ),
        "unrealized_profit_loss": (
            round(unrealized_profit_loss, 2)
            if unrealized_profit_loss is not None
            else None
        ),
        "total_profit_loss": round(
            total_profit_loss,
            2,
        ),
    }


def get_latest_price(db, symbol_id):
    intraday_price = (
        db.query(IntradayPrice)
        .filter(
            IntradayPrice.symbol_id == symbol_id
        )
        .order_by(
            IntradayPrice.timestamp.desc()
        )
        .first()
    )

    if intraday_price is not None:
        return intraday_price.close

    daily_price = (
        db.query(DailyPrice)
        .filter(
            DailyPrice.symbol_id == symbol_id
        )
        .order_by(DailyPrice.date.desc())
        .first()
    )

    if daily_price is not None:
        return daily_price.close

    return None


def get_symbol_trades(db, symbol_id):
    return (
        db.query(PaperTrade)
        .filter(
            PaperTrade.symbol_id == symbol_id
        )
        .order_by(
            PaperTrade.traded_at,
            PaperTrade.id,
        )
        .all()
    )


def trade_response(trade):
    return {
        "id": trade.id,
        "side": trade.side,
        "quantity": trade.quantity,
        "price": trade.price,
        "note": trade.note,
        "traded_at": trade.traded_at.isoformat(),
    }


@router.get("/{ticker}")
def get_paper_trades(
    ticker: str,
    db=Depends(get_db),
):
    symbol = find_symbol(db, ticker)

    if symbol is None:
        raise HTTPException(
            status_code=404,
            detail="Symbol not found",
        )

    trades = get_symbol_trades(db, symbol.id)
    current_price = get_latest_price(
        db,
        symbol.id,
    )

    return {
        "ticker": symbol.ticker,
        "name": symbol.name,
        "current_price": current_price,
        "summary": calculate_position(
            trades,
            current_price,
        ),
        "trades": [
            trade_response(trade)
            for trade in reversed(trades)
        ],
    }


@router.post("/{ticker}")
def add_paper_trade(
    ticker: str,
    trade_input: TradeInput,
    db=Depends(get_db),
):
    symbol = find_symbol(db, ticker)

    if symbol is None:
        raise HTTPException(
            status_code=404,
            detail="Symbol not found",
        )

    side = trade_input.side.upper()

    if side not in {"BUY", "SELL"}:
        raise HTTPException(
            status_code=400,
            detail="Side must be BUY or SELL",
        )

    trades = get_symbol_trades(db, symbol.id)
    position = calculate_position(trades)

    if (
        side == "SELL"
        and trade_input.quantity
        > position["shares"] + 0.000001
    ):
        raise HTTPException(
            status_code=400,
            detail=(
                "You cannot sell more shares "
                "than you own"
            ),
        )

    note = trade_input.note

    if note is not None:
        note = note.strip() or None

    trade = PaperTrade(
        symbol_id=symbol.id,
        side=side,
        quantity=trade_input.quantity,
        price=trade_input.price,
        note=note,
    )

    db.add(trade)
    db.commit()

    return get_paper_trades(
        ticker=symbol.ticker,
        db=db,
    )


@router.delete("/trade/{trade_id}")
def delete_paper_trade(
    trade_id: int,
    db=Depends(get_db),
):
    trade = (
        db.query(PaperTrade)
        .filter(PaperTrade.id == trade_id)
        .first()
    )

    if trade is None:
        raise HTTPException(
            status_code=404,
            detail="Trade not found",
        )

    other_trades = [
        item
        for item in get_symbol_trades(
            db,
            trade.symbol_id,
        )
        if item.id != trade.id
    ]

    try:
        calculate_position(other_trades)
    except ValueError as error:
        raise HTTPException(
            status_code=400,
            detail=(
                "Delete later sell trades first"
            ),
        ) from error

    db.delete(trade)
    db.commit()

    return {
        "trade_id": trade_id,
        "deleted": True,
    }
