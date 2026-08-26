import time
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import pandas as pd
import yfinance as yf

from sqlalchemy.dialects.postgresql import insert

from backend.app.data.yahoo_prices import get_symbol_data
from backend.app.db.database import SessionLocal
from backend.app.db.models import IntradayPrice, Symbol


INTERVAL = "15m"
INTERVAL_MINUTES = 15


def download_intraday(tickers, days=5):
    today = datetime.now(ZoneInfo("America/New_York")).date()
    start = today - timedelta(days=days)
    end = today + timedelta(days=1)

    return yf.download(
        tickers=tickers,
        start=start,
        end=end,
        interval=INTERVAL,
        group_by="ticker",
        auto_adjust=False,
        actions=False,
        progress=False,
        threads=True,
    )


def to_utc(timestamp):
    value = pd.Timestamp(timestamp)

    if value.tzinfo is None:
        value = value.tz_localize("America/New_York")

    return value.tz_convert("UTC")


def save_intraday_prices(db, symbol, data, current_time=None):
    if data.empty:
        return 0

    data = data.dropna(subset=["Open", "High", "Low", "Close"])
    if current_time is None:
        current_time = pd.Timestamp.now(tz="UTC")
    else:
        current_time = to_utc(current_time)

    interval_length = pd.Timedelta(minutes=INTERVAL_MINUTES)
    rows = []

    for timestamp, row in data.iterrows():
        candle_time = to_utc(timestamp)

        if candle_time + interval_length > current_time:
            continue

        volume = row.get("Volume")
        rows.append(
            {
                "symbol_id": symbol.id,
                "timestamp": candle_time.to_pydatetime(),
                "interval": INTERVAL,
                "open": float(row["Open"]),
                "high": float(row["High"]),
                "low": float(row["Low"]),
                "close": float(row["Close"]),
                "volume": None if pd.isna(volume) else int(volume),
                "source": "yfinance",
            }
        )

    if not rows:
        return 0

    statement = insert(IntradayPrice).values(rows)
    statement = statement.on_conflict_do_update(
        constraint="uq_intraday_prices_symbol_time_interval",
        set_={
            "open": statement.excluded.open,
            "high": statement.excluded.high,
            "low": statement.excluded.low,
            "close": statement.excluded.close,
            "volume": statement.excluded.volume,
            "source": statement.excluded.source,
        },
    )
    db.execute(statement)
    return len(rows)


def download_with_retry(tickers, days):
    for attempt in range(1, 4):
        try:
            return download_intraday(tickers, days=days)
        except Exception as error:
            print(f"Download attempt {attempt}/3 failed: {error}")
            if attempt < 3:
                time.sleep(5 * attempt)

    return None


def sync_intraday_prices(tickers, days=5, batch_size=20):
    if not tickers:
        raise ValueError("At least one ticker is required")

    if batch_size < 1:
        raise ValueError("batch_size must be positive")

    if days < 1 or days > 59:
        raise ValueError("days must be between 1 and 59")

    requested = list(dict.fromkeys(ticker.upper() for ticker in tickers))
    db = SessionLocal()
    result = {
        "requested_symbols": len(requested),
        "updated_symbols": 0,
        "rows": 0,
        "failed_symbols": 0,
        "missing_symbols": 0,
    }

    try:
        symbols = (
            db.query(Symbol)
            .filter(Symbol.active.is_(True), Symbol.ticker.in_(requested))
            .all()
        )
        symbols_by_ticker = {symbol.ticker: symbol for symbol in symbols}

        for ticker in requested:
            if ticker not in symbols_by_ticker:
                print(f"{ticker}: symbol not found")
                result["missing_symbols"] += 1

        ordered_symbols = [
            symbols_by_ticker[ticker]
            for ticker in requested
            if ticker in symbols_by_ticker
        ]

        for start in range(0, len(ordered_symbols), batch_size):
            batch = ordered_symbols[start:start + batch_size]
            yahoo_tickers = [symbol.ticker.replace(".", "-") for symbol in batch]

            print()
            print(f"Downloading: {', '.join(yahoo_tickers)}")
            data = download_with_retry(yahoo_tickers, days)

            if data is None:
                result["failed_symbols"] += len(batch)
                continue

            for symbol in batch:
                yahoo_ticker = symbol.ticker.replace(".", "-")
                symbol_data = get_symbol_data(data, yahoo_ticker)

                if symbol_data.empty:
                    print(f"{symbol.ticker}: no intraday data")
                    result["failed_symbols"] += 1
                    continue

                try:
                    count = save_intraday_prices(db, symbol, symbol_data)
                    db.commit()
                    result["rows"] += count
                    result["updated_symbols"] += 1
                    print(f"{symbol.ticker}: {count} rows")
                except Exception as error:
                    db.rollback()
                    result["failed_symbols"] += 1
                    print(f"{symbol.ticker}: database error: {error}")

        return result
    finally:
        db.close()
