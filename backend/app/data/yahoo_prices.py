import time
from datetime import datetime, timedelta, time as datetime_time
from zoneinfo import ZoneInfo

import pandas as pd
import yfinance as yf

from sqlalchemy import func
from sqlalchemy.dialects.postgresql import insert

from backend.app.db.database import SessionLocal
from backend.app.db.models import DailyPrice, Symbol


def download_prices(
    tickers,
    period=None,
    start=None,
    end=None,
):
    options = {
        "tickers": tickers,
        "interval": "1d",
        "group_by": "ticker",
        "auto_adjust": False,
        "actions": False,
        "progress": False,
        "threads": True,
    }

    if start is not None:
        options["start"] = start
        options["end"] = end
    else:
        options["period"] = period or "1mo"

    return yf.download(**options)


def normalize_columns(data):
    if not isinstance(data.columns, pd.MultiIndex):
        return data

    price_columns = {
        "Open",
        "High",
        "Low",
        "Close",
        "Adj Close",
        "Volume",
    }

    for level in range(data.columns.nlevels):
        values = data.columns.get_level_values(level)

        if price_columns.intersection(values):
            result = data.copy()
            result.columns = values
            return result

    return data


def get_symbol_data(data, ticker):
    if data.empty:
        return pd.DataFrame()

    if isinstance(data.columns, pd.MultiIndex):
        if ticker in data.columns.get_level_values(0):
            return normalize_columns(
                data[ticker].copy()
            )

        if ticker in data.columns.get_level_values(1):
            return normalize_columns(
                data.xs(
                    ticker,
                    axis=1,
                    level=1,
                ).copy()
            )

        return pd.DataFrame()

    return normalize_columns(
        data.copy()
    )


def save_symbol_prices(db, symbol, data):
    if data.empty:
        return 0

    data = data.dropna(
        subset=["Open", "High", "Low", "Close"]
    )

    rows = []

    for timestamp, row in data.iterrows():
        adjusted_close = row.get(
            "Adj Close",
            row["Close"],
        )

        volume = row.get("Volume")

        rows.append(
            {
                "symbol_id": symbol.id,
                "date": pd.Timestamp(timestamp).date(),
                "open": float(row["Open"]),
                "high": float(row["High"]),
                "low": float(row["Low"]),
                "close": float(row["Close"]),
                "adjusted_close": (
                    None
                    if pd.isna(adjusted_close)
                    else float(adjusted_close)
                ),
                "volume": (
                    None
                    if pd.isna(volume)
                    else int(volume)
                ),
                "source": "yfinance",
            }
        )

    if not rows:
        return 0

    batch_size = 1000

    for start in range(0, len(rows), batch_size):
        batch = rows[start:start + batch_size]

        statement = insert(DailyPrice).values(batch)

        statement = statement.on_conflict_do_update(
            constraint="uq_daily_prices_symbol_date",
            set_={
                "open": statement.excluded.open,
                "high": statement.excluded.high,
                "low": statement.excluded.low,
                "close": statement.excluded.close,
                "adjusted_close": statement.excluded.adjusted_close,
                "volume": statement.excluded.volume,
                "source": statement.excluded.source,
            },
        )

        db.execute(statement)

    return len(rows)


def get_completed_daily_end():
    now = datetime.now(
        ZoneInfo("America/New_York")
    )

    market_is_closed = (
        now.weekday() < 5
        and now.time() >= datetime_time(17, 0)
    )

    if market_is_closed:
        return now.date() + timedelta(days=1)

    return now.date()


def get_latest_price_dates(db):
    rows = (
        db.query(
            DailyPrice.symbol_id,
            func.max(DailyPrice.date),
        )
        .group_by(DailyPrice.symbol_id)
        .all()
    )

    return {
        symbol_id: last_date
        for symbol_id, last_date in rows
    }


def download_batch(
    tickers,
    period=None,
    start=None,
    end=None,
):
    for attempt in range(1, 4):
        try:
            return download_prices(
                tickers=tickers,
                period=period,
                start=start,
                end=end,
            )

        except Exception as error:
            print(
                f"Download attempt "
                f"{attempt}/3 failed: {error}"
            )

            if attempt < 3:
                time.sleep(5 * attempt)

    return None


def save_batch(db, symbols, data, empty_is_error=True):
    result = {
        "rows": 0,
        "updated_symbols": 0,
        "failed_symbols": 0,
        "skipped_symbols": 0,
    }

    if data is None:
        result["failed_symbols"] = len(symbols)
        return result

    for symbol in symbols:
        yahoo_ticker = symbol.ticker.replace(
            ".",
            "-",
        )

        symbol_data = get_symbol_data(
            data,
            yahoo_ticker,
        )

        if symbol_data.empty:
            print(f"{symbol.ticker}: no data")

            if empty_is_error:
                result["failed_symbols"] += 1
            else:
                result["skipped_symbols"] += 1

            continue

        try:
            count = save_symbol_prices(
                db,
                symbol,
                symbol_data,
            )

            db.commit()

            result["rows"] += count
            result["updated_symbols"] += 1

            print(f"{symbol.ticker}: {count} rows")

        except Exception as error:
            db.rollback()
            result["failed_symbols"] += 1

            print(
                f"{symbol.ticker}: "
                f"database error: {error}"
            )

    return result


def add_result(total, batch_result):
    for key in (
        "rows",
        "updated_symbols",
        "failed_symbols",
        "skipped_symbols",
    ):
        total[key] += batch_result[key]


def sync_missing_daily_prices(
    overlap_days=5,
    batch_size=25,
    limit=None,
):
    if overlap_days < 0:
        raise ValueError("overlap_days cannot be negative")

    if batch_size < 1:
        raise ValueError("batch_size must be positive")

    db = SessionLocal()

    result = {
        "symbols": 0,
        "rows": 0,
        "updated_symbols": 0,
        "failed_symbols": 0,
        "skipped_symbols": 0,
    }

    try:
        symbols = (
            db.query(Symbol)
            .filter(Symbol.active.is_(True))
            .order_by(Symbol.ticker)
            .all()
        )

        if limit is not None:
            symbols = symbols[:limit]

        result["symbols"] = len(symbols)

        latest_dates = get_latest_price_dates(db)

        existing_symbols = [
            symbol
            for symbol in symbols
            if symbol.id in latest_dates
        ]

        new_symbols = [
            symbol
            for symbol in symbols
            if symbol.id not in latest_dates
        ]

        existing_symbols.sort(
            key=lambda symbol: latest_dates[symbol.id],
            reverse=True,
        )

        download_end = get_completed_daily_end()

        print(f"Active symbols: {len(symbols)}")
        print(f"Existing symbols: {len(existing_symbols)}")
        print(f"New symbols: {len(new_symbols)}")
        print(f"Download end: {download_end}")

        for start_index in range(
            0,
            len(existing_symbols),
            batch_size,
        ):
            batch = existing_symbols[
                start_index:start_index + batch_size
            ]

            batch_start = min(
                latest_dates[symbol.id]
                for symbol in batch
            ) - timedelta(days=overlap_days)

            tickers = [
                symbol.ticker.replace(".", "-")
                for symbol in batch
            ]

            print()
            print(
                f"Catch-up {start_index + 1}-"
                f"{start_index + len(batch)} "
                f"from {batch_start}"
            )

            data = download_batch(
                tickers=tickers,
                start=batch_start.isoformat(),
                end=download_end.isoformat(),
            )

            batch_result = save_batch(
                db,
                batch,
                data,
            )

            add_result(result, batch_result)
            time.sleep(1)

        for start_index in range(
            0,
            len(new_symbols),
            batch_size,
        ):
            batch = new_symbols[
                start_index:start_index + batch_size
            ]

            tickers = [
                symbol.ticker.replace(".", "-")
                for symbol in batch
            ]

            print()
            print(
                f"Full history for new symbols "
                f"{start_index + 1}-"
                f"{start_index + len(batch)}"
            )

            data = download_batch(
                tickers=tickers,
                period="max",
            )

            batch_result = save_batch(
                db,
                batch,
                data,
                empty_is_error=False,
            )

            add_result(result, batch_result)
            time.sleep(1)

        return result

    finally:
        db.close()


def sync_selected_symbols(tickers, period="max"):
    db = SessionLocal()

    try:
        symbols = (
            db.query(Symbol)
            .filter(Symbol.ticker.in_(tickers))
            .all()
        )

        found = {symbol.ticker for symbol in symbols}

        for ticker in tickers:
            if ticker not in found:
                print(f"Symbol not found in database: {ticker}")

        yahoo_tickers = [
            symbol.ticker.replace(".", "-")
            for symbol in symbols
        ]

        print(
            f"Downloading: {', '.join(yahoo_tickers)}"
        )

        data = download_prices(
            yahoo_tickers,
            period=period,
        )

        for symbol in symbols:
            yahoo_ticker = symbol.ticker.replace(
                ".",
                "-",
            )

            symbol_data = get_symbol_data(
                data,
                yahoo_ticker,
            )

            if symbol_data.empty:
                print(f"No data: {symbol.ticker}")
                continue

            count = save_symbol_prices(
                db,
                symbol,
                symbol_data,
            )

            print(
                f"{symbol.ticker}: {count} rows"
            )

        db.commit()

    except Exception:
        db.rollback()
        raise

    finally:
        db.close()


def sync_all_symbols(
    period="1mo",
    only_without_prices=False,
    batch_size=10,
    limit=None,
):
    db = SessionLocal()

    try:
        symbols = (
            db.query(Symbol)
            .filter(Symbol.active.is_(True))
            .order_by(Symbol.ticker)
            .all()
        )

        if only_without_prices:
            existing_ids = {
                row[0]
                for row in db.query(
                    DailyPrice.symbol_id
                ).distinct()
            }

            symbols = [
                symbol
                for symbol in symbols
                if symbol.id not in existing_ids
            ]

        if limit is not None:
            symbols = symbols[:limit]

        total_symbols = len(symbols)

        print(f"Symbols to download: {total_symbols}")

        for start in range(
            0,
            total_symbols,
            batch_size,
        ):
            batch = symbols[start:start + batch_size]

            yahoo_tickers = [
                symbol.ticker.replace(".", "-")
                for symbol in batch
            ]

            batch_number = start // batch_size + 1
            total_batches = (
                total_symbols + batch_size - 1
            ) // batch_size

            print()
            print(
                f"Batch {batch_number}/{total_batches}"
            )
            print(", ".join(yahoo_tickers))

            data = None

            for attempt in range(1, 4):
                try:
                    data = download_prices(
                        yahoo_tickers,
                        period=period,
                    )
                    break

                except Exception as error:
                    print(
                        f"Download attempt "
                        f"{attempt}/3 failed: {error}"
                    )

                    time.sleep(10 * attempt)

            if data is None:
                print("Skipping batch")
                continue

            for symbol in batch:
                yahoo_ticker = symbol.ticker.replace(
                    ".",
                    "-",
                )

                symbol_data = get_symbol_data(
                    data,
                    yahoo_ticker,
                )

                if symbol_data.empty:
                    print(
                        f"{symbol.ticker}: no data"
                    )
                    continue

                try:
                    count = save_symbol_prices(
                        db,
                        symbol,
                        symbol_data,
                    )

                    db.commit()

                    print(
                        f"{symbol.ticker}: "
                        f"{count} rows"
                    )

                except Exception as error:
                    db.rollback()

                    print(
                        f"{symbol.ticker}: "
                        f"database error: {error}"
                    )

            time.sleep(2)

    finally:
        db.close()
