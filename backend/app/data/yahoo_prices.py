import time
import pandas as pd
import yfinance as yf

from sqlalchemy.dialects.postgresql import insert

from backend.app.db.database import SessionLocal
from backend.app.db.models import DailyPrice, Symbol


def download_prices(tickers, period="max"):
    return yf.download(
        tickers=tickers,
        period=period,
        interval="1d",
        group_by="ticker",
        auto_adjust=False,
        actions=False,
        progress=False,
        threads=True,
    )


def get_symbol_data(data, ticker):
    if data.empty:
        return pd.DataFrame()

    if isinstance(data.columns, pd.MultiIndex):
        if ticker in data.columns.get_level_values(0):
            return data[ticker].copy()

        if ticker in data.columns.get_level_values(1):
            return data.xs(
                ticker,
                axis=1,
                level=1,
            ).copy()

        return pd.DataFrame()

    return data.copy()


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
