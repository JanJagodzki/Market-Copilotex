from datetime import date
from decimal import Decimal, InvalidOperation
from typing import Any

import pandas as pd
import yfinance as yf
from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert

from app.db.database import SessionLocal
from app.models.asset import Asset
from app.models.market_price_daily import MarketPriceDaily


def scalar_value(value: Any) -> Any:
    """
    yfinance/pandas can sometimes return a Series instead of a single value.
    This function safely extracts one scalar value.
    """
    if isinstance(value, pd.Series):
        if value.empty:
            return None
        return value.iloc[0]

    return value


def decimal_or_none(value: Any) -> Decimal | None:
    value = scalar_value(value)

    if value is None or pd.isna(value):
        return None

    try:
        return Decimal(str(round(float(value), 6)))
    except (ValueError, TypeError, InvalidOperation):
        return None


def int_or_none(value: Any) -> int | None:
    value = scalar_value(value)

    if value is None or pd.isna(value):
        return None

    try:
        return int(float(value))
    except (ValueError, TypeError):
        return None


def fetch_price_history(symbol: str, period: str = "5y") -> pd.DataFrame:
    """
    Fetch daily OHLCV price history from Yahoo Finance via yfinance.
    Ticker.history usually returns cleaner columns than yf.download.
    """
    ticker = yf.Ticker(symbol)

    data = ticker.history(
        period=period,
        interval="1d",
        auto_adjust=False,
        actions=False,
    )

    if data.empty:
        return data

    data = data.reset_index()

    return data


def get_row_date(row: pd.Series) -> date | None:
    raw_date = row.get("Date")

    if raw_date is None:
        raw_date = row.get("Datetime")

    raw_date = scalar_value(raw_date)

    if raw_date is None or pd.isna(raw_date):
        return None

    if hasattr(raw_date, "date"):
        return raw_date.date()

    return raw_date


def upsert_prices_for_asset(asset_id: int, symbol: str, period: str = "5y") -> int:
    data = fetch_price_history(symbol, period=period)

    if data.empty:
        print(f"{symbol}: no price data")
        return 0

    inserted_or_updated = 0

    with SessionLocal() as session:
        for _, row in data.iterrows():
            row_date = get_row_date(row)

            if row_date is None:
                continue

            values = {
                "asset_id": asset_id,
                "date": row_date,
                "open": decimal_or_none(row.get("Open")),
                "high": decimal_or_none(row.get("High")),
                "low": decimal_or_none(row.get("Low")),
                "close": decimal_or_none(row.get("Close")),
                "adjusted_close": decimal_or_none(row.get("Adj Close")),
                "volume": int_or_none(row.get("Volume")),
            }

            statement = insert(MarketPriceDaily).values(**values)

            statement = statement.on_conflict_do_update(
                index_elements=["asset_id", "date"],
                set_={
                    "open": statement.excluded.open,
                    "high": statement.excluded.high,
                    "low": statement.excluded.low,
                    "close": statement.excluded.close,
                    "adjusted_close": statement.excluded.adjusted_close,
                    "volume": statement.excluded.volume,
                },
            )

            session.execute(statement)
            inserted_or_updated += 1

        session.commit()

    print(f"{symbol}: {inserted_or_updated} daily prices updated")
    return inserted_or_updated


def update_market_prices_for_active_assets(
    universe_name: str = "USA_TOP_100",
    period: str = "5y",
    limit: int | None = None,
) -> dict[str, int]:
    total_assets = 0
    total_rows = 0
    failed_assets = 0

    with SessionLocal() as session:
        query = (
            select(Asset)
            .where(Asset.universe_name == universe_name)
            .where(Asset.is_active.is_(True))
            .order_by(Asset.universe_rank.asc())
        )

        if limit is not None:
            query = query.limit(limit)

        assets = list(session.execute(query).scalars().all())

    for asset in assets:
        total_assets += 1

        try:
            updated_rows = upsert_prices_for_asset(
                asset_id=asset.id,
                symbol=asset.symbol,
                period=period,
            )
            total_rows += updated_rows
        except Exception as error:
            failed_assets += 1
            print(f"Failed to update {asset.symbol}: {error}")

    return {
        "assets_processed": total_assets,
        "price_rows_updated": total_rows,
        "failed_assets": failed_assets,
    }