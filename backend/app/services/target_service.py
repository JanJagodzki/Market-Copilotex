from typing import Any

import pandas as pd
from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert

from app.db.database import SessionLocal
from app.models.asset import Asset
from app.models.market_price_daily import MarketPriceDaily
from app.models.target_daily import TargetDaily


HORIZONS = [1, 7, 30, 90, 180, 252]


def float_or_none(value: Any) -> float | None:
    if value is None or pd.isna(value):
        return None

    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def direction_or_none(value: Any) -> int | None:
    if value is None or pd.isna(value):
        return None

    return 1 if float(value) > 0 else 0


def calculate_targets_for_asset(asset_id: int, symbol: str) -> int:
    with SessionLocal() as session:
        price_rows = list(
            session.execute(
                select(
                    MarketPriceDaily.date,
                    MarketPriceDaily.close,
                    MarketPriceDaily.adjusted_close,
                )
                .where(MarketPriceDaily.asset_id == asset_id)
                .order_by(MarketPriceDaily.date.asc())
            ).all()
        )

    if not price_rows:
        print(f"{symbol}: no price rows")
        return 0

    df = pd.DataFrame(
        price_rows,
        columns=["date", "close", "adjusted_close"],
    )

    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df["adjusted_close"] = pd.to_numeric(df["adjusted_close"], errors="coerce")
    df["price"] = df["adjusted_close"].fillna(df["close"])

    for horizon in HORIZONS:
        future_price = df["price"].shift(-horizon)
        df[f"future_return_{horizon}d"] = (future_price / df["price"]) - 1

    rows_updated = 0

    with SessionLocal() as session:
        for _, row in df.iterrows():
            values = {
                "asset_id": asset_id,
                "date": row["date"],
                "future_return_1d": float_or_none(row["future_return_1d"]),
                "future_return_7d": float_or_none(row["future_return_7d"]),
                "future_return_30d": float_or_none(row["future_return_30d"]),
                "future_return_90d": float_or_none(row["future_return_90d"]),
                "future_return_180d": float_or_none(row["future_return_180d"]),
                "future_return_252d": float_or_none(row["future_return_252d"]),
                "future_direction_1d": direction_or_none(row["future_return_1d"]),
                "future_direction_7d": direction_or_none(row["future_return_7d"]),
                "future_direction_30d": direction_or_none(row["future_return_30d"]),
                "future_direction_90d": direction_or_none(row["future_return_90d"]),
                "future_direction_180d": direction_or_none(row["future_return_180d"]),
                "future_direction_252d": direction_or_none(row["future_return_252d"]),
            }

            statement = insert(TargetDaily).values(**values)

            statement = statement.on_conflict_do_update(
                index_elements=["asset_id", "date"],
                set_={
                    "future_return_1d": statement.excluded.future_return_1d,
                    "future_return_7d": statement.excluded.future_return_7d,
                    "future_return_30d": statement.excluded.future_return_30d,
                    "future_return_90d": statement.excluded.future_return_90d,
                    "future_return_180d": statement.excluded.future_return_180d,
                    "future_return_252d": statement.excluded.future_return_252d,
                    "future_direction_1d": statement.excluded.future_direction_1d,
                    "future_direction_7d": statement.excluded.future_direction_7d,
                    "future_direction_30d": statement.excluded.future_direction_30d,
                    "future_direction_90d": statement.excluded.future_direction_90d,
                    "future_direction_180d": statement.excluded.future_direction_180d,
                    "future_direction_252d": statement.excluded.future_direction_252d,
                },
            )

            session.execute(statement)
            rows_updated += 1

        session.commit()

    print(f"{symbol}: {rows_updated} target rows updated")
    return rows_updated


def calculate_targets_for_active_assets(
    universe_name: str = "USA_TOP_100",
    limit: int | None = None,
) -> dict[str, int]:
    with SessionLocal() as session:
        query = (
            select(Asset.id, Asset.symbol)
            .where(Asset.universe_name == universe_name)
            .where(Asset.is_active.is_(True))
            .order_by(Asset.universe_rank.asc())
        )

        if limit is not None:
            query = query.limit(limit)

        assets = list(session.execute(query).all())

    assets_processed = 0
    target_rows_updated = 0
    failed_assets = 0

    for asset_id, symbol in assets:
        assets_processed += 1

        try:
            rows = calculate_targets_for_asset(asset_id=asset_id, symbol=symbol)
            target_rows_updated += rows
        except Exception as error:
            failed_assets += 1
            print(f"Failed to calculate targets for {symbol}: {error}")

    return {
        "assets_processed": assets_processed,
        "target_rows_updated": target_rows_updated,
        "failed_assets": failed_assets,
    }