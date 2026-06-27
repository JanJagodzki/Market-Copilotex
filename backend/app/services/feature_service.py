from typing import Any

import numpy as np
import pandas as pd
from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert

from app.db.database import SessionLocal
from app.models.asset import Asset
from app.models.feature_daily import FeatureDaily
from app.models.market_price_daily import MarketPriceDaily


def float_or_none(value: Any) -> float | None:
    if value is None or pd.isna(value):
        return None

    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def calculate_features_for_asset(asset_id: int, symbol: str) -> int:
    with SessionLocal() as session:
        price_rows = list(
            session.execute(
                select(
                    MarketPriceDaily.date,
                    MarketPriceDaily.close,
                    MarketPriceDaily.adjusted_close,
                    MarketPriceDaily.volume,
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
        columns=["date", "close", "adjusted_close", "volume"],
    )

    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df["adjusted_close"] = pd.to_numeric(df["adjusted_close"], errors="coerce")
    df["volume"] = pd.to_numeric(df["volume"], errors="coerce")

    df["price"] = df["adjusted_close"].fillna(df["close"])

    df["daily_return"] = df["price"].pct_change()
    df["log_return"] = np.log(df["price"] / df["price"].shift(1))

    df["volume_change"] = df["volume"].pct_change()

    df["volatility_7d"] = df["log_return"].rolling(window=7).std()
    df["volatility_30d"] = df["log_return"].rolling(window=30).std()

    df["sma_20"] = df["price"].rolling(window=20).mean()
    df["sma_50"] = df["price"].rolling(window=50).mean()
    df["sma_200"] = df["price"].rolling(window=200).mean()

    df["distance_from_sma_20"] = (df["price"] / df["sma_20"]) - 1
    df["distance_from_sma_200"] = (df["price"] / df["sma_200"]) - 1

    df["momentum_7d"] = df["price"].pct_change(periods=7)
    df["momentum_30d"] = df["price"].pct_change(periods=30)
    df["momentum_90d"] = df["price"].pct_change(periods=90)

    rolling_high_252d = df["price"].rolling(window=252, min_periods=1).max()
    df["drawdown_252d"] = (df["price"] / rolling_high_252d) - 1

    rows_updated = 0

    with SessionLocal() as session:
        for _, row in df.iterrows():
            values = {
                "asset_id": asset_id,
                "date": row["date"],
                "daily_return": float_or_none(row["daily_return"]),
                "log_return": float_or_none(row["log_return"]),
                "volume_change": float_or_none(row["volume_change"]),
                "volatility_7d": float_or_none(row["volatility_7d"]),
                "volatility_30d": float_or_none(row["volatility_30d"]),
                "sma_20": float_or_none(row["sma_20"]),
                "sma_50": float_or_none(row["sma_50"]),
                "sma_200": float_or_none(row["sma_200"]),
                "distance_from_sma_20": float_or_none(row["distance_from_sma_20"]),
                "distance_from_sma_200": float_or_none(row["distance_from_sma_200"]),
                "momentum_7d": float_or_none(row["momentum_7d"]),
                "momentum_30d": float_or_none(row["momentum_30d"]),
                "momentum_90d": float_or_none(row["momentum_90d"]),
                "drawdown_252d": float_or_none(row["drawdown_252d"]),
            }

            statement = insert(FeatureDaily).values(**values)

            statement = statement.on_conflict_do_update(
                index_elements=["asset_id", "date"],
                set_={
                    "daily_return": statement.excluded.daily_return,
                    "log_return": statement.excluded.log_return,
                    "volume_change": statement.excluded.volume_change,
                    "volatility_7d": statement.excluded.volatility_7d,
                    "volatility_30d": statement.excluded.volatility_30d,
                    "sma_20": statement.excluded.sma_20,
                    "sma_50": statement.excluded.sma_50,
                    "sma_200": statement.excluded.sma_200,
                    "distance_from_sma_20": statement.excluded.distance_from_sma_20,
                    "distance_from_sma_200": statement.excluded.distance_from_sma_200,
                    "momentum_7d": statement.excluded.momentum_7d,
                    "momentum_30d": statement.excluded.momentum_30d,
                    "momentum_90d": statement.excluded.momentum_90d,
                    "drawdown_252d": statement.excluded.drawdown_252d,
                },
            )

            session.execute(statement)
            rows_updated += 1

        session.commit()

    print(f"{symbol}: {rows_updated} feature rows updated")
    return rows_updated


def calculate_features_for_active_assets(
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
    feature_rows_updated = 0
    failed_assets = 0

    for asset_id, symbol in assets:
        assets_processed += 1

        try:
            rows = calculate_features_for_asset(asset_id=asset_id, symbol=symbol)
            feature_rows_updated += rows
        except Exception as error:
            failed_assets += 1
            print(f"Failed to calculate features for {symbol}: {error}")

    return {
        "assets_processed": assets_processed,
        "feature_rows_updated": feature_rows_updated,
        "failed_assets": failed_assets,
    }

