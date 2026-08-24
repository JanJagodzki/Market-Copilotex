import pandas as pd

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert

from backend.app.db.database import (
    SessionLocal,
    engine,
)
from backend.app.db.models import (
    DailyFeature,
    DailyPrice,
    Symbol,
)
from backend.app.ml.features.build_features import (
    FEATURE_COLUMNS,
    build_features,
)


def optional_float(value):
    if pd.isna(value):
        return None

    return float(value)


def optional_int(value):
    if pd.isna(value):
        return None

    return int(value)


def load_prices(symbol_id):
    query = (
        select(
            DailyPrice.date,
            DailyPrice.open,
            DailyPrice.high,
            DailyPrice.low,
            DailyPrice.close,
            DailyPrice.adjusted_close,
            DailyPrice.volume,
        )
        .where(
            DailyPrice.symbol_id == symbol_id
        )
        .order_by(DailyPrice.date)
    )

    return pd.read_sql(
        query,
        engine,
    )


def save_features(db, symbol, data):
    rows = []

    for _, row in data.iterrows():
        rows.append(
            {
                "symbol_id": symbol.id,
                "date": row["date"],

                "return_1d": optional_float(
                    row["return_1d"]
                ),
                "return_5d": optional_float(
                    row["return_5d"]
                ),
                "return_20d": optional_float(
                    row["return_20d"]
                ),

                "volatility_5d": optional_float(
                    row["volatility_5d"]
                ),
                "volatility_20d": optional_float(
                    row["volatility_20d"]
                ),

                "sma_5_ratio": optional_float(
                    row["sma_5_ratio"]
                ),
                "sma_20_ratio": optional_float(
                    row["sma_20_ratio"]
                ),
                "sma_50_ratio": optional_float(
                    row["sma_50_ratio"]
                ),

                "ema_12_ratio": optional_float(
                    row["ema_12_ratio"]
                ),
                "ema_26_ratio": optional_float(
                    row["ema_26_ratio"]
                ),

                "rsi_14": optional_float(
                    row["rsi_14"]
                ),

                "macd_ratio": optional_float(
                    row["macd_ratio"]
                ),

                "high_low_range": optional_float(
                    row["high_low_range"]
                ),

                "open_close_return": optional_float(
                    row["open_close_return"]
                ),

                "volume_change_1d": optional_float(
                    row["volume_change_1d"]
                ),

                "target_return_1d": optional_float(
                    row["target_return_1d"]
                ),

                "target_direction_1d": optional_int(
                    row["target_direction_1d"]
                ),

                "target_return_5d": optional_float(
                    row["target_return_5d"]
                ),

                "target_direction_5d": optional_int(
                    row["target_direction_5d"]
                ),
            }
        )

    batch_size = 1000

    for start in range(
        0,
        len(rows),
        batch_size,
    ):
        batch = rows[
            start:start + batch_size
        ]

        statement = insert(
            DailyFeature
        ).values(batch)

        statement = statement.on_conflict_do_update(
            constraint="uq_daily_features_symbol_date",
            set_={
                column: getattr(
                    statement.excluded,
                    column,
                )
                for column in (
                    FEATURE_COLUMNS
                    + [
                        "target_return_1d",
                        "target_direction_1d",
                        "target_return_5d",
                        "target_direction_5d",
                    ]
                )
            },
        )

        db.execute(statement)

    return len(rows)


def build_for_tickers(tickers):
    db = SessionLocal()

    try:
        symbols = (
            db.query(Symbol)
            .filter(
                Symbol.ticker.in_(tickers)
            )
            .order_by(Symbol.ticker)
            .all()
        )

        for symbol in symbols:
            print(
                f"Building features: "
                f"{symbol.ticker}"
            )

            prices = load_prices(
                symbol.id
            )

            features = build_features(
                prices
            )

            features = features.dropna(
                subset=FEATURE_COLUMNS
            )

            count = save_features(
                db,
                symbol,
                features,
            )

            db.commit()

            print(
                f"{symbol.ticker}: "
                f"{count} feature rows"
            )

    except Exception:
        db.rollback()
        raise

    finally:
        db.close()


def main():
    build_for_tickers(
        [
            "AAPL",
            "MSFT",
            "NVDA",
        ]
    )


if __name__ == "__main__":
    main()
