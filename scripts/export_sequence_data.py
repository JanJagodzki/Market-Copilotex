from pathlib import Path

import numpy as np
import pandas as pd
from sqlalchemy import select

from backend.app.db.database import engine, SessionLocal
from backend.app.db.models import DailyFeature, Symbol
from backend.app.ml.features.build_features import FEATURE_COLUMNS


OUTPUT_DIR = Path("data/processed/sequences")


def load_symbol_features(symbol_id):
    feature_columns = [
        getattr(DailyFeature, column)
        for column in FEATURE_COLUMNS
    ]

    query = (
        select(
            DailyFeature.date,
            *feature_columns,
            DailyFeature.target_direction_1d,
        )
        .where(
            DailyFeature.symbol_id == symbol_id
        )
        .order_by(DailyFeature.date)
    )

    return pd.read_sql(
        query,
        engine,
    )


def export_symbol(symbol):
    data = load_symbol_features(symbol.id)

    if data.empty:
        return 0

    data = data.dropna(
        subset=FEATURE_COLUMNS
    )

    if data.empty:
        return 0

    x = data[
        FEATURE_COLUMNS
    ].to_numpy(
        dtype=np.float32
    )

    y = (
        data["target_direction_1d"]
        .fillna(-1)
        .to_numpy(dtype=np.int8)
    )

    dates = (
        pd.to_datetime(data["date"])
        .to_numpy(dtype="datetime64[D]")
    )

    path = OUTPUT_DIR / f"{symbol.id}.npz"

    np.savez_compressed(
        path,
        x=x,
        y=y,
        dates=dates,
        ticker=np.array(symbol.ticker),
    )

    return len(data)


def main():
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    db = SessionLocal()

    try:
        symbols = (
            db.query(Symbol)
            .filter(Symbol.active.is_(True))
            .order_by(Symbol.ticker)
            .all()
        )

        total = len(symbols)

        for number, symbol in enumerate(
            symbols,
            start=1,
        ):
            path = OUTPUT_DIR / f"{symbol.id}.npz"

            if path.exists():
                print(
                    f"[{number}/{total}] "
                    f"{symbol.ticker}: already exported"
                )
                continue

            try:
                rows = export_symbol(symbol)

                print(
                    f"[{number}/{total}] "
                    f"{symbol.ticker}: {rows} rows"
                )

            except Exception as error:
                print(
                    f"[{number}/{total}] "
                    f"{symbol.ticker}: error: {error}"
                )

    finally:
        db.close()


if __name__ == "__main__":
    main()
