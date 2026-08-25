from pathlib import Path

import numpy as np
import pandas as pd
from sqlalchemy import text

from backend.app.db.database import engine
from backend.app.ml.features.build_features import (
    FEATURE_COLUMNS,
)


DATA_DIR = Path(
    "data/processed/sequences"
)

HORIZONS = [
    1,
    5,
    20,
    60,
    120,
    252,
]


def get_symbols():
    query = text(
        """
        SELECT id, ticker
        FROM symbols
        WHERE active = TRUE
        ORDER BY ticker
        """
    )

    return pd.read_sql(
        query,
        engine,
    )


def export_symbol(
    symbol_id,
    ticker,
):
    feature_columns = ", ".join(
        FEATURE_COLUMNS
    )

    target_columns = ", ".join(
        [
            f"target_direction_{h}d"
            for h in HORIZONS
        ]
    )

    query = text(
        f"""
        SELECT
            date,
            {feature_columns},
            {target_columns}
        FROM daily_features
        WHERE symbol_id = :symbol_id
        ORDER BY date
        """
    )

    data = pd.read_sql(
        query,
        engine,
        params={
            "symbol_id": symbol_id,
        },
    )

    data = data.dropna(
        subset=FEATURE_COLUMNS
    )

    if len(data) < 60:
        return 0

    x = (
        data[FEATURE_COLUMNS]
        .astype("float32")
        .to_numpy()
    )

    dates = (
        pd.to_datetime(
            data["date"]
        )
        .to_numpy(
            dtype="datetime64[D]"
        )
    )

    payload = {
        "x": x,
        "dates": dates,
        "ticker": np.array(
            ticker
        ),
    }

    for horizon in HORIZONS:
        key = (
            f"target_direction_"
            f"{horizon}d"
        )

        y = (
            data[key]
            .fillna(-1)
            .astype("int8")
            .to_numpy()
        )

        payload[
            f"y_{horizon}d"
        ] = y

        if horizon == 1:
            payload["y"] = y

    path = (
        DATA_DIR /
        f"{ticker}.npz"
    )

    np.savez_compressed(
        path,
        **payload,
    )

    return len(data)


def main():
    DATA_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    symbols = get_symbols()

    print(
        "Exporting multi-horizon "
        "sequence data"
    )

    print(
        f"Symbols: {len(symbols)}"
    )

    exported = 0

    for index, row in (
        symbols.iterrows()
    ):
        rows = export_symbol(
            symbol_id=row["id"],
            ticker=row["ticker"],
        )

        if rows > 0:
            exported += 1

        print(
            f"[{index + 1}/"
            f"{len(symbols)}] "
            f"{row['ticker']} "
            f"{rows}"
        )

    print()
    print(
        f"Exported symbols: "
        f"{exported}"
    )


if __name__ == "__main__":
    main()
