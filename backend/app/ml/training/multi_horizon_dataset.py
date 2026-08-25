import pandas as pd

from sqlalchemy import text

from backend.app.db.database import engine
from backend.app.ml.features.build_features import FEATURE_COLUMNS


HORIZONS = [1, 5, 20, 60, 120, 252]

TRAIN_START = "2000-01-01"
VALIDATION_START = "2023-01-01"
TEST_START = "2025-01-01"


def get_market_dates():
    query = text(
        """
        SELECT DISTINCT date
        FROM daily_prices
        WHERE date >= :start_date
        ORDER BY date
        """
    )

    data = pd.read_sql(
        query,
        engine,
        params={
            "start_date": TRAIN_START,
        },
    )

    return pd.DatetimeIndex(
        pd.to_datetime(data["date"])
    )


def get_split_ranges(horizon):
    if horizon not in HORIZONS:
        raise ValueError(
            f"Unknown horizon: {horizon}"
        )

    dates = get_market_dates()

    validation_index = dates.searchsorted(
        pd.Timestamp(VALIDATION_START),
        side="left",
    )

    test_index = dates.searchsorted(
        pd.Timestamp(TEST_START),
        side="left",
    )

    train_end_index = (
        validation_index
        - horizon
        - 1
    )

    validation_end_index = (
        test_index
        - horizon
        - 1
    )

    test_end_index = (
        len(dates)
        - horizon
        - 1
    )

    if min(
        train_end_index,
        validation_end_index,
        test_end_index,
    ) < 0:
        raise RuntimeError(
            "Not enough market dates"
        )

    return {
        "train": (
            TRAIN_START,
            dates[train_end_index].date(),
        ),
        "validation": (
            VALIDATION_START,
            dates[validation_end_index].date(),
        ),
        "test": (
            TEST_START,
            dates[test_end_index].date(),
        ),
    }


def load_dataset(
    horizon,
    split,
    sample_percent=100,
):
    if horizon not in HORIZONS:
        raise ValueError(
            f"Unknown horizon: {horizon}"
        )

    if sample_percent < 1 or sample_percent > 100:
        raise ValueError(
            "sample_percent must be 1-100"
        )

    ranges = get_split_ranges(
        horizon
    )

    if split not in ranges:
        raise ValueError(
            f"Unknown split: {split}"
        )

    start_date, end_date = ranges[
        split
    ]

    target_column = (
        f"target_direction_{horizon}d"
    )

    columns = ", ".join(
        FEATURE_COLUMNS
    )

    sample_filter = ""

    if sample_percent < 100:
        sample_filter = """
        AND MOD(
            ABS(
                HASHTEXT(
                    symbol_id::TEXT
                    || date::TEXT
                )::BIGINT
            ),
            100
        ) < :sample_percent
        """

    query = text(
        f"""
        SELECT
            symbol_id,
            date,
            {columns},
            {target_column} AS target
        FROM daily_features
        WHERE date >= :start_date
          AND date <= :end_date
          AND {target_column} IS NOT NULL
          {sample_filter}
        """
    )

    params = {
        "start_date": start_date,
        "end_date": end_date,
        "sample_percent": sample_percent,
    }

    data = pd.read_sql(
        query,
        engine,
        params=params,
    )

    data = data.dropna(
        subset=FEATURE_COLUMNS
        + ["target"]
    )

    return data


def split_xy(data):
    x = data[
        FEATURE_COLUMNS
    ].astype("float32")

    y = data[
        "target"
    ].astype("int8")

    return x, y
