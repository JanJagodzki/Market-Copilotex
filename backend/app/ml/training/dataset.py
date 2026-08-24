import pandas as pd

from sqlalchemy import text

from backend.app.db.database import engine
from backend.app.ml.features.build_features import FEATURE_COLUMNS


def load_dataset(
    start_date,
    end_date,
    sample_percent=100,
):
    if sample_percent < 1 or sample_percent > 100:
        raise ValueError(
            "sample_percent must be between 1 and 100"
        )

    sample = ""

    if sample_percent < 100:
        sample = (
            f"TABLESAMPLE BERNOULLI "
            f"({sample_percent}) REPEATABLE (42)"
        )

    columns = ", ".join(FEATURE_COLUMNS)

    query = text(
        f"""
        SELECT
            symbol_id,
            date,
            {columns},
            target_direction_1d
        FROM daily_features
        {sample}
        WHERE date >= :start_date
          AND date <= :end_date
          AND target_direction_1d IS NOT NULL
        """
    )

    data = pd.read_sql(
        query,
        engine,
        params={
            "start_date": start_date,
            "end_date": end_date,
        },
    )

    data = data.dropna(
        subset=FEATURE_COLUMNS
        + ["target_direction_1d"]
    )

    return data


def split_xy(data):
    x = data[
        FEATURE_COLUMNS
    ].astype("float32")

    y = data[
        "target_direction_1d"
    ].astype("int8")

    return x, y
