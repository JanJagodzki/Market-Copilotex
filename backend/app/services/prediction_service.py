from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sqlalchemy import text
from sqlalchemy.dialects.postgresql import insert

from app.db.database import SessionLocal
from app.models.model_prediction_daily import ModelPredictionDaily


MODEL_NAME = "baseline_30d_hgbr"
HORIZON_DAYS = 30


def float_or_none(value: Any) -> float | None:
    if value is None or pd.isna(value):
        return None

    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def load_model_bundle() -> dict:
    project_root = Path(__file__).resolve().parents[3]
    model_path = project_root / "ml" / "artifacts" / "baseline_30d_hgbr.joblib"

    if not model_path.exists():
        raise FileNotFoundError(
            f"Missing model file: {model_path}. "
            "Run: python -m scripts.train_baseline_30d"
        )

    return joblib.load(model_path)


def load_latest_prediction_dataset(feature_columns: list[str]) -> pd.DataFrame:
    query = text(
        """
        WITH latest_date AS (
            SELECT MAX(date) AS date
            FROM features_daily
        ),
        asset_counts AS (
            SELECT asset_id, COUNT(*) AS rows_count
            FROM market_prices_daily
            GROUP BY asset_id
        )
        SELECT
            a.id AS asset_id,
            a.symbol,
            f.date,
            f.daily_return,
            f.log_return,
            f.volume_change,
            f.volatility_7d,
            f.volatility_30d,
            f.distance_from_sma_20,
            f.distance_from_sma_200,
            f.momentum_7d,
            f.momentum_30d,
            f.momentum_90d,
            f.drawdown_252d
        FROM features_daily f
        JOIN latest_date ld
            ON ld.date = f.date
        JOIN assets a
            ON a.id = f.asset_id
        JOIN asset_counts ac
            ON ac.asset_id = f.asset_id
        WHERE a.universe_name = 'USA_TOP_100'
          AND a.is_active = TRUE
          AND ac.rows_count >= 500
        ORDER BY a.universe_rank ASC
        """
    )

    with SessionLocal() as session:
        df = pd.read_sql_query(query, session.connection())

    df["date"] = pd.to_datetime(df["date"])
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=feature_columns)

    return df


def generate_latest_predictions() -> dict[str, int | str]:
    bundle = load_model_bundle()

    model = bundle["model"]
    feature_columns = bundle["feature_columns"]

    df = load_latest_prediction_dataset(feature_columns=feature_columns)

    if df.empty:
        raise ValueError("No rows available for latest prediction dataset.")

    predictions = model.predict(df[feature_columns])

    df["predicted_return"] = predictions

    df["prediction_score"] = df["predicted_return"].rank(pct=True) * 100.0
    df["risk_score"] = df["volatility_30d"].rank(pct=True) * 100.0

    df["final_score"] = (
        0.75 * df["prediction_score"]
        + 0.25 * (100.0 - df["risk_score"])
    )

    df = df.sort_values("final_score", ascending=False).reset_index(drop=True)
    df["prediction_rank"] = df.index + 1

    prediction_date = df["date"].iloc[0].date()

    rows_updated = 0

    with SessionLocal() as session:
        for _, row in df.iterrows():
            values = {
                "asset_id": int(row["asset_id"]),
                "date": prediction_date,
                "horizon_days": HORIZON_DAYS,
                "model_name": MODEL_NAME,
                "predicted_return": float(row["predicted_return"]),
                "prediction_score": float(row["prediction_score"]),
                "risk_score": float(row["risk_score"]),
                "final_score": float(row["final_score"]),
                "prediction_rank": int(row["prediction_rank"]),
            }

            statement = insert(ModelPredictionDaily).values(**values)

            statement = statement.on_conflict_do_update(
                index_elements=[
                    "asset_id",
                    "date",
                    "horizon_days",
                    "model_name",
                ],
                set_={
                    "predicted_return": statement.excluded.predicted_return,
                    "prediction_score": statement.excluded.prediction_score,
                    "risk_score": statement.excluded.risk_score,
                    "final_score": statement.excluded.final_score,
                    "prediction_rank": statement.excluded.prediction_rank,
                },
            )

            session.execute(statement)
            rows_updated += 1

        session.commit()

    return {
        "prediction_date": str(prediction_date),
        "rows_updated": rows_updated,
        "model_name": MODEL_NAME,
        "horizon_days": HORIZON_DAYS,
    }
