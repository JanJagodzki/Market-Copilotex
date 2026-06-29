from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sqlalchemy import delete, func, text
from sqlalchemy.dialects.postgresql import insert

from app.db.database import SessionLocal
from app.models.model_prediction_daily import ModelPredictionDaily
from app.services.horizon_target_service import DEFAULT_HORIZONS


def get_project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def load_production_model(horizon_days: int) -> dict:
    model_path = (
        get_project_root()
        / "ml"
        / "artifacts"
        / "production"
        / f"production_h{horizon_days}.joblib"
    )

    if not model_path.exists():
        raise FileNotFoundError(
            f"Missing production model: {model_path}. "
            "Run: python -m scripts.train_model_tournament"
        )

    return joblib.load(model_path)


def load_latest_prediction_dataset(feature_columns: list[str]) -> pd.DataFrame:
    query = text(
        """
        WITH asset_counts AS (
            SELECT asset_id, COUNT(*) AS rows_count
            FROM market_prices_daily
            GROUP BY asset_id
        ),
        candidate_rows AS (
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
            JOIN assets a
                ON a.id = f.asset_id
            JOIN asset_counts ac
                ON ac.asset_id = f.asset_id
            WHERE a.universe_name = 'USA_TOP_100'
              AND a.is_active = TRUE
              AND ac.rows_count >= 500
        ),
        latest_usable_date AS (
            SELECT date
            FROM candidate_rows
            WHERE daily_return IS NOT NULL
              AND log_return IS NOT NULL
              AND volume_change IS NOT NULL
              AND volatility_7d IS NOT NULL
              AND volatility_30d IS NOT NULL
              AND distance_from_sma_20 IS NOT NULL
              AND distance_from_sma_200 IS NOT NULL
              AND momentum_7d IS NOT NULL
              AND momentum_30d IS NOT NULL
              AND momentum_90d IS NOT NULL
              AND drawdown_252d IS NOT NULL
            GROUP BY date
            HAVING COUNT(*) >= 50
            ORDER BY date DESC
            LIMIT 1
        )
        SELECT cr.*
        FROM candidate_rows cr
        JOIN latest_usable_date lud
            ON lud.date = cr.date
        ORDER BY cr.symbol ASC
        """
    )

    with SessionLocal() as session:
        df = pd.read_sql_query(query, session.connection())

    if df.empty:
        return df

    df["date"] = pd.to_datetime(df["date"])
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=feature_columns)

    return df


def generate_production_predictions_for_horizon(horizon_days: int) -> dict:
    bundle = load_production_model(horizon_days)

    model = bundle["model"]
    model_name = bundle["model_name"]
    feature_columns = bundle["feature_columns"]

    df = load_latest_prediction_dataset(feature_columns)

    if df.empty:
        raise ValueError(f"No prediction rows for horizon {horizon_days}")

    df["predicted_return"] = model.predict(df[feature_columns])
    df["prediction_score"] = df["predicted_return"].rank(pct=True) * 100.0
    df["risk_score"] = df["volatility_30d"].rank(pct=True) * 100.0

    df["final_score"] = (
        0.75 * df["prediction_score"]
        + 0.25 * (100.0 - df["risk_score"])
    )

    df = df.sort_values("final_score", ascending=False).reset_index(drop=True)
    df["prediction_rank"] = df.index + 1

    prediction_date = df["date"].iloc[0].date()

    with SessionLocal() as session:
        session.execute(
            delete(ModelPredictionDaily)
            .where(ModelPredictionDaily.date == prediction_date)
            .where(ModelPredictionDaily.horizon_days == horizon_days)
        )

        rows_updated = 0

        for _, row in df.iterrows():
            values = {
                "asset_id": int(row["asset_id"]),
                "date": prediction_date,
                "horizon_days": horizon_days,
                "model_name": model_name,
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
                    "updated_at": func.now(),
                },
            )

            session.execute(statement)
            rows_updated += 1

        session.commit()

    return {
        "prediction_date": str(prediction_date),
        "horizon_days": horizon_days,
        "model_name": model_name,
        "rows_updated": rows_updated,
    }


def generate_all_production_predictions(
    horizons: list[int] | None = None,
) -> dict:
    horizons_to_run = horizons or DEFAULT_HORIZONS

    results = []
    total_rows_updated = 0

    for horizon_days in horizons_to_run:
        result = generate_production_predictions_for_horizon(horizon_days)
        results.append(result)
        total_rows_updated += result["rows_updated"]

        print(
            f"Horizon {horizon_days}D: "
            f"{result['rows_updated']} predictions updated "
            f"using {result['model_name']}"
        )

    return {
        "horizons": horizons_to_run,
        "total_rows_updated": total_rows_updated,
        "results": results,
    }
