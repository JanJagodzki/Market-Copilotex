from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sqlalchemy import text

from app.db.database import SessionLocal


HORIZONS = [1, 5, 10, 20, 30, 60, 90, 120, 150, 180, 252]

FEATURE_COLUMNS = [
    "daily_return",
    "log_return",
    "volume_change",
    "volatility_7d",
    "volatility_30d",
    "distance_from_sma_20",
    "distance_from_sma_200",
    "momentum_7d",
    "momentum_30d",
    "momentum_90d",
    "drawdown_252d",
]

TRAIN_END = "2021-01-01"
VALID_END = "2024-01-01"

MODEL_NAME_TEMPLATE = "baseline_h{horizon}_hgbr"


def get_project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def load_dataset_for_horizon(horizon_days: int) -> pd.DataFrame:
    query = text(
        """
        WITH asset_counts AS (
            SELECT asset_id, COUNT(*) AS rows_count
            FROM market_prices_daily
            GROUP BY asset_id
        )
        SELECT
            a.symbol,
            a.id AS asset_id,
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
            f.drawdown_252d,
            th.future_return,
            th.future_direction
        FROM features_daily f
        JOIN targets_horizon_daily th
            ON th.asset_id = f.asset_id
           AND th.date = f.date
        JOIN assets a
            ON a.id = f.asset_id
        JOIN asset_counts ac
            ON ac.asset_id = f.asset_id
        WHERE a.universe_name = 'USA_TOP_100'
          AND a.is_active = TRUE
          AND ac.rows_count >= 500
          AND th.horizon_days = :horizon_days
          AND th.future_return IS NOT NULL
        ORDER BY f.date ASC, a.symbol ASC
        """
    )

    with SessionLocal() as session:
        df = pd.read_sql_query(
            query,
            session.connection(),
            params={"horizon_days": horizon_days},
        )

    df["date"] = pd.to_datetime(df["date"])
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=FEATURE_COLUMNS + ["future_return"])

    return df


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    direction_true = y_true > 0
    direction_pred = y_pred > 0

    always_up_accuracy = float(direction_true.mean())
    direction_accuracy = float((direction_true == direction_pred).mean())

    pearson = float(pd.Series(y_true).corr(pd.Series(y_pred), method="pearson"))
    spearman = float(pd.Series(y_true).corr(pd.Series(y_pred), method="spearman"))

    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(mean_squared_error(y_true, y_pred) ** 0.5),
        "r2": float(r2_score(y_true, y_pred)),
        "direction_accuracy": direction_accuracy,
        "always_up_accuracy": always_up_accuracy,
        "pearson": pearson,
        "spearman": spearman,
    }


def ranking_diagnostic(test_df: pd.DataFrame) -> dict:
    grouped = []

    for date, group in test_df.groupby("date"):
        if len(group) < 20:
            continue

        group = group.copy()

        model_top10 = group.sort_values("prediction", ascending=False).head(10)
        model_bottom10 = group.sort_values("prediction", ascending=True).head(10)
        momentum_top10 = group.sort_values("momentum_30d", ascending=False).head(10)

        grouped.append(
            {
                "date": date,
                "model_top10_return": model_top10["future_return"].mean(),
                "model_bottom10_return": model_bottom10["future_return"].mean(),
                "momentum_top10_return": momentum_top10["future_return"].mean(),
                "universe_return": group["future_return"].mean(),
            }
        )

    if not grouped:
        return {
            "dates_evaluated": 0,
            "model_top10_avg_return": None,
            "model_bottom10_avg_return": None,
            "momentum_top10_avg_return": None,
            "universe_avg_return": None,
            "model_top10_vs_universe_win_rate": None,
            "model_top10_vs_momentum_win_rate": None,
            "model_top10_vs_bottom10_win_rate": None,
        }

    result_df = pd.DataFrame(grouped)

    return {
        "dates_evaluated": int(len(result_df)),
        "model_top10_avg_return": float(result_df["model_top10_return"].mean()),
        "model_bottom10_avg_return": float(result_df["model_bottom10_return"].mean()),
        "momentum_top10_avg_return": float(result_df["momentum_top10_return"].mean()),
        "universe_avg_return": float(result_df["universe_return"].mean()),
        "model_top10_vs_universe_win_rate": float(
            (result_df["model_top10_return"] > result_df["universe_return"]).mean()
        ),
        "model_top10_vs_momentum_win_rate": float(
            (result_df["model_top10_return"] > result_df["momentum_top10_return"]).mean()
        ),
        "model_top10_vs_bottom10_win_rate": float(
            (result_df["model_top10_return"] > result_df["model_bottom10_return"]).mean()
        ),
    }


def train_one_horizon(horizon_days: int) -> dict:
    print("")
    print("=" * 80)
    print(f"Training horizon: {horizon_days}D")
    print("=" * 80)

    df = load_dataset_for_horizon(horizon_days)

    print(f"Rows loaded: {len(df)}")
    print(f"Date range: {df['date'].min().date()} -> {df['date'].max().date()}")
    print(f"Symbols: {df['symbol'].nunique()}")

    train_df = df[df["date"] < TRAIN_END].copy()
    valid_df = df[(df["date"] >= TRAIN_END) & (df["date"] < VALID_END)].copy()
    test_df = df[df["date"] >= VALID_END].copy()

    print("")
    print("Split:")
    print(f"Train rows: {len(train_df)}")
    print(f"Valid rows: {len(valid_df)}")
    print(f"Test rows: {len(test_df)}")

    X_train = train_df[FEATURE_COLUMNS]
    y_train = train_df["future_return"]

    X_valid = valid_df[FEATURE_COLUMNS]
    y_valid = valid_df["future_return"]

    X_test = test_df[FEATURE_COLUMNS]
    y_test = test_df["future_return"]

    model = HistGradientBoostingRegressor(
        max_iter=300,
        learning_rate=0.05,
        max_leaf_nodes=31,
        l2_regularization=0.01,
        random_state=42,
    )

    model.fit(X_train, y_train)

    valid_pred = model.predict(X_valid)
    test_pred = model.predict(X_test)

    valid_metrics = calculate_metrics(y_valid.to_numpy(), valid_pred)
    test_metrics = calculate_metrics(y_test.to_numpy(), test_pred)

    test_df["prediction"] = test_pred
    ranking_metrics = ranking_diagnostic(test_df)

    print("")
    print("Validation metrics:")
    for key, value in valid_metrics.items():
        print(f"{key}: {value:.6f}")

    print("")
    print("Test metrics:")
    for key, value in test_metrics.items():
        print(f"{key}: {value:.6f}")

    print("")
    print("Ranking diagnostic:")
    for key, value in ranking_metrics.items():
        if value is None:
            print(f"{key}: None")
        elif isinstance(value, int):
            print(f"{key}: {value}")
        else:
            print(f"{key}: {value:.6f}")

    project_root = get_project_root()
    artifacts_dir = project_root / "ml" / "artifacts" / "multi_horizon"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    model_name = MODEL_NAME_TEMPLATE.format(horizon=horizon_days)
    model_path = artifacts_dir / f"{model_name}.joblib"

    bundle = {
        "model": model,
        "model_name": model_name,
        "horizon_days": horizon_days,
        "feature_columns": FEATURE_COLUMNS,
        "train_end": TRAIN_END,
        "valid_end": VALID_END,
        "valid_metrics": valid_metrics,
        "test_metrics": test_metrics,
        "ranking_metrics": ranking_metrics,
    }

    joblib.dump(bundle, model_path)

    print("")
    print(f"Saved model: {model_path}")

    return {
        "horizon_days": horizon_days,
        "model_name": model_name,
        "model_path": str(model_path),
        **{f"valid_{k}": v for k, v in valid_metrics.items()},
        **{f"test_{k}": v for k, v in test_metrics.items()},
        **{f"ranking_{k}": v for k, v in ranking_metrics.items()},
    }


def main() -> None:
    results = []

    for horizon_days in HORIZONS:
        result = train_one_horizon(horizon_days)
        results.append(result)

    project_root = get_project_root()
    artifacts_dir = project_root / "ml" / "artifacts" / "multi_horizon"
    report_path = artifacts_dir / "baseline_multi_horizon_report.csv"

    report_df = pd.DataFrame(results)
    report_df.to_csv(report_path, index=False)

    print("")
    print("=" * 80)
    print("Multi-horizon baseline training finished.")
    print(f"Report saved: {report_path}")
    print("=" * 80)

    print("")
    print(report_df[
        [
            "horizon_days",
            "model_name",
            "test_direction_accuracy",
            "test_always_up_accuracy",
            "test_spearman",
            "ranking_model_top10_avg_return",
            "ranking_universe_avg_return",
            "ranking_model_top10_vs_universe_win_rate",
        ]
    ].to_string(index=False))


if __name__ == "__main__":
    main()