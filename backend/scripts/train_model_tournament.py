from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
from typing import Callable

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sqlalchemy import text

from app.db.database import SessionLocal
from app.services.horizon_target_service import DEFAULT_HORIZONS


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


def has_package(package_name: str) -> bool:
    return importlib.util.find_spec(package_name) is not None


def get_project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def parse_csv_ints(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def parse_csv_strings(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


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
            th.future_return
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


def build_model(model_key: str):
    if model_key == "hist_gradient":
        return HistGradientBoostingRegressor(
            max_iter=350,
            learning_rate=0.04,
            max_leaf_nodes=31,
            l2_regularization=0.01,
            random_state=42,
        )

    if model_key == "extra_trees":
        return ExtraTreesRegressor(
            n_estimators=250,
            max_depth=12,
            min_samples_leaf=20,
            bootstrap=True,
            max_samples=0.75,
            random_state=42,
            n_jobs=-1,
        )

    if model_key == "xgboost":
        from xgboost import XGBRegressor

        return XGBRegressor(
            n_estimators=500,
            max_depth=4,
            learning_rate=0.03,
            subsample=0.85,
            colsample_bytree=0.85,
            objective="reg:squarederror",
            tree_method="hist",
            reg_lambda=1.0,
            reg_alpha=0.05,
            random_state=42,
            n_jobs=-1,
        )

    if model_key == "lightgbm":
        from lightgbm import LGBMRegressor

        return LGBMRegressor(
            n_estimators=500,
            learning_rate=0.03,
            num_leaves=31,
            max_depth=-1,
            min_child_samples=30,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_lambda=1.0,
            reg_alpha=0.05,
            random_state=42,
            n_jobs=-1,
            verbose=-1,
        )

    if model_key == "catboost":
        from catboost import CatBoostRegressor

        return CatBoostRegressor(
            iterations=500,
            depth=6,
            learning_rate=0.04,
            loss_function="RMSE",
            random_seed=42,
            verbose=False,
            thread_count=-1,
        )

    raise ValueError(f"Unknown model key: {model_key}")


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    direction_true = y_true > 0
    direction_pred = y_pred > 0

    pearson = pd.Series(y_true).corr(pd.Series(y_pred), method="pearson")
    spearman = pd.Series(y_true).corr(pd.Series(y_pred), method="spearman")

    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(mean_squared_error(y_true, y_pred) ** 0.5),
        "r2": float(r2_score(y_true, y_pred)),
        "direction_accuracy": float((direction_true == direction_pred).mean()),
        "always_up_accuracy": float(direction_true.mean()),
        "pearson": float(pearson) if pd.notna(pearson) else 0.0,
        "spearman": float(spearman) if pd.notna(spearman) else 0.0,
    }


def ranking_diagnostic(test_df: pd.DataFrame) -> dict:
    rows = []

    for date, group in test_df.groupby("date"):
        if len(group) < 20:
            continue

        model_top10 = group.sort_values("prediction", ascending=False).head(10)
        model_bottom10 = group.sort_values("prediction", ascending=True).head(10)
        momentum_top10 = group.sort_values("momentum_30d", ascending=False).head(10)

        rows.append(
            {
                "date": date,
                "model_top10_return": model_top10["future_return"].mean(),
                "model_bottom10_return": model_bottom10["future_return"].mean(),
                "momentum_top10_return": momentum_top10["future_return"].mean(),
                "universe_return": group["future_return"].mean(),
            }
        )

    if not rows:
        return {
            "dates_evaluated": 0,
            "model_top10_avg_return": 0.0,
            "model_bottom10_avg_return": 0.0,
            "momentum_top10_avg_return": 0.0,
            "universe_avg_return": 0.0,
            "model_top10_vs_universe_win_rate": 0.0,
            "model_top10_vs_momentum_win_rate": 0.0,
            "model_top10_vs_bottom10_win_rate": 0.0,
            "model_top10_excess_return": 0.0,
        }

    result_df = pd.DataFrame(rows)

    model_top10_avg = float(result_df["model_top10_return"].mean())
    universe_avg = float(result_df["universe_return"].mean())

    return {
        "dates_evaluated": int(len(result_df)),
        "model_top10_avg_return": model_top10_avg,
        "model_bottom10_avg_return": float(result_df["model_bottom10_return"].mean()),
        "momentum_top10_avg_return": float(result_df["momentum_top10_return"].mean()),
        "universe_avg_return": universe_avg,
        "model_top10_vs_universe_win_rate": float(
            (result_df["model_top10_return"] > result_df["universe_return"]).mean()
        ),
        "model_top10_vs_momentum_win_rate": float(
            (result_df["model_top10_return"] > result_df["momentum_top10_return"]).mean()
        ),
        "model_top10_vs_bottom10_win_rate": float(
            (result_df["model_top10_return"] > result_df["model_bottom10_return"]).mean()
        ),
        "model_top10_excess_return": model_top10_avg - universe_avg,
    }


def selection_score(metrics: dict, ranking: dict) -> float:
    return (
        100.0 * ranking["model_top10_vs_universe_win_rate"]
        + 250.0 * ranking["model_top10_excess_return"]
        + 20.0 * metrics["spearman"]
    )


def train_and_evaluate_model(
    model_key: str,
    horizon_days: int,
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> dict:
    print(f"Training model: {model_key}")

    model = build_model(model_key)

    X_train = train_df[FEATURE_COLUMNS]
    y_train = train_df["future_return"]

    X_valid = valid_df[FEATURE_COLUMNS]
    y_valid = valid_df["future_return"]

    X_test = test_df[FEATURE_COLUMNS]
    y_test = test_df["future_return"]

    model.fit(X_train, y_train)

    valid_pred = model.predict(X_valid)
    test_pred = model.predict(X_test)

    valid_metrics = calculate_metrics(y_valid.to_numpy(), valid_pred)
    test_metrics = calculate_metrics(y_test.to_numpy(), test_pred)

    diagnostic_df = test_df.copy()
    diagnostic_df["prediction"] = test_pred

    ranking_metrics = ranking_diagnostic(diagnostic_df)
    score = selection_score(test_metrics, ranking_metrics)

    print(
        f"{model_key}: "
        f"score={score:.4f}, "
        f"spearman={test_metrics['spearman']:.4f}, "
        f"top10={ranking_metrics['model_top10_avg_return']:.4f}, "
        f"universe={ranking_metrics['universe_avg_return']:.4f}, "
        f"win_rate={ranking_metrics['model_top10_vs_universe_win_rate']:.4f}"
    )

    return {
        "horizon_days": horizon_days,
        "model_key": model_key,
        "selection_score": score,
        **{f"valid_{key}": value for key, value in valid_metrics.items()},
        **{f"test_{key}": value for key, value in test_metrics.items()},
        **{f"ranking_{key}": value for key, value in ranking_metrics.items()},
    }


def train_production_model(
    model_key: str,
    horizon_days: int,
    full_df: pd.DataFrame,
    best_result: dict,
) -> Path:
    print(f"Retraining production model for {horizon_days}D using {model_key}")

    model = build_model(model_key)

    X_full = full_df[FEATURE_COLUMNS]
    y_full = full_df["future_return"]

    model.fit(X_full, y_full)

    project_root = get_project_root()
    production_dir = project_root / "ml" / "artifacts" / "production"
    production_dir.mkdir(parents=True, exist_ok=True)

    model_name = f"production_h{horizon_days}_{model_key}"
    model_path = production_dir / f"production_h{horizon_days}.joblib"

    bundle = {
        "model": model,
        "model_name": model_name,
        "model_key": model_key,
        "horizon_days": horizon_days,
        "feature_columns": FEATURE_COLUMNS,
        "train_end": TRAIN_END,
        "valid_end": VALID_END,
        "trained_on_rows": int(len(full_df)),
        "best_result": best_result,
    }

    joblib.dump(bundle, model_path)

    print(f"Saved production model: {model_path}")

    return model_path


def available_models(requested_models: list[str]) -> list[str]:
    result = []

    for model_key in requested_models:
        if model_key == "xgboost" and not has_package("xgboost"):
            print("Skipping xgboost: package not installed")
            continue

        if model_key == "lightgbm" and not has_package("lightgbm"):
            print("Skipping lightgbm: package not installed")
            continue

        if model_key == "catboost" and not has_package("catboost"):
            print("Skipping catboost: package not installed")
            continue

        result.append(model_key)

    return result


def run_tournament_for_horizon(
    horizon_days: int,
    model_keys: list[str],
) -> tuple[list[dict], dict]:
    print("")
    print("=" * 90)
    print(f"MODEL TOURNAMENT — HORIZON {horizon_days}D")
    print("=" * 90)

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

    results = []

    for model_key in model_keys:
        result = train_and_evaluate_model(
            model_key=model_key,
            horizon_days=horizon_days,
            train_df=train_df,
            valid_df=valid_df,
            test_df=test_df,
        )
        results.append(result)

    best_result = max(results, key=lambda row: row["selection_score"])
    best_model_key = best_result["model_key"]

    print("")
    print(f"Best model for {horizon_days}D: {best_model_key}")
    print(f"Best selection score: {best_result['selection_score']:.4f}")

    production_path = train_production_model(
        model_key=best_model_key,
        horizon_days=horizon_days,
        full_df=df,
        best_result=best_result,
    )

    best_result = {
        **best_result,
        "production_model_path": str(production_path),
    }

    return results, best_result


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--horizons",
        type=str,
        default=",".join(str(horizon) for horizon in DEFAULT_HORIZONS),
        help="Comma-separated horizons, for example: 1,5,10,30,60",
    )

    parser.add_argument(
        "--models",
        type=str,
        default="hist_gradient,extra_trees,xgboost,lightgbm,catboost",
        help="Comma-separated models: hist_gradient,extra_trees,xgboost,lightgbm,catboost",
    )

    args = parser.parse_args()

    horizons = parse_csv_ints(args.horizons)
    requested_models = parse_csv_strings(args.models)
    model_keys = available_models(requested_models)

    if not model_keys:
        raise ValueError("No models available to train.")

    print(f"Horizons: {horizons}")
    print(f"Models: {model_keys}")

    all_results = []
    registry_rows = []

    for horizon_days in horizons:
        horizon_results, best_result = run_tournament_for_horizon(
            horizon_days=horizon_days,
            model_keys=model_keys,
        )

        all_results.extend(horizon_results)
        registry_rows.append(best_result)

    project_root = get_project_root()
    production_dir = project_root / "ml" / "artifacts" / "production"
    production_dir.mkdir(parents=True, exist_ok=True)

    comparison_path = production_dir / "model_tournament_report.csv"
    registry_path = production_dir / "production_model_registry.csv"

    comparison_df = pd.DataFrame(all_results)
    registry_df = pd.DataFrame(registry_rows)

    comparison_df.to_csv(comparison_path, index=False)
    registry_df.to_csv(registry_path, index=False)

    print("")
    print("=" * 90)
    print("MODEL TOURNAMENT FINISHED")
    print("=" * 90)
    print(f"Comparison report: {comparison_path}")
    print(f"Production registry: {registry_path}")

    print("")
    print("Production models:")
    print(
        registry_df[
            [
                "horizon_days",
                "model_key",
                "selection_score",
                "test_spearman",
                "ranking_model_top10_avg_return",
                "ranking_universe_avg_return",
                "ranking_model_top10_vs_universe_win_rate",
                "production_model_path",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
