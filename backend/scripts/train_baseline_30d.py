from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error, r2_score
from sqlalchemy import text

from app.db.database import SessionLocal


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

TARGET_COLUMN = "future_return_30d"

TRAIN_END = "2021-01-01"
VALID_END = "2024-01-01"


def load_dataset() -> pd.DataFrame:
    query = text(
        """
        WITH asset_counts AS (
            SELECT asset_id, COUNT(*) AS rows_count
            FROM market_prices_daily
            GROUP BY asset_id
        )
        SELECT
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
            f.drawdown_252d,
            t.future_return_30d,
            t.future_direction_30d
        FROM features_daily f
        JOIN targets_daily t
            ON t.asset_id = f.asset_id
            AND t.date = f.date
        JOIN assets a
            ON a.id = f.asset_id
        JOIN asset_counts ac
            ON ac.asset_id = f.asset_id
        WHERE t.future_return_30d IS NOT NULL
          AND ac.rows_count >= 500
          AND f.daily_return IS NOT NULL
          AND f.log_return IS NOT NULL
          AND f.volume_change IS NOT NULL
          AND f.volatility_7d IS NOT NULL
          AND f.volatility_30d IS NOT NULL
          AND f.distance_from_sma_20 IS NOT NULL
          AND f.distance_from_sma_200 IS NOT NULL
          AND f.momentum_7d IS NOT NULL
          AND f.momentum_30d IS NOT NULL
          AND f.momentum_90d IS NOT NULL
          AND f.drawdown_252d IS NOT NULL
        ORDER BY f.date ASC
        """
    )

    with SessionLocal() as session:
        df = pd.read_sql_query(query, session.connection())

    df["date"] = pd.to_datetime(df["date"])
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=FEATURE_COLUMNS + [TARGET_COLUMN])

    return df


def evaluate_regression(name: str, y_true: pd.Series, y_pred: np.ndarray) -> None:
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)

    direction_true = (y_true > 0).astype(int)
    direction_pred = (y_pred > 0).astype(int)

    direction_accuracy = accuracy_score(direction_true, direction_pred)
    always_up_accuracy = direction_true.mean()

    pearson_corr = np.corrcoef(y_true, y_pred)[0, 1]
    spearman_corr = pd.Series(y_pred).corr(pd.Series(y_true).reset_index(drop=True), method="spearman")

    print()
    print("=" * 80)
    print(name)
    print("=" * 80)
    print(f"MAE: {mae:.6f}")
    print(f"RMSE: {rmse:.6f}")
    print(f"R2: {r2:.6f}")
    print(f"Direction accuracy: {direction_accuracy:.4f}")
    print(f"Always UP baseline accuracy: {always_up_accuracy:.4f}")
    print(f"Pearson correlation: {pearson_corr:.6f}")
    print(f"Spearman correlation: {spearman_corr:.6f}")


def evaluate_top10_ranking(test_df: pd.DataFrame) -> None:
    rows = []

    for date, group in test_df.groupby("date"):
        if len(group) < 30:
            continue

        top_model = group.nlargest(10, "prediction")[TARGET_COLUMN].mean()
        bottom_model = group.nsmallest(10, "prediction")[TARGET_COLUMN].mean()
        universe_avg = group[TARGET_COLUMN].mean()
        top_momentum = group.nlargest(10, "momentum_30d")[TARGET_COLUMN].mean()

        rows.append(
            {
                "date": date,
                "model_top10": top_model,
                "model_bottom10": bottom_model,
                "universe_avg": universe_avg,
                "momentum_top10": top_momentum,
            }
        )

    result = pd.DataFrame(rows)

    print()
    print("=" * 80)
    print("TOP 10 RANKING DIAGNOSTIC — TEST SET")
    print("=" * 80)

    if result.empty:
        print("Not enough dates for ranking diagnostic.")
        return

    print(f"Dates evaluated: {len(result)}")
    print(f"Model Top 10 avg 30d return: {result['model_top10'].mean():.6f}")
    print(f"Model Bottom 10 avg 30d return: {result['model_bottom10'].mean():.6f}")
    print(f"Universe avg 30d return: {result['universe_avg'].mean():.6f}")
    print(f"Momentum Top 10 avg 30d return: {result['momentum_top10'].mean():.6f}")

    model_vs_universe_win_rate = (result["model_top10"] > result["universe_avg"]).mean()
    model_vs_momentum_win_rate = (result["model_top10"] > result["momentum_top10"]).mean()
    top_vs_bottom_win_rate = (result["model_top10"] > result["model_bottom10"]).mean()

    print(f"Model Top 10 > Universe win rate: {model_vs_universe_win_rate:.4f}")
    print(f"Model Top 10 > Momentum Top 10 win rate: {model_vs_momentum_win_rate:.4f}")
    print(f"Model Top 10 > Model Bottom 10 win rate: {top_vs_bottom_win_rate:.4f}")


def main() -> None:
    print("Loading dataset...")
    df = load_dataset()

    print(f"Rows loaded: {len(df)}")
    print(f"Date range: {df['date'].min().date()} -> {df['date'].max().date()}")
    print(f"Symbols: {df['symbol'].nunique()}")

    train_df = df[df["date"] < TRAIN_END].copy()
    valid_df = df[(df["date"] >= TRAIN_END) & (df["date"] < VALID_END)].copy()
    test_df = df[df["date"] >= VALID_END].copy()

    print()
    print("Split:")
    print(f"Train rows: {len(train_df)}")
    print(f"Valid rows: {len(valid_df)}")
    print(f"Test rows: {len(test_df)}")

    x_train = train_df[FEATURE_COLUMNS]
    y_train = train_df[TARGET_COLUMN]

    x_valid = valid_df[FEATURE_COLUMNS]
    y_valid = valid_df[TARGET_COLUMN]

    x_test = test_df[FEATURE_COLUMNS]
    y_test = test_df[TARGET_COLUMN]

    lower = y_train.quantile(0.01)
    upper = y_train.quantile(0.99)
    y_train_clipped = y_train.clip(lower, upper)

    print()
    print("Training HistGradientBoostingRegressor baseline...")

    model = HistGradientBoostingRegressor(
        max_iter=200,
        learning_rate=0.05,
        max_leaf_nodes=31,
        l2_regularization=0.01,
        random_state=42,
    )

    model.fit(x_train, y_train_clipped)

    valid_predictions = model.predict(x_valid)
    test_predictions = model.predict(x_test)

    evaluate_regression("VALIDATION METRICS", y_valid, valid_predictions)
    evaluate_regression("TEST METRICS", y_test, test_predictions)

    test_df["prediction"] = test_predictions
    evaluate_top10_ranking(test_df)

    project_root = Path(__file__).resolve().parents[2]
    artifacts_dir = project_root / "ml" / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    model_path = artifacts_dir / "baseline_30d_hgbr.joblib"
    predictions_path = artifacts_dir / "baseline_30d_test_predictions.csv"

    joblib.dump(
        {
            "model": model,
            "feature_columns": FEATURE_COLUMNS,
            "target_column": TARGET_COLUMN,
            "train_end": TRAIN_END,
            "valid_end": VALID_END,
        },
        model_path,
    )

    test_df[["symbol", "date", TARGET_COLUMN, "prediction", "momentum_30d"]].to_csv(
        predictions_path,
        index=False,
    )

    print()
    print(f"Saved model: {model_path}")
    print(f"Saved test predictions: {predictions_path}")


if __name__ == "__main__":
    main()