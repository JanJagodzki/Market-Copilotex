from pathlib import Path

import numpy as np
import pandas as pd


TOP_N = 10
HOLDING_PERIOD_DAYS = 30
RANDOM_SIMULATIONS = 200
TRANSACTION_COST_PER_PERIOD = 0.001


def max_drawdown(equity_curve: pd.Series) -> float:
    running_max = equity_curve.cummax()
    drawdown = (equity_curve / running_max) - 1
    return float(drawdown.min())


def summarize_strategy(
    name: str,
    returns: pd.Series,
    benchmark_returns: pd.Series | None = None,
) -> dict:
    equity = (1 + returns).cumprod()

    summary = {
        "strategy": name,
        "periods": len(returns),
        "avg_period_return": returns.mean(),
        "median_period_return": returns.median(),
        "positive_period_rate": (returns > 0).mean(),
        "cumulative_return": equity.iloc[-1] - 1,
        "max_drawdown": max_drawdown(equity),
        "period_volatility": returns.std(),
        "sharpe_like": (
            (returns.mean() / returns.std()) * np.sqrt(252 / HOLDING_PERIOD_DAYS)
            if returns.std() != 0
            else np.nan
        ),
    }

    if benchmark_returns is not None:
        summary["win_rate_vs_benchmark"] = (returns > benchmark_returns).mean()
        summary["avg_excess_return"] = (returns - benchmark_returns).mean()

    return summary


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]
    predictions_path = project_root / "ml" / "artifacts" / "baseline_30d_test_predictions.csv"

    if not predictions_path.exists():
        raise FileNotFoundError(
            f"Missing predictions file: {predictions_path}. "
            "Run: python -m scripts.train_baseline_30d"
        )

    df = pd.read_csv(predictions_path)
    df["date"] = pd.to_datetime(df["date"])

    target_column = "future_return_30d"

    df = df.dropna(
        subset=[
            "prediction",
            "momentum_30d",
            target_column,
        ]
    )

    unique_dates = sorted(df["date"].unique())
    rebalance_dates = unique_dates[::HOLDING_PERIOD_DAYS]

    rows = []
    rng = np.random.default_rng(seed=42)

    for date in rebalance_dates:
        group = df[df["date"] == date].copy()

        if len(group) < TOP_N * 3:
            continue

        model_top = group.nlargest(TOP_N, "prediction")
        model_bottom = group.nsmallest(TOP_N, "prediction")
        momentum_top = group.nlargest(TOP_N, "momentum_30d")

        random_returns = []

        for _ in range(RANDOM_SIMULATIONS):
            random_top = group.sample(
                n=TOP_N,
                replace=False,
                random_state=int(rng.integers(0, 1_000_000)),
            )
            random_returns.append(random_top[target_column].mean())

        rows.append(
            {
                "date": date,
                "model_top10": model_top[target_column].mean()
                - TRANSACTION_COST_PER_PERIOD,
                "model_bottom10": model_bottom[target_column].mean()
                - TRANSACTION_COST_PER_PERIOD,
                "momentum_top10": momentum_top[target_column].mean()
                - TRANSACTION_COST_PER_PERIOD,
                "universe_equal": group[target_column].mean()
                - TRANSACTION_COST_PER_PERIOD,
                "random_top10_avg": float(np.mean(random_returns))
                - TRANSACTION_COST_PER_PERIOD,
            }
        )

    results = pd.DataFrame(rows)

    if results.empty:
        print("No backtest rows generated.")
        return

    print()
    print("=" * 80)
    print("NON-OVERLAPPING 30D BACKTEST")
    print("=" * 80)
    print(f"Start date: {results['date'].min().date()}")
    print(f"End date: {results['date'].max().date()}")
    print(f"Periods: {len(results)}")
    print(f"Top N: {TOP_N}")
    print(f"Holding period: {HOLDING_PERIOD_DAYS} sessions")
    print(f"Transaction cost per period: {TRANSACTION_COST_PER_PERIOD:.4f}")

    strategies = [
        "model_top10",
        "momentum_top10",
        "universe_equal",
        "random_top10_avg",
        "model_bottom10",
    ]

    summaries = []
    benchmark = results["universe_equal"]

    for strategy in strategies:
        summaries.append(
            summarize_strategy(
                name=strategy,
                returns=results[strategy],
                benchmark_returns=benchmark if strategy != "universe_equal" else None,
            )
        )

    summary_df = pd.DataFrame(summaries)

    print()
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)

    display_columns = [
        "strategy",
        "periods",
        "avg_period_return",
        "median_period_return",
        "positive_period_rate",
        "cumulative_return",
        "max_drawdown",
        "period_volatility",
        "sharpe_like",
    ]

    optional_columns = [
        "win_rate_vs_benchmark",
        "avg_excess_return",
    ]

    display_columns += [
        column for column in optional_columns if column in summary_df.columns
    ]

    print(summary_df[display_columns].to_string(index=False))

    print()
    print("=" * 80)
    print("PERIOD RETURNS")
    print("=" * 80)
    print(results.to_string(index=False))

    output_path = project_root / "ml" / "artifacts" / "baseline_30d_backtest_results.csv"
    results.to_csv(output_path, index=False)

    print()
    print(f"Saved backtest results: {output_path}")


if __name__ == "__main__":
    main()
