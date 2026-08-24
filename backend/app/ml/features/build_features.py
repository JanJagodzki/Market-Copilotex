import numpy as np
import pandas as pd


FEATURE_COLUMNS = [
    "return_1d",
    "return_5d",
    "return_20d",
    "volatility_5d",
    "volatility_20d",
    "sma_5_ratio",
    "sma_20_ratio",
    "sma_50_ratio",
    "ema_12_ratio",
    "ema_26_ratio",
    "rsi_14",
    "macd_ratio",
    "high_low_range",
    "open_close_return",
    "volume_change_1d",
]


def calculate_rsi(series, period=14):
    change = series.diff()

    gain = change.clip(lower=0)
    loss = -change.clip(upper=0)

    average_gain = gain.rolling(period).mean()
    average_loss = loss.rolling(period).mean()

    rs = average_gain / average_loss.replace(0, np.nan)

    return 100 - (100 / (1 + rs))


def build_features(data):
    df = data.copy()
    df = df.sort_values("date").reset_index(drop=True)

    df["adj_close"] = df["adjusted_close"].fillna(
        df["close"]
    )

    factor = (
        df["adj_close"] /
        df["close"].replace(0, np.nan)
    )

    df["adj_open"] = df["open"] * factor
    df["adj_high"] = df["high"] * factor
    df["adj_low"] = df["low"] * factor

    df["return_1d"] = df["adj_close"].pct_change(
        periods=1,
        fill_method=None,
    )

    df["return_5d"] = df["adj_close"].pct_change(
        periods=5,
        fill_method=None,
    )

    df["return_20d"] = df["adj_close"].pct_change(
        periods=20,
        fill_method=None,
    )

    log_return = np.log(
        df["adj_close"] /
        df["adj_close"].shift(1)
    )

    df["volatility_5d"] = (
        log_return.rolling(5).std()
    )

    df["volatility_20d"] = (
        log_return.rolling(20).std()
    )

    sma_5 = df["adj_close"].rolling(5).mean()
    sma_20 = df["adj_close"].rolling(20).mean()
    sma_50 = df["adj_close"].rolling(50).mean()

    df["sma_5_ratio"] = (
        df["adj_close"] / sma_5 - 1
    )

    df["sma_20_ratio"] = (
        df["adj_close"] / sma_20 - 1
    )

    df["sma_50_ratio"] = (
        df["adj_close"] / sma_50 - 1
    )

    ema_12 = df["adj_close"].ewm(
        span=12,
        adjust=False,
    ).mean()

    ema_26 = df["adj_close"].ewm(
        span=26,
        adjust=False,
    ).mean()

    df["ema_12_ratio"] = (
        df["adj_close"] / ema_12 - 1
    )

    df["ema_26_ratio"] = (
        df["adj_close"] / ema_26 - 1
    )

    df["rsi_14"] = calculate_rsi(
        df["adj_close"]
    )

    macd = ema_12 - ema_26

    df["macd_ratio"] = (
        macd / df["adj_close"]
    )

    df["high_low_range"] = (
        (df["adj_high"] - df["adj_low"]) /
        df["adj_close"]
    )

    df["open_close_return"] = (
        (df["adj_close"] - df["adj_open"]) /
        df["adj_open"]
    )

    df["volume_change_1d"] = (
        df["volume"].pct_change(
            fill_method=None
        )
    )

    future_close_1d = df["adj_close"].shift(-1)

    df["target_return_1d"] = (
        future_close_1d /
        df["adj_close"] - 1
    )

    direction_1d = pd.Series(
        pd.NA,
        index=df.index,
        dtype="Int64",
    )

    valid_1d = future_close_1d.notna()

    direction_1d.loc[valid_1d] = (
        df.loc[
            valid_1d,
            "target_return_1d",
        ] > 0
    ).astype(int)

    df["target_direction_1d"] = direction_1d

    future_close_5d = df["adj_close"].shift(-5)

    df["target_return_5d"] = (
        future_close_5d /
        df["adj_close"] - 1
    )

    direction_5d = pd.Series(
        pd.NA,
        index=df.index,
        dtype="Int64",
    )

    valid_5d = future_close_5d.notna()

    direction_5d.loc[valid_5d] = (
        df.loc[
            valid_5d,
            "target_return_5d",
        ] > 0
    ).astype(int)

    df["target_direction_5d"] = direction_5d

    df = df.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    return df
