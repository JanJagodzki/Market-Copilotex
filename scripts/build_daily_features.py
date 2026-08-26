import argparse
from datetime import timedelta

import pandas as pd
from sqlalchemy import func, select
from sqlalchemy.dialects.postgresql import insert

from backend.app.db.database import SessionLocal, engine
from backend.app.db.models import DailyFeature, DailyPrice, Symbol
from backend.app.ml.features.build_features import FEATURE_COLUMNS, build_features


TARGET_COLUMNS = [
    "target_return_1d",
    "target_direction_1d",
    "target_return_5d",
    "target_direction_5d",
]


def optional_float(value):
    if pd.isna(value):
        return None
    return float(value)


def optional_int(value):
    if pd.isna(value):
        return None
    return int(value)


def get_symbols(db, tickers=None, limit=None):
    query = db.query(Symbol).filter(Symbol.active.is_(True))

    if tickers:
        query = query.filter(Symbol.ticker.in_(tickers))

    symbols = query.order_by(Symbol.ticker).all()

    if limit is not None:
        symbols = symbols[:limit]

    return symbols


def get_last_feature_date(db, symbol_id):
    return db.query(func.max(DailyFeature.date)).filter(
        DailyFeature.symbol_id == symbol_id
    ).scalar()


def load_prices(symbol_id, start_date=None):
    query = select(
        DailyPrice.date,
        DailyPrice.open,
        DailyPrice.high,
        DailyPrice.low,
        DailyPrice.close,
        DailyPrice.adjusted_close,
        DailyPrice.volume,
    ).where(DailyPrice.symbol_id == symbol_id)

    if start_date is not None:
        query = query.where(DailyPrice.date >= start_date)

    query = query.order_by(DailyPrice.date)
    return pd.read_sql(query, engine)


def make_feature_row(symbol_id, row):
    result = {
        "symbol_id": symbol_id,
        "date": row["date"],
    }

    for column in FEATURE_COLUMNS:
        result[column] = optional_float(row[column])

    result["target_return_1d"] = optional_float(row["target_return_1d"])
    result["target_direction_1d"] = optional_int(row["target_direction_1d"])
    result["target_return_5d"] = optional_float(row["target_return_5d"])
    result["target_direction_5d"] = optional_int(row["target_direction_5d"])

    return result


def save_features(db, symbol_id, data):
    rows = [make_feature_row(symbol_id, row) for _, row in data.iterrows()]

    for start in range(0, len(rows), 1000):
        batch = rows[start:start + 1000]
        statement = insert(DailyFeature).values(batch)
        update_columns = FEATURE_COLUMNS + TARGET_COLUMNS

        statement = statement.on_conflict_do_update(
            constraint="uq_daily_features_symbol_date",
            set_={column: getattr(statement.excluded, column) for column in update_columns},
        )
        db.execute(statement)

    return len(rows)


def build_symbol_features(db, symbol, overlap_days):
    last_feature_date = get_last_feature_date(db, symbol.id)

    if last_feature_date is None:
        load_start = None
        save_start = None
    else:
        load_start = last_feature_date - timedelta(days=400)
        save_start = last_feature_date - timedelta(days=overlap_days + 10)

    prices = load_prices(symbol.id, load_start)
    if prices.empty:
        return 0, last_feature_date is None

    features = build_features(prices).dropna(subset=FEATURE_COLUMNS)

    if save_start is not None:
        dates = pd.to_datetime(features["date"]).dt.date
        features = features[dates >= save_start]

    if features.empty:
        return 0, last_feature_date is None

    count = save_features(db, symbol.id, features)
    return count, last_feature_date is None


def build_incremental_features(tickers=None, limit=None, overlap_days=5):
    if overlap_days < 0:
        raise ValueError("overlap_days cannot be negative")

    db = SessionLocal()
    result = {
        "symbols": 0,
        "rows": 0,
        "new_symbols": 0,
        "failed_symbols": 0,
    }

    try:
        symbols = get_symbols(db, tickers, limit)
        result["symbols"] = len(symbols)
        print(f"Symbols to update: {len(symbols)}")

        for number, symbol in enumerate(symbols, start=1):
            try:
                count, is_new = build_symbol_features(db, symbol, overlap_days)
                db.commit()

                result["rows"] += count
                result["new_symbols"] += int(is_new)
                print(f"[{number}/{len(symbols)}] {symbol.ticker}: {count} rows")
            except Exception as error:
                db.rollback()
                result["failed_symbols"] += 1
                print(f"[{number}/{len(symbols)}] {symbol.ticker}: {error}")

        return result
    finally:
        db.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--tickers", nargs="+")
    parser.add_argument("--overlap-days", type=int, default=5)
    args = parser.parse_args()

    result = build_incremental_features(
        tickers=args.tickers,
        limit=args.limit,
        overlap_days=args.overlap_days,
    )

    print()
    print(f"Saved feature rows: {result['rows']}")
    print(f"Failed symbols: {result['failed_symbols']}")


if __name__ == "__main__":
    main()
