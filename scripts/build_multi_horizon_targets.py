import argparse

from sqlalchemy import text

from backend.app.db.database import SessionLocal
from backend.app.db.models import Symbol


TARGET_QUERY = """
WITH source_prices AS (
    {source_query}
),
prices AS (
    SELECT
        symbol_id,
        date,
        price,
        ROW_NUMBER() OVER (ORDER BY date DESC) AS recent_number,
        LEAD(price, 1) OVER (ORDER BY date) AS price_1d,
        LEAD(price, 5) OVER (ORDER BY date) AS price_5d,
        LEAD(price, 20) OVER (ORDER BY date) AS price_20d,
        LEAD(price, 60) OVER (ORDER BY date) AS price_60d,
        LEAD(price, 120) OVER (ORDER BY date) AS price_120d,
        LEAD(price, 252) OVER (ORDER BY date) AS price_252d
    FROM source_prices
)
UPDATE daily_features AS feature
SET
    target_return_1d = prices.price_1d / NULLIF(prices.price, 0) - 1,
    target_direction_1d = CASE
        WHEN prices.price_1d IS NULL THEN NULL
        WHEN prices.price_1d > prices.price THEN 1
        ELSE 0
    END,
    target_return_5d = prices.price_5d / NULLIF(prices.price, 0) - 1,
    target_direction_5d = CASE
        WHEN prices.price_5d IS NULL THEN NULL
        WHEN prices.price_5d > prices.price THEN 1
        ELSE 0
    END,
    target_return_20d = prices.price_20d / NULLIF(prices.price, 0) - 1,
    target_direction_20d = CASE
        WHEN prices.price_20d IS NULL THEN NULL
        WHEN prices.price_20d > prices.price THEN 1
        ELSE 0
    END,
    target_return_60d = prices.price_60d / NULLIF(prices.price, 0) - 1,
    target_direction_60d = CASE
        WHEN prices.price_60d IS NULL THEN NULL
        WHEN prices.price_60d > prices.price THEN 1
        ELSE 0
    END,
    target_return_120d = prices.price_120d / NULLIF(prices.price, 0) - 1,
    target_direction_120d = CASE
        WHEN prices.price_120d IS NULL THEN NULL
        WHEN prices.price_120d > prices.price THEN 1
        ELSE 0
    END,
    target_return_252d = prices.price_252d / NULLIF(prices.price, 0) - 1,
    target_direction_252d = CASE
        WHEN prices.price_252d IS NULL THEN NULL
        WHEN prices.price_252d > prices.price THEN 1
        ELSE 0
    END
FROM prices
WHERE feature.symbol_id = prices.symbol_id
  AND feature.date = prices.date
  {update_condition};
"""


RECENT_PRICES = """
SELECT
    symbol_id,
    date,
    COALESCE(adjusted_close, close) AS price
FROM daily_prices
WHERE symbol_id = :symbol_id
ORDER BY date DESC
LIMIT :recent_rows
"""


ALL_PRICES = """
SELECT
    symbol_id,
    date,
    COALESCE(adjusted_close, close) AS price
FROM daily_prices
WHERE symbol_id = :symbol_id
"""


def get_symbols(db, tickers=None, limit=None):
    query = db.query(Symbol).filter(Symbol.active.is_(True))

    if tickers:
        query = query.filter(Symbol.ticker.in_(tickers))

    symbols = query.order_by(Symbol.ticker).all()

    if limit is not None:
        symbols = symbols[:limit]

    return symbols


def has_old_targets(db, symbol_id):
    query = text("""
        SELECT EXISTS (
            SELECT 1
            FROM daily_features
            WHERE symbol_id = :symbol_id
              AND target_direction_20d IS NOT NULL
        )
    """)

    return bool(db.execute(query, {"symbol_id": symbol_id}).scalar())


def build_target_query(full_history):
    if full_history:
        source_query = ALL_PRICES
        update_condition = ""
    else:
        source_query = RECENT_PRICES
        update_condition = "AND prices.recent_number <= :recent_rows"

    return text(TARGET_QUERY.format(
        source_query=source_query,
        update_condition=update_condition,
    ))


def update_symbol_targets(db, symbol_id, recent_rows):
    full_history = not has_old_targets(db, symbol_id)
    query = build_target_query(full_history)

    params = {"symbol_id": symbol_id}
    if not full_history:
        params["recent_rows"] = recent_rows

    result = db.execute(query, params)

    return max(result.rowcount, 0), full_history


def update_targets(tickers=None, limit=None, recent_rows=320):
    if recent_rows < 260:
        raise ValueError("recent_rows must be at least 260")

    db = SessionLocal()
    result = {
        "symbols": 0,
        "rows": 0,
        "full_history_symbols": 0,
        "failed_symbols": 0,
    }

    try:
        symbols = get_symbols(db, tickers, limit)
        result["symbols"] = len(symbols)
        print(f"Symbols to update: {len(symbols)}")

        for number, symbol in enumerate(symbols, start=1):
            try:
                count, full_history = update_symbol_targets(
                    db, symbol.id, recent_rows
                )
                db.commit()

                result["rows"] += count
                result["full_history_symbols"] += int(full_history)
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
    parser.add_argument("--recent-rows", type=int, default=320)
    args = parser.parse_args()

    result = update_targets(
        tickers=args.tickers,
        limit=args.limit,
        recent_rows=args.recent_rows,
    )

    print()
    print(f"Updated target rows: {result['rows']}")
    print(f"Failed symbols: {result['failed_symbols']}")


if __name__ == "__main__":
    main()
