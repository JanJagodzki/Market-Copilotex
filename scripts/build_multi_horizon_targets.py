from sqlalchemy import text

from backend.app.db.database import engine


QUERY = """
WITH prices AS (
    SELECT
        symbol_id,
        date,
        COALESCE(adjusted_close, close) AS price,

        LEAD(
            COALESCE(adjusted_close, close),
            1
        ) OVER (
            PARTITION BY symbol_id
            ORDER BY date
        ) AS price_1d,

        LEAD(
            COALESCE(adjusted_close, close),
            5
        ) OVER (
            PARTITION BY symbol_id
            ORDER BY date
        ) AS price_5d,

        LEAD(
            COALESCE(adjusted_close, close),
            20
        ) OVER (
            PARTITION BY symbol_id
            ORDER BY date
        ) AS price_20d,

        LEAD(
            COALESCE(adjusted_close, close),
            60
        ) OVER (
            PARTITION BY symbol_id
            ORDER BY date
        ) AS price_60d,

        LEAD(
            COALESCE(adjusted_close, close),
            120
        ) OVER (
            PARTITION BY symbol_id
            ORDER BY date
        ) AS price_120d,

        LEAD(
            COALESCE(adjusted_close, close),
            252
        ) OVER (
            PARTITION BY symbol_id
            ORDER BY date
        ) AS price_252d

    FROM daily_prices
)

UPDATE daily_features AS f

SET
    target_return_1d =
        prices.price_1d / prices.price - 1,

    target_direction_1d =
        CASE
            WHEN prices.price_1d IS NULL
                THEN NULL
            WHEN prices.price_1d > prices.price
                THEN 1
            ELSE 0
        END,

    target_return_5d =
        prices.price_5d / prices.price - 1,

    target_direction_5d =
        CASE
            WHEN prices.price_5d IS NULL
                THEN NULL
            WHEN prices.price_5d > prices.price
                THEN 1
            ELSE 0
        END,

    target_return_20d =
        prices.price_20d / prices.price - 1,

    target_direction_20d =
        CASE
            WHEN prices.price_20d IS NULL
                THEN NULL
            WHEN prices.price_20d > prices.price
                THEN 1
            ELSE 0
        END,

    target_return_60d =
        prices.price_60d / prices.price - 1,

    target_direction_60d =
        CASE
            WHEN prices.price_60d IS NULL
                THEN NULL
            WHEN prices.price_60d > prices.price
                THEN 1
            ELSE 0
        END,

    target_return_120d =
        prices.price_120d / prices.price - 1,

    target_direction_120d =
        CASE
            WHEN prices.price_120d IS NULL
                THEN NULL
            WHEN prices.price_120d > prices.price
                THEN 1
            ELSE 0
        END,

    target_return_252d =
        prices.price_252d / prices.price - 1,

    target_direction_252d =
        CASE
            WHEN prices.price_252d IS NULL
                THEN NULL
            WHEN prices.price_252d > prices.price
                THEN 1
            ELSE 0
        END

FROM prices

WHERE
    f.symbol_id = prices.symbol_id
    AND f.date = prices.date;
"""


def main():
    print("Building multi-horizon targets...")
    print()

    with engine.begin() as connection:
        result = connection.execute(
            text(QUERY)
        )

        print(
            f"Updated rows: {result.rowcount}"
        )

    print()
    print("Targets finished")


if __name__ == "__main__":
    main()
