from sqlalchemy import text

from app.db.database import SessionLocal


def print_section(title: str) -> None:
    print()
    print("=" * 80)
    print(title)
    print("=" * 80)


def print_rows(rows) -> None:
    for row in rows:
        print(dict(row._mapping))


def main() -> None:
    with SessionLocal() as session:
        print_section("BASIC COUNTS")

        queries = {
            "active_assets": """
                SELECT COUNT(*) 
                FROM assets 
                WHERE universe_name = 'USA_TOP_100' AND is_active = TRUE
            """,
            "prices_daily": "SELECT COUNT(*) FROM market_prices_daily",
            "features_daily": "SELECT COUNT(*) FROM features_daily",
            "targets_daily": "SELECT COUNT(*) FROM targets_daily",
        }

        for name, query in queries.items():
            value = session.execute(text(query)).scalar()
            print(f"{name}: {value}")

        print_section("DATE RANGES")

        rows = session.execute(
            text(
                """
                SELECT 
                    'prices_daily' AS table_name,
                    MIN(date) AS first_date,
                    MAX(date) AS last_date
                FROM market_prices_daily

                UNION ALL

                SELECT 
                    'features_daily' AS table_name,
                    MIN(date) AS first_date,
                    MAX(date) AS last_date
                FROM features_daily

                UNION ALL

                SELECT 
                    'targets_daily' AS table_name,
                    MIN(date) AS first_date,
                    MAX(date) AS last_date
                FROM targets_daily
                """
            )
        ).all()
        print_rows(rows)

        print_section("DUPLICATE CHECKS")

        duplicate_queries = {
            "price_duplicates": """
                SELECT COUNT(*) FROM (
                    SELECT asset_id, date, COUNT(*)
                    FROM market_prices_daily
                    GROUP BY asset_id, date
                    HAVING COUNT(*) > 1
                ) x
            """,
            "feature_duplicates": """
                SELECT COUNT(*) FROM (
                    SELECT asset_id, date, COUNT(*)
                    FROM features_daily
                    GROUP BY asset_id, date
                    HAVING COUNT(*) > 1
                ) x
            """,
            "target_duplicates": """
                SELECT COUNT(*) FROM (
                    SELECT asset_id, date, COUNT(*)
                    FROM targets_daily
                    GROUP BY asset_id, date
                    HAVING COUNT(*) > 1
                ) x
            """,
        }

        for name, query in duplicate_queries.items():
            value = session.execute(text(query)).scalar()
            print(f"{name}: {value}")

        print_section("30D TARGET AVAILABILITY")

        rows = session.execute(
            text(
                """
                SELECT
                    COUNT(*) AS total_targets,
                    COUNT(future_return_30d) AS non_null_future_return_30d,
                    COUNT(*) - COUNT(future_return_30d) AS null_future_return_30d
                FROM targets_daily
                """
            )
        ).all()
        print_rows(rows)

        print_section("30D UP/DOWN BALANCE")

        rows = session.execute(
            text(
                """
                SELECT 
                    future_direction_30d,
                    COUNT(*) AS rows_count,
                    ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 2) AS percent
                FROM targets_daily
                WHERE future_direction_30d IS NOT NULL
                GROUP BY future_direction_30d
                ORDER BY future_direction_30d
                """
            )
        ).all()
        print_rows(rows)

        print_section("30D RETURN DISTRIBUTION")

        rows = session.execute(
            text(
                """
                SELECT
                    AVG(future_return_30d) AS avg_30d_return,
                    MIN(future_return_30d) AS min_30d_return,
                    MAX(future_return_30d) AS max_30d_return,
                    percentile_cont(0.01) WITHIN GROUP (ORDER BY future_return_30d) AS p01,
                    percentile_cont(0.05) WITHIN GROUP (ORDER BY future_return_30d) AS p05,
                    percentile_cont(0.50) WITHIN GROUP (ORDER BY future_return_30d) AS median,
                    percentile_cont(0.95) WITHIN GROUP (ORDER BY future_return_30d) AS p95,
                    percentile_cont(0.99) WITHIN GROUP (ORDER BY future_return_30d) AS p99
                FROM targets_daily
                WHERE future_return_30d IS NOT NULL
                """
            )
        ).all()
        print_rows(rows)

        print_section("TRAINING ROWS FOR 30D MODEL")

        rows = session.execute(
            text(
                """
                SELECT COUNT(*) AS usable_training_rows_30d
                FROM features_daily f
                JOIN targets_daily t
                    ON t.asset_id = f.asset_id
                    AND t.date = f.date
                WHERE t.future_return_30d IS NOT NULL
                  AND f.log_return IS NOT NULL
                  AND f.volatility_30d IS NOT NULL
                  AND f.sma_200 IS NOT NULL
                  AND f.momentum_30d IS NOT NULL
                  AND f.drawdown_252d IS NOT NULL
                """
            )
        ).all()
        print_rows(rows)

        print_section("ASSETS WITH SHORT HISTORY")

        rows = session.execute(
            text(
                """
                SELECT 
                    a.symbol,
                    COUNT(f.id) AS feature_rows,
                    MIN(f.date) AS first_date,
                    MAX(f.date) AS last_date
                FROM assets a
                LEFT JOIN features_daily f ON f.asset_id = a.id
                WHERE a.universe_name = 'USA_TOP_100'
                  AND a.is_active = TRUE
                GROUP BY a.symbol
                HAVING COUNT(f.id) < 252
                ORDER BY feature_rows ASC
                """
            )
        ).all()
        print_rows(rows)

        print_section("BIGGEST 30D RETURN OUTLIERS")

        rows = session.execute(
            text(
                """
                SELECT 
                    a.symbol,
                    t.date,
                    t.future_return_30d
                FROM targets_daily t
                JOIN assets a ON a.id = t.asset_id
                WHERE t.future_return_30d IS NOT NULL
                ORDER BY ABS(t.future_return_30d) DESC
                LIMIT 20
                """
            )
        ).all()
        print_rows(rows)


if __name__ == "__main__":
    main()