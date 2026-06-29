from sqlalchemy import text

from app.db.database import SessionLocal


DEFAULT_HORIZONS = [1, 5, 10, 20, 30, 60, 90, 120, 150, 180, 252]
UNIVERSE_NAME = "USA_TOP_100"


def calculate_horizon_targets_for_horizon(
    horizon_days: int,
    universe_name: str = UNIVERSE_NAME,
    limit_assets: int | None = None,
) -> int:
    limit_clause = ""
    params = {
        "horizon_days": horizon_days,
        "universe_name": universe_name,
    }

    if limit_assets is not None:
        limit_clause = "LIMIT :limit_assets"
        params["limit_assets"] = limit_assets

    query = text(
        f"""
        WITH selected_assets AS (
            SELECT id
            FROM assets
            WHERE universe_name = :universe_name
              AND is_active = TRUE
            ORDER BY universe_rank ASC
            {limit_clause}
        ),
        price_rows AS (
            SELECT
                mp.asset_id,
                mp.date,
                COALESCE(mp.adjusted_close, mp.close) AS current_price,
                LEAD(COALESCE(mp.adjusted_close, mp.close), :horizon_days)
                    OVER (
                        PARTITION BY mp.asset_id
                        ORDER BY mp.date ASC
                    ) AS future_price
            FROM market_prices_daily mp
            JOIN selected_assets sa
                ON sa.id = mp.asset_id
        ),
        target_rows AS (
            SELECT
                asset_id,
                date,
                :horizon_days AS horizon_days,
                CASE
                    WHEN current_price IS NULL THEN NULL
                    WHEN future_price IS NULL THEN NULL
                    WHEN current_price = 0 THEN NULL
                    ELSE (future_price / current_price) - 1.0
                END AS future_return,
                CASE
                    WHEN current_price IS NULL THEN NULL
                    WHEN future_price IS NULL THEN NULL
                    ELSE future_price > current_price
                END AS future_direction
            FROM price_rows
        )
        INSERT INTO targets_horizon_daily (
            asset_id,
            date,
            horizon_days,
            future_return,
            future_direction
        )
        SELECT
            asset_id,
            date,
            horizon_days,
            future_return,
            future_direction
        FROM target_rows
        ON CONFLICT (asset_id, date, horizon_days)
        DO UPDATE SET
            future_return = EXCLUDED.future_return,
            future_direction = EXCLUDED.future_direction,
            updated_at = NOW()
        """
    )

    with SessionLocal() as session:
        result = session.execute(query, params)
        session.commit()

        return result.rowcount or 0


def calculate_horizon_targets_for_active_assets(
    horizons: list[int] | None = None,
    universe_name: str = UNIVERSE_NAME,
    limit_assets: int | None = None,
) -> dict:
    horizons_to_run = horizons or DEFAULT_HORIZONS

    total_rows_updated = 0
    horizon_results = []

    for horizon_days in horizons_to_run:
        rows_updated = calculate_horizon_targets_for_horizon(
            horizon_days=horizon_days,
            universe_name=universe_name,
            limit_assets=limit_assets,
        )

        total_rows_updated += rows_updated

        horizon_results.append(
            {
                "horizon_days": horizon_days,
                "rows_updated": rows_updated,
            }
        )

        print(f"Horizon {horizon_days}d: {rows_updated} target rows updated")

    return {
        "universe_name": universe_name,
        "horizons": horizons_to_run,
        "total_rows_updated": total_rows_updated,
        "horizon_results": horizon_results,
    }