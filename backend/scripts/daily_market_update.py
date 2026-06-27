from sqlalchemy import exists, select

from app.db.database import SessionLocal
from app.models.asset import Asset
from app.models.market_price_daily import MarketPriceDaily
from app.services.asset_universe_service import update_usa_top_100_assets
from app.services.feature_service import calculate_features_for_active_assets
from app.services.market_price_service import (
    update_market_prices_for_active_assets,
    upsert_prices_for_asset,
)
from app.services.prediction_service import generate_latest_predictions


UNIVERSE_NAME = "USA_TOP_100"


def get_active_assets_without_prices() -> list[tuple[int, str]]:
    with SessionLocal() as session:
        query = (
            select(Asset.id, Asset.symbol)
            .where(Asset.universe_name == UNIVERSE_NAME)
            .where(Asset.is_active.is_(True))
            .where(~exists().where(MarketPriceDaily.asset_id == Asset.id))
            .order_by(Asset.universe_rank.asc())
        )

        return list(session.execute(query).all())


def backfill_new_assets_without_history() -> dict[str, int]:
    assets_without_prices = get_active_assets_without_prices()

    processed = 0
    rows_updated = 0
    failed = 0

    for asset_id, symbol in assets_without_prices:
        processed += 1

        try:
            updated_rows = upsert_prices_for_asset(
                asset_id=asset_id,
                symbol=symbol,
                period="max",
            )
            rows_updated += updated_rows
        except Exception as error:
            failed += 1
            print(f"Failed full backfill for new asset {symbol}: {error}")

    return {
        "new_assets_processed": processed,
        "new_assets_rows_updated": rows_updated,
        "new_assets_failed": failed,
    }


def main() -> None:
    print("Starting daily market update...")

    universe_result = update_usa_top_100_assets()

    print("Asset universe updated.")
    print(f"Universe fetched: {universe_result['fetched']}")
    print(f"Universe created: {universe_result['created']}")
    print(f"Universe updated: {universe_result['updated']}")

    new_assets_result = backfill_new_assets_without_history()

    print("New assets history backfill finished.")
    print(f"New assets processed: {new_assets_result['new_assets_processed']}")
    print(f"New assets rows updated: {new_assets_result['new_assets_rows_updated']}")
    print(f"New assets failed: {new_assets_result['new_assets_failed']}")

    daily_prices_result = update_market_prices_for_active_assets(
        universe_name=UNIVERSE_NAME,
        period="1mo",
        limit=None,
    )

    print("Daily prices update finished.")
    print(f"Assets processed: {daily_prices_result['assets_processed']}")
    print(f"Price rows updated: {daily_prices_result['price_rows_updated']}")
    print(f"Failed assets: {daily_prices_result['failed_assets']}")

    features_result = calculate_features_for_active_assets(
        universe_name=UNIVERSE_NAME,
        limit=None,
    )

    print("Daily features calculation finished.")
    print(f"Assets processed: {features_result['assets_processed']}")
    print(f"Feature rows updated: {features_result['feature_rows_updated']}")
    print(f"Failed assets: {features_result['failed_assets']}")

    predictions_result = generate_latest_predictions()

    print("Daily predictions generation finished.")
    print(f"Prediction date: {predictions_result['prediction_date']}")
    print(f"Rows updated: {predictions_result['rows_updated']}")
    print(f"Model name: {predictions_result['model_name']}")
    print(f"Horizon days: {predictions_result['horizon_days']}")

    print("Daily market update completed.")


if __name__ == "__main__":
    main()