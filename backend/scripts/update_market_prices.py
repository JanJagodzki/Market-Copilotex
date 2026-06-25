from app.services.market_price_service import update_market_prices_for_active_assets


def main() -> None:
    result = update_market_prices_for_active_assets(
        universe_name="USA_TOP_100",
        period="max",
        limit=None,
    )

    print("Market prices update finished.")
    print(f"Assets processed: {result['assets_processed']}")
    print(f"Price rows updated: {result['price_rows_updated']}")
    print(f"Failed assets: {result['failed_assets']}")


if __name__ == "__main__":
    main()