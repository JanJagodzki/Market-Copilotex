from app.services.asset_universe_service import update_usa_top_100_assets


def main() -> None:
    result = update_usa_top_100_assets()

    print("USA Top 100 asset universe updated.")
    print(f"Fetched: {result['fetched']}")
    print(f"Created: {result['created']}")
    print(f"Updated: {result['updated']}")


if __name__ == "__main__":
    main()