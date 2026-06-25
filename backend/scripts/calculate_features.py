from app.services.feature_service import calculate_features_for_active_assets


def main() -> None:
    result = calculate_features_for_active_assets(
        universe_name="USA_TOP_100",
        limit=None,
    )

    print("Feature calculation finished.")
    print(f"Assets processed: {result['assets_processed']}")
    print(f"Feature rows updated: {result['feature_rows_updated']}")
    print(f"Failed assets: {result['failed_assets']}")


if __name__ == "__main__":
    main()