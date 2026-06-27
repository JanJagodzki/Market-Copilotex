from app.services.target_service import calculate_targets_for_active_assets


def main() -> None:
    result = calculate_targets_for_active_assets(
        universe_name="USA_TOP_100",
        limit=None,
    )

    print("Target calculation finished.")
    print(f"Assets processed: {result['assets_processed']}")
    print(f"Target rows updated: {result['target_rows_updated']}")
    print(f"Failed assets: {result['failed_assets']}")


if __name__ == "__main__":
    main()