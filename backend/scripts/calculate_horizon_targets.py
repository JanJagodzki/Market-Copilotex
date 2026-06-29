import argparse

from app.services.horizon_target_service import (
    DEFAULT_HORIZONS,
    calculate_horizon_targets_for_active_assets,
)


def parse_horizons(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--horizons",
        type=str,
        default=",".join(str(horizon) for horizon in DEFAULT_HORIZONS),
        help="Comma-separated horizons, for example: 1,5,10,30,60",
    )

    parser.add_argument(
        "--limit-assets",
        type=int,
        default=None,
        help="Limit number of active assets for testing",
    )

    args = parser.parse_args()

    horizons = parse_horizons(args.horizons)

    result = calculate_horizon_targets_for_active_assets(
        horizons=horizons,
        limit_assets=args.limit_assets,
    )

    print("Horizon target calculation finished.")
    print(f"Universe: {result['universe_name']}")
    print(f"Horizons: {result['horizons']}")
    print(f"Total rows updated: {result['total_rows_updated']}")


if __name__ == "__main__":
    main()