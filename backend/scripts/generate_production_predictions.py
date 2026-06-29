import argparse

from app.services.horizon_target_service import DEFAULT_HORIZONS
from app.services.production_prediction_service import generate_all_production_predictions


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

    args = parser.parse_args()

    result = generate_all_production_predictions(
        horizons=parse_horizons(args.horizons),
    )

    print("")
    print("Production prediction generation finished.")
    print(f"Horizons: {result['horizons']}")
    print(f"Total rows updated: {result['total_rows_updated']}")


if __name__ == "__main__":
    main()
