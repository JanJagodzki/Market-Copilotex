from app.services.prediction_service import generate_latest_predictions


def main() -> None:
    result = generate_latest_predictions()

    print("Prediction generation finished.")
    print(f"Prediction date: {result['prediction_date']}")
    print(f"Rows updated: {result['rows_updated']}")
    print(f"Model name: {result['model_name']}")
    print(f"Horizon days: {result['horizon_days']}")


if __name__ == "__main__":
    main()