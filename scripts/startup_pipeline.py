import argparse
import json
from datetime import datetime, timezone

from backend.app.data.yahoo_prices import sync_missing_daily_prices
from backend.app.db.database import Base, SessionLocal, engine
from backend.app.db.models import PipelineRun
from scripts.build_daily_features import build_incremental_features
from scripts.build_multi_horizon_targets import update_targets


def create_tables():
    Base.metadata.create_all(bind=engine)


def create_pipeline_run():
    db = SessionLocal()

    try:
        pipeline_run = PipelineRun(
            pipeline_name="startup_update",
            status="running",
        )
        db.add(pipeline_run)
        db.commit()
        db.refresh(pipeline_run)
        return pipeline_run.id
    finally:
        db.close()


def finish_pipeline_run(run_id, status, processed_rows, message):
    db = SessionLocal()

    try:
        pipeline_run = db.get(PipelineRun, run_id)
        if pipeline_run is None:
            return

        pipeline_run.status = status
        pipeline_run.processed_rows = processed_rows
        pipeline_run.message = message
        pipeline_run.finished_at = datetime.now(timezone.utc)
        db.commit()
    finally:
        db.close()


def run_startup_pipeline(overlap_days=5, batch_size=25, limit=None):
    create_tables()
    run_id = create_pipeline_run()
    print("Starting MarketCopilotex update")

    try:
        print("\n1/3 Updating daily prices")
        prices = sync_missing_daily_prices(
            overlap_days=overlap_days,
            batch_size=batch_size,
            limit=limit,
        )

        print("\n2/3 Updating daily features")
        features = build_incremental_features(
            limit=limit,
            overlap_days=overlap_days,
        )

        print("\n3/3 Updating prediction targets")
        targets = update_targets(limit=limit)

        result = {
            "prices": prices,
            "features": features,
            "targets": targets,
        }

        processed_rows = prices["rows"] + features["rows"] + targets["rows"]
        failed_symbols = (
            prices["failed_symbols"]
            + features["failed_symbols"]
            + targets["failed_symbols"]
        )
        status = "success" if failed_symbols == 0 else "partial"

        finish_pipeline_run(
            run_id,
            status,
            processed_rows,
            json.dumps(result, ensure_ascii=False),
        )

        print("\nStartup update finished")
        print(f"Price rows: {prices['rows']}")
        print(f"Skipped price symbols: {prices['skipped_symbols']}")
        print(f"Feature rows: {features['rows']}")
        print(f"Target rows: {targets['rows']}")
        print(f"Failed symbols: {failed_symbols}")
        return result

    except Exception as error:
        finish_pipeline_run(run_id, "failed", 0, str(error))
        raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--overlap-days", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=25)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    run_startup_pipeline(
        overlap_days=args.overlap_days,
        batch_size=args.batch_size,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()
