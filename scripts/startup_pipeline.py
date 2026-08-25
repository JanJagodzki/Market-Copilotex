import argparse
import json
from datetime import datetime, timezone

from backend.app.data.yahoo_prices import (
    sync_missing_daily_prices,
)
from backend.app.db.database import (
    Base,
    SessionLocal,
    engine,
)
from backend.app.db.models import PipelineRun


def create_tables():
    Base.metadata.create_all(bind=engine)


def create_pipeline_run():
    db = SessionLocal()

    try:
        pipeline_run = PipelineRun(
            pipeline_name="startup_daily_catch_up",
            status="running",
        )

        db.add(pipeline_run)
        db.commit()
        db.refresh(pipeline_run)

        return pipeline_run.id

    finally:
        db.close()


def finish_pipeline_run(
    run_id,
    status,
    processed_rows,
    message,
):
    db = SessionLocal()

    try:
        pipeline_run = db.get(
            PipelineRun,
            run_id,
        )

        if pipeline_run is None:
            return

        pipeline_run.status = status
        pipeline_run.processed_rows = processed_rows
        pipeline_run.message = message
        pipeline_run.finished_at = datetime.now(
            timezone.utc
        )

        db.commit()

    finally:
        db.close()


def run_startup_pipeline(
    overlap_days=5,
    batch_size=25,
    limit=None,
):
    create_tables()
    run_id = create_pipeline_run()

    print("Starting MarketCopilotex catch-up")

    try:
        result = sync_missing_daily_prices(
            overlap_days=overlap_days,
            batch_size=batch_size,
            limit=limit,
        )

        message = json.dumps(
            result,
            ensure_ascii=False,
        )

        finish_pipeline_run(
            run_id=run_id,
            status="success",
            processed_rows=result["rows"],
            message=message,
        )

        print()
        print("Catch-up finished")
        print(
            f"Symbols: {result['symbols']}"
        )
        print(
            f"Processed rows: {result['rows']}"
        )
        print(
            f"Failed symbols: "
            f"{result['failed_symbols']}"
        )

        return result

    except Exception as error:
        finish_pipeline_run(
            run_id=run_id,
            status="failed",
            processed_rows=0,
            message=str(error),
        )

        raise


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--overlap-days",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=25,
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
    )

    args = parser.parse_args()

    run_startup_pipeline(
        overlap_days=args.overlap_days,
        batch_size=args.batch_size,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()
