import asyncio
from datetime import datetime, timezone

from backend.app.data.yahoo_intraday import (
    sync_intraday_prices,
)
from backend.app.db.database import SessionLocal
from backend.app.db.models import Symbol, WatchlistItem


SYNC_INTERVAL_SECONDS = 20 * 60

SYNC_STATUS = {
    "running": False,
    "interval_minutes": 20,
    "last_started": None,
    "last_finished": None,
    "last_rows": 0,
    "last_error": None,
    "message": "Waiting for the first update",
}


def utc_now():
    return datetime.now(
        timezone.utc
    ).isoformat()


def get_watchlist_tickers():
    db = SessionLocal()

    try:
        rows = (
            db.query(Symbol.ticker)
            .join(
                WatchlistItem,
                WatchlistItem.symbol_id
                == Symbol.id,
            )
            .filter(Symbol.active.is_(True))
            .order_by(Symbol.ticker)
            .all()
        )

        return [
            row[0]
            for row in rows
        ]
    finally:
        db.close()


def get_watchlist_sync_status():
    return dict(SYNC_STATUS)


async def sync_watchlist_once():
    tickers = await asyncio.to_thread(
        get_watchlist_tickers
    )

    if not tickers:
        SYNC_STATUS.update(
            {
                "running": False,
                "last_error": None,
                "message": "Watchlist is empty",
            }
        )
        return None

    SYNC_STATUS.update(
        {
            "running": True,
            "last_started": utc_now(),
            "last_error": None,
            "message": (
                f"Updating {len(tickers)} symbols"
            ),
        }
    )

    try:
        result = await asyncio.to_thread(
            sync_intraday_prices,
            tickers,
            5,
            20,
        )

        SYNC_STATUS.update(
            {
                "running": False,
                "last_finished": utc_now(),
                "last_rows": result["rows"],
                "message": (
                    f"Updated {result['updated_symbols']} "
                    "symbols"
                ),
            }
        )

        return result
    except Exception as error:
        SYNC_STATUS.update(
            {
                "running": False,
                "last_finished": utc_now(),
                "last_error": str(error),
                "message": "Automatic update failed",
            }
        )

        return None


async def watchlist_sync_loop():
    try:
        while True:
            await sync_watchlist_once()
            await asyncio.sleep(
                SYNC_INTERVAL_SECONDS
            )
    except asyncio.CancelledError:
        return
