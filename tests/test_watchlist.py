import unittest
from types import SimpleNamespace
from unittest.mock import patch

from backend.app.api.watchlist import (
    add_to_watchlist,
    get_watchlist,
    remove_from_watchlist,
)
from backend.app.services.watchlist_sync import (
    SYNC_STATUS,
    sync_watchlist_once,
)


class FakeQuery:
    def __init__(self, rows):
        self.rows = rows

    def join(self, *args):
        return self

    def filter(self, *args):
        return self

    def order_by(self, *args):
        return self

    def all(self):
        return self.rows

    def first(self):
        return self.rows[0] if self.rows else None


class FakeDb:
    def __init__(self, watchlisted=False):
        self.symbol = SimpleNamespace(
            id=1,
            ticker="AAPL",
            name="Apple Inc.",
            primary_exchange="NASDAQ",
        )
        self.item = (
            SimpleNamespace(symbol_id=1)
            if watchlisted
            else None
        )
        self.commit_count = 0

    def query(self, model):
        if model.__name__ == "Symbol":
            return FakeQuery([self.symbol])

        rows = [self.item] if self.item else []
        return FakeQuery(rows)

    def add(self, item):
        self.item = item

    def delete(self, item):
        self.item = None

    def commit(self):
        self.commit_count += 1


class WatchlistApiTests(unittest.TestCase):
    def test_get_watchlist(self):
        result = get_watchlist(
            db=FakeDb(watchlisted=True)
        )

        self.assertEqual(
            result[0]["ticker"],
            "AAPL",
        )

    def test_add_to_watchlist(self):
        db = FakeDb()

        result = add_to_watchlist(
            ticker="aapl",
            db=db,
        )

        self.assertEqual(
            result["ticker"],
            "AAPL",
        )
        self.assertEqual(
            db.item.symbol_id,
            1,
        )
        self.assertEqual(
            db.commit_count,
            1,
        )

    def test_remove_from_watchlist(self):
        db = FakeDb(watchlisted=True)

        result = remove_from_watchlist(
            ticker="AAPL",
            db=db,
        )

        self.assertTrue(result["removed"])
        self.assertIsNone(db.item)
        self.assertEqual(
            db.commit_count,
            1,
        )


class WatchlistSyncTests(
    unittest.IsolatedAsyncioTestCase
):
    @patch(
        "backend.app.services.watchlist_sync.sync_intraday_prices"
    )
    @patch(
        "backend.app.services.watchlist_sync.get_watchlist_tickers"
    )
    async def test_sync_watchlist_once(
        self,
        tickers_mock,
        sync_mock,
    ):
        tickers_mock.return_value = [
            "AAPL",
            "MSFT",
        ]
        sync_mock.return_value = {
            "updated_symbols": 2,
            "rows": 40,
        }

        result = await sync_watchlist_once()

        self.assertEqual(result["rows"], 40)
        self.assertEqual(
            SYNC_STATUS["last_rows"],
            40,
        )
        self.assertFalse(
            SYNC_STATUS["running"]
        )

    @patch(
        "backend.app.services.watchlist_sync.sync_intraday_prices"
    )
    @patch(
        "backend.app.services.watchlist_sync.get_watchlist_tickers"
    )
    async def test_empty_watchlist_skips_download(
        self,
        tickers_mock,
        sync_mock,
    ):
        tickers_mock.return_value = []

        result = await sync_watchlist_once()

        self.assertIsNone(result)
        sync_mock.assert_not_called()


if __name__ == "__main__":
    unittest.main()
