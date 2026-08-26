import unittest
from datetime import datetime, timezone
from types import SimpleNamespace

from backend.app.api.symbols import get_symbol_prices, search_symbols
from backend.app.main import health


class FakeQuery:
    def __init__(self, rows):
        self.rows = rows

    def filter(self, *args):
        return self

    def order_by(self, *args):
        return self

    def limit(self, value):
        self.rows = self.rows[:value]
        return self

    def all(self):
        return self.rows

    def first(self):
        return self.rows[0] if self.rows else None


class FakeDb:
    def __init__(self):
        self.symbol = SimpleNamespace(
            id=1,
            ticker="AAPL",
            name="Apple Inc.",
            primary_exchange="NASDAQ",
        )
        self.prices = [
            SimpleNamespace(
                timestamp=datetime(2026, 8, 25, 14, 15, tzinfo=timezone.utc),
                open=101.0,
                high=102.0,
                low=100.5,
                close=101.5,
                volume=600,
            ),
            SimpleNamespace(
                timestamp=datetime(2026, 8, 25, 14, 0, tzinfo=timezone.utc),
                open=100.0,
                high=101.0,
                low=99.5,
                close=100.5,
                volume=500,
            ),
        ]

    def query(self, model):
        if model.__name__ == "Symbol":
            return FakeQuery([self.symbol])
        return FakeQuery(self.prices.copy())


class ApiTests(unittest.TestCase):
    def test_health(self):
        result = health()

        self.assertEqual(result, {"status": "ok"})

    def test_search_symbols(self):
        result = search_symbols(search="AAPL", limit=20, db=FakeDb())

        self.assertEqual(result[0]["ticker"], "AAPL")

    def test_get_intraday_prices(self):
        result = get_symbol_prices(
            ticker="AAPL",
            interval="15m",
            limit=200,
            db=FakeDb(),
        )

        self.assertEqual(result["count"], 2)
        self.assertEqual(result["prices"][0]["close"], 100.5)


if __name__ == "__main__":
    unittest.main()
