import unittest
from datetime import date, datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import patch

from backend.app.api.symbols import (
    get_ai_predictions,
    get_symbol_prices,
    search_symbols,
)
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


class FakePredictionDb:
    def __init__(self):
        self.symbol = SimpleNamespace(
            id=1,
            ticker="AAPL",
            name="Apple Inc.",
            primary_exchange="NASDAQ",
        )

        last_date = date(2026, 8, 27)

        self.features = [
            SimpleNamespace(
                date=(
                    last_date
                    - timedelta(days=number)
                )
            )
            for number in range(60)
        ]

        self.price = SimpleNamespace(
            date=last_date,
            close=230.50,
        )

    def query(self, model):
        if model.__name__ == "Symbol":
            return FakeQuery([self.symbol])

        if model.__name__ == "DailyFeature":
            return FakeQuery(
                self.features.copy()
            )

        return FakeQuery([self.price])


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

    @patch(
        "backend.app.api.symbols.predict_for_rows"
    )
    def test_get_ai_predictions(
        self,
        predict_mock,
    ):
        predict_mock.return_value = [
            {
                "horizon_days": 120,
                "model": "Transformer",
                "probability_up": 0.61,
                "direction": "up",
                "validation_auc": 0.6312,
                "test_auc": 0.6194,
                "quality": "moderate",
            }
        ]

        result = get_ai_predictions(
            ticker="AAPL",
            db=FakePredictionDb(),
        )

        self.assertEqual(
            result["reference_price"],
            230.50,
        )
        self.assertEqual(
            result["predictions"][0][
                "horizon_days"
            ],
            120,
        )
        predict_mock.assert_called_once()


if __name__ == "__main__":
    unittest.main()
