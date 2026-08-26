import unittest
from types import SimpleNamespace
import pandas as pd

from backend.app.data.yahoo_intraday import save_intraday_prices, to_utc


class FakeDb:
    def __init__(self):
        self.statements = []

    def execute(self, statement):
        self.statements.append(statement)


class IntradayPriceTests(unittest.TestCase):
    def test_naive_timestamp_is_changed_to_utc(self):
        result = to_utc("2026-08-25 09:30:00")

        self.assertEqual(str(result.tz), "UTC")
        self.assertEqual(result.hour, 13)
        self.assertEqual(result.minute, 30)

    def test_unfinished_candle_is_not_saved(self):
        data = pd.DataFrame(
            {
                "Open": [100.0],
                "High": [101.0],
                "Low": [99.0],
                "Close": [100.5],
                "Volume": [500],
            },
            index=[pd.Timestamp("2026-08-25 10:00:00", tz="America/New_York")],
        )

        db = FakeDb()
        symbol = SimpleNamespace(id=1)
        count = save_intraday_prices(
            db,
            symbol,
            data,
            current_time=pd.Timestamp("2026-08-25 14:10:00", tz="UTC"),
        )

        self.assertEqual(count, 0)
        self.assertEqual(db.statements, [])

    def test_completed_candle_is_saved(self):
        data = pd.DataFrame(
            {
                "Open": [100.0],
                "High": [101.0],
                "Low": [99.0],
                "Close": [100.5],
                "Volume": [500],
            },
            index=[pd.Timestamp("2026-08-25 10:00:00", tz="America/New_York")],
        )

        db = FakeDb()
        symbol = SimpleNamespace(id=1)
        count = save_intraday_prices(
            db,
            symbol,
            data,
            current_time=pd.Timestamp("2026-08-25 14:16:00", tz="UTC"),
        )

        self.assertEqual(count, 1)
        self.assertEqual(len(db.statements), 1)


if __name__ == "__main__":
    unittest.main()
