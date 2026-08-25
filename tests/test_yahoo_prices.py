import os
import unittest
from unittest.mock import patch

import pandas as pd


os.environ.setdefault(
    "DATABASE_URL",
    "postgresql+psycopg://user:password@localhost/test",
)

from backend.app.data.yahoo_prices import (
    download_prices,
    get_symbol_data,
)


class YahooPricesTest(unittest.TestCase):
    def test_get_symbol_data_for_ticker_first_columns(self):
        columns = pd.MultiIndex.from_tuples(
            [
                ("AAPL", "Open"),
                ("AAPL", "Close"),
            ]
        )

        data = pd.DataFrame(
            [[100.0, 101.0]],
            columns=columns,
        )

        result = get_symbol_data(data, "AAPL")

        self.assertEqual(
            list(result.columns),
            ["Open", "Close"],
        )

    def test_get_symbol_data_for_price_first_columns(self):
        columns = pd.MultiIndex.from_tuples(
            [
                ("Open", "AAPL"),
                ("Close", "AAPL"),
            ]
        )

        data = pd.DataFrame(
            [[100.0, 101.0]],
            columns=columns,
        )

        result = get_symbol_data(data, "AAPL")

        self.assertEqual(
            list(result.columns),
            ["Open", "Close"],
        )

    @patch("backend.app.data.yahoo_prices.yf.download")
    def test_download_with_date_range_does_not_use_period(
        self,
        download_mock,
    ):
        download_prices(
            tickers=["AAPL"],
            start="2026-08-20",
            end="2026-08-26",
        )

        options = download_mock.call_args.kwargs

        self.assertEqual(options["start"], "2026-08-20")
        self.assertEqual(options["end"], "2026-08-26")
        self.assertNotIn("period", options)


if __name__ == "__main__":
    unittest.main()
