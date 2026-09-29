import unittest
from types import SimpleNamespace

from backend.app.api.paper_trades import (
    calculate_position,
)


def trade(side, quantity, price):
    return SimpleNamespace(
        side=side,
        quantity=quantity,
        price=price,
    )


class PaperTradeTests(unittest.TestCase):
    def test_buy_calculates_average_price(self):
        trades = [
            trade("BUY", 2, 100),
            trade("BUY", 1, 130),
        ]

        result = calculate_position(
            trades,
            current_price=120,
        )

        self.assertEqual(result["shares"], 3)
        self.assertEqual(
            result["average_buy_price"],
            110,
        )
        self.assertEqual(
            result["unrealized_profit_loss"],
            30,
        )

    def test_sell_calculates_realized_result(self):
        trades = [
            trade("BUY", 4, 100),
            trade("SELL", 1, 120),
        ]

        result = calculate_position(
            trades,
            current_price=110,
        )

        self.assertEqual(result["shares"], 3)
        self.assertEqual(
            result["realized_profit_loss"],
            20,
        )
        self.assertEqual(
            result["total_profit_loss"],
            50,
        )

    def test_sell_cannot_exceed_position(self):
        trades = [
            trade("BUY", 1, 100),
            trade("SELL", 2, 110),
        ]

        with self.assertRaises(ValueError):
            calculate_position(trades)

    def test_closed_position_has_no_average_price(self):
        trades = [
            trade("BUY", 1, 100),
            trade("SELL", 1, 105),
        ]

        result = calculate_position(trades)

        self.assertEqual(result["shares"], 0)
        self.assertIsNone(
            result["average_buy_price"]
        )
        self.assertEqual(
            result["realized_profit_loss"],
            5,
        )


if __name__ == "__main__":
    unittest.main()
