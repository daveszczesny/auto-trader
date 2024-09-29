import unittest
from unittest.mock import patch, MagicMock

from currency_converter import CurrencyConverter

from brooksai.env.models.constants import TradeType, Environments, CONTRACT_SIZE, LEVERAGE
from brooksai.env.models.trade import Trade, open_trades, reset_open_trades, get_trade_profit, pip_to_profit

class TradeTest(unittest.TestCase):
    def setUp(self):
        # Reset open trades before each test
        reset_open_trades()

    def test_trade_initialization(self):
        # Initialize a trade
        trade = Trade(
            lot_size=1.0,
            open_price=1.2,
            trade_type=TradeType.LONG,
            stop_loss=1.1,
            take_profit=1.3
        )

        # Check if the trade is initialized correctly
        self.assertEqual(trade.lot_size, 1.0)
        self.assertEqual(trade.open_price, 1.2)
        self.assertEqual(trade.trade_type, TradeType.LONG)
        self.assertEqual(trade.stop_loss, 1.1)
        self.assertEqual(trade.take_profit, 1.3)

    @patch('brooksai.env.models.trade.c')
    def test_get_margin(self, mock_currency_converter):
        # Mock the currency converter
        mock_currency_converter.convert.return_value = 0.85  # Mock conversion rate

        # Initialize a trade
        trade = Trade(
            lot_size=1.0,
            open_price=1.2,
            trade_type=TradeType.LONG
        )

        # Calculate the margin
        margin = trade.get_margin()

        # Check if the margin is calculated correctly
        expected_margin = ((1 * CONTRACT_SIZE) / LEVERAGE) * 0.85
        self.assertAlmostEqual(margin, expected_margin)


    @patch('brooksai.env.models.trade.c')
    def test_trade_profit(self, mock_currency_converter):
        # Mock the currency converter
        mock_currency_converter.convert.return_value = 1

        # Initialize a trade
        trade = Trade(
            lot_size=1.0,
            open_price=1.2,
            trade_type=TradeType.LONG
        )

        current_price = 1.205

        profit = (current_price - 1.2) * CONTRACT_SIZE * 1
        self.assertEqual(profit, get_trade_profit(trade, current_price))

