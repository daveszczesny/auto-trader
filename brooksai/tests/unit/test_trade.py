import unittest
from unittest.mock import patch

from brooksai.env.models.constants import TradeType, ApplicationConstants
from brooksai.env.models.trade import Trade, reset_open_trades, get_trade_profit, close_trade,\
    trigger_stop_or_take_profit, get_trade_state, get_trade_by_id, open_trades


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

    def test_rest_open_trades(self):
        Trade(
            lot_size=1.0,
            open_price=1.2,
            trade_type=TradeType.LONG
        )

        self.assertGreater(len(open_trades), 0)
        reset_open_trades()
        self.assertEqual(len(open_trades), 0)

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
        print(margin)

        # Check if the margin is calculated correctly
        expected_margin = ((1 * ApplicationConstants.CONTRACT_SIZE) / ApplicationConstants.LEVERAGE) * 0.85
        self.assertAlmostEqual(margin, expected_margin)


    @patch('brooksai.env.models.trade.c')
    def test_trade_profit_long(self, mock_currency_converter):
        # Mock the currency converter
        mock_currency_converter.convert.return_value = 1

        # Initialize a trade
        trade = Trade(
            lot_size=1.0,
            open_price=1.2,
            trade_type=TradeType.LONG
        )

        current_price = 1.205

        profit = (current_price - 1.2) * ApplicationConstants.CONTRACT_SIZE * 1
        self.assertEqual(profit, get_trade_profit(trade, current_price))

    @patch('brooksai.env.models.trade.c')
    def test_trade_profit_short(self, mock_currency_converter):
        # Mock the currency converter
        mock_currency_converter.convert.return_value = 1

        # Initialize a trade
        trade = Trade(
            lot_size=1.0,
            open_price=1.2,
            trade_type=TradeType.SHORT
        )

        current_price = 1.195

        profit = (1.2 - current_price) * ApplicationConstants.CONTRACT_SIZE * 1
        self.assertEqual(profit, get_trade_profit(trade, current_price))

    @patch('brooksai.env.models.trade.close_trade')
    def test_closing_trades(self, mock_close_trade):
        from brooksai.env.models import trade as trade_module

        def close_trade_wrapper(trade, current_price):
            return close_trade(trade, current_price)
        
        mock_close_trade.side_effect = close_trade_wrapper

        for _ in range(10):
            trade_module.Trade(
                lot_size=1.0,
                open_price=1.2,
                trade_type=TradeType.LONG
            )

        self.assertEqual(len(trade_module.open_trades), 10)

        trade = trade_module.open_trades[0]
        trade_module.close_trade(trade, 1.2)
        self.assertEqual(len(trade_module.open_trades), 9)
        self.assertEqual(mock_close_trade.call_count, 1)

        trade_module.close_all_trades(1.2)
        self.assertEqual(len(trade_module.open_trades), 0)
        self.assertEqual(mock_close_trade.call_count, 10)

    def test_trigger_stop_or_take_profit(self):
        reset_open_trades()

        Trade(
            lot_size=1.0,
            open_price=1.2,
            trade_type=TradeType.LONG,
            stop_loss=1.1,
            take_profit=1.3
        )

        total_value = trigger_stop_or_take_profit(1.3, 1.2)
        self.assertEqual(len(open_trades), 0)
        self.assertGreater(total_value, 0)

    def test_get_trade_state(self):
        trade = Trade(
            lot_size=1.0,
            open_price=1.2,
            trade_type=TradeType.LONG,
            stop_loss=1.1,
            take_profit=1.3
        )

        trade_state = get_trade_state(trade.uuid, 1.3)
        self.assertEqual(trade_state['id'], trade.uuid)
        self.assertEqual(trade_state['lot_size'], trade.lot_size)
        self.assertEqual(trade_state['trade_type'], trade.trade_type)
        self.assertGreater(trade_state['profit'], 0)

    def test_get_trade_by_id(self):
        trade = Trade(
            lot_size=1.0,
            open_price=1.2,
            trade_type=TradeType.LONG,
            stop_loss=1.1,
            take_profit=1.3
        )

        self.assertEqual(trade, get_trade_by_id(trade.uuid))
