import unittest

import numpy as np

import brooksai.models.trade as trade_model
from brooksai.models.action import Action as ActionModel, TradeAction
from brooksai.utils.action import ActionBuilder, ActionApply
from brooksai.models.constants import ActionType, ApplicationConstants

class ActionTest(unittest.TestCase):

    def test_action_builder(self):

        trade_model.open_trades.clear()

        # Test DO_NOTHING action
        action = np.array([0.0, 0.0, 0.0, 0.0], np.float32)
        action_model = ActionBuilder.construct_action(action)

        self.assertIsInstance(action_model, ActionModel)
        self.assertEqual(action_model.action_type, ActionType.DO_NOTHING)

        # Test LONG action
        action = np.array([0.34, 0.5, 0, 0], np.float32)

        action_model = ActionBuilder.construct_action(action)

        self.assertIsInstance(action_model, ActionModel)
        self.assertEqual(action_model.action_type, ActionType.LONG)
        self.assertIsInstance(action_model.trade_data, TradeAction)
        self.assertEqual(action_model.trade_data.lot_size, 0.5)
        self.assertEqual(action_model.trade_data.stop_loss, None)
        self.assertEqual(action_model.trade_data.take_profit, None)

        # Test SHORT action
        action = np.array([0.69, 0.5, 0.0, 0.0], np.float32)

        action_model = ActionBuilder.construct_action(action)

        self.assertIsInstance(action_model, ActionModel)
        self.assertEqual(action_model.action_type, ActionType.SHORT)
        self.assertIsInstance(action_model.trade_data, TradeAction)
        self.assertEqual(action_model.trade_data.lot_size, 0.5)
        self.assertEqual(action_model.trade_data.stop_loss, None)
        self.assertEqual(action_model.trade_data.take_profit, None)

        # Test CLOSE action
        # This assumes that there is only one open trade at a time
        trade_model.open_trades.append('foo')

        action = np.array([1.0, 0.0, 0.0, 0.0], np.float32)
        action_model = ActionBuilder.construct_action(action)

        self.assertIsInstance(action_model, ActionModel)
        self.assertEqual(action_model.action_type, ActionType.CLOSE)
        trade_model.open_trades.clear()

    def test_invalid_action_builder(self):

        # Test invalid action spaces
        action = np.array([-1, 0.0, 0.0, 0.0], np.float32)
        action_model = ActionBuilder.construct_action(action)

        self.assertIsInstance(action_model, ActionModel)
        self.assertEqual(action_model.action_type, ActionType.DO_NOTHING)

        action = np.array([2.0, 0.0, 0.0, 0.0], np.float32)
        action_model = ActionBuilder.construct_action(action)

        self.assertEqual(action_model.action_type, ActionType.DO_NOTHING)

        # Test invalid CLOSE action
        action = np.array([1.0, 0.0, 0.0, 0.0], np.float32)
        action_model = ActionBuilder.construct_action(action)

        # No trades opened, cannot close a trade
        self.assertEqual(trade_model.open_trades, [])

        self.assertEqual(action_model.action_type, ActionType.DO_NOTHING)

        # Test invalid LONG or SHORT actions

        # Invalid lot size
        action = np.array([0.34, 0.0, 0, 0], np.float32)
        action_model = ActionBuilder.construct_action(action)

        self.assertEqual(action_model.action_type, ActionType.DO_NOTHING)

        action = np.array([0.69, 0.0, 0.0, 0.0], np.float32)
        action_model = ActionBuilder.construct_action(action)

        self.assertEqual(action_model.action_type, ActionType.DO_NOTHING)

        # Invalid action with open trades
        trade_model.open_trades.append('foo')

        action = np.array([0.34, 0.5, 0, 0], np.float32)
        action_model = ActionBuilder.construct_action(action)

        self.assertNotEqual(trade_model.open_trades, [])
        self.assertEqual(action_model.action_type, ActionType.DO_NOTHING)

        action = np.array([0.69, 0.5, 0.0, 0.0], np.float32)
        action_model = ActionBuilder.construct_action(action)

        self.assertEqual(action_model.action_type, ActionType.DO_NOTHING)
        trade_model.open_trades.clear()


    # Test Action Apply
    def test_action_apply(self):
        trade_window = 10
        current_price = 1.0

        # Test LONG action
        action = np.array([0.34, 0.5, 0, 0], np.float32)
        action_model = ActionBuilder.construct_action(action)

        value, tw = ActionApply.apply_action(
            action_model,
            trade_window=trade_window,
            current_price=current_price)

        # Trade was opened, value should be 0.0
        # Trade window should be reset to default
        self.assertEqual(value, 0.0)
        self.assertEqual(tw, ApplicationConstants.DEFAULT_TRADE_WINDOW)

        self.assertNotEqual(trade_model.open_trades, [])

        # Test CLOSE action
        action = np.array([1.0, 0.0, 0.0, 0.0], np.float32)
        action_model = ActionBuilder.construct_action(action)

        trade = trade_model.open_trades[0]

        trade_window = 8
        current_price = 1.5

        value, tw = ActionApply.apply_action(
            action_model,
            trade_window=trade_window,
            current_price=current_price)

        self.assertEqual(
            value,
            trade_model.get_trade_profit(trade, current_price) - ApplicationConstants.TRANSACTION_FEE)

        # Trade window does not change
        self.assertEqual(tw, trade_window)
        self.assertEqual(trade_model.open_trades, [])

        # Test Do Nothing action
        action = np.array([0.0, 0.0, 0.0, 0.0], np.float32)
        action_model = ActionBuilder.construct_action(action)

        trade_window = 5
        value, tw = ActionApply.apply_action(
            action_model,
            trade_window=trade_window,
            current_price=current_price)

        self.assertEqual(value, 0.0)
        self.assertNotEqual(tw, trade_window)
        self.assertEqual(trade_model.open_trades, [])

        # Test Short trade
        action = np.array([0.69, 0.5, 0.0, 0.0], np.float32)
        action_model = ActionBuilder.construct_action(action)

        value, tw = ActionApply.apply_action(
            action_model,
            trade_window=trade_window,
            current_price=current_price)

        # Trade was opened, value should be 0.0
        # Trade window should be reset to default
        self.assertEqual(value, 0.0)
        self.assertEqual(tw, ApplicationConstants.DEFAULT_TRADE_WINDOW)
        self.assertNotEqual(trade_model.open_trades, [])

        trade_model.open_trades.clear()



    def test_action_tracker(self):
        ActionApply.reset_tracker()

        self.assertEqual(ActionApply.get_action_tracker('trades_opened'), 0)
        self.assertEqual(ActionApply.get_action_tracker('trades_closed'), 0)
        self.assertEqual(ActionApply.get_action_tracker('total_won'), 0)
        self.assertEqual(ActionApply.get_action_tracker('total_lost'), 0)
        self.assertEqual(ActionApply.get_action_tracker('times_won'), 0)
        self.assertEqual(ActionApply.get_action_tracker('times_lost'), 0)

        self.assertIsInstance(ActionApply.get_action_tracker('times_won'), int)

        # Open a trade
        action = np.array([0.34, 0.5, 0, 0], np.float32)
        action_model = ActionBuilder.construct_action(action)

        ActionApply.apply_action(
            action_model,
            trade_window=10,
            current_price=1.0)


        # Close the trade
        current_price = 1.5
        action = np.array([1.0, 0.0, 0.0, 0.0], np.float32)
        action_model = ActionBuilder.construct_action(action)

        trade = trade_model.open_trades[0]

        ActionApply.apply_action(
            action_model,
            trade_window=8,
            current_price=current_price)

        self.assertEqual(ActionApply.get_action_tracker('trades_opened'), 1)
        self.assertEqual(ActionApply.get_action_tracker('trades_closed'), 1)
        self.assertEqual(
            ActionApply.get_action_tracker('total_won'),
            trade_model.get_trade_profit(trade, current_price) - ApplicationConstants.TRANSACTION_FEE)
        self.assertEqual(ActionApply.get_action_tracker('times_won'), 1)

        # Open another trade
        action = np.array([0.34, 0.5, 0.0, 0.0], np.float32)
        action_model = ActionBuilder.construct_action(action)

        ActionApply.apply_action(
            action_model,
            trade_window=5,
            current_price=current_price)

        trade = trade_model.open_trades[0]

        action = np.array([1, 0.0, 0.0, 0.0], np.float32)
        action_model = ActionBuilder.construct_action(action)

        current_price = 0.8

        ActionApply.apply_action(
            action_model,
            trade_window=5,
            current_price=current_price)

        self.assertEqual(ActionApply.get_action_tracker('trades_opened'), 2)
        self.assertEqual(ActionApply.get_action_tracker('trades_closed'), 2)
        self.assertEqual(
            ActionApply.get_action_tracker('total_lost'),
            trade_model.get_trade_profit(trade, current_price) - ApplicationConstants.TRANSACTION_FEE)
        self.assertEqual(ActionApply.get_action_tracker('times_lost'), 1)

        ActionApply.reset_tracker()
