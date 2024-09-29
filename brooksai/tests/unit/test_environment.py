import unittest

import pandas as pd
import pandas.testing as pdt

import numpy as np

from brooksai.env.forex import ForexEnv
from brooksai.env.models.constants import MAX_TRADES, ActionType, DEFAULT_TRADE_TTL
from brooksai.env.models.trade import open_trades, TradeType, close_trade

class EnvironmentText(unittest.TestCase):
    def setUp(self):
        self.data_path = "brooksai/tests/data/sample_data.csv"
        self.env = ForexEnv(self.data_path)

    def test_initialization(self):
        # Test if the data is loaded correctly
        self.assertIsInstance(self.env.data, pd.DataFrame)
        pdt.assert_frame_equal(self.env.data, pd.read_csv(self.data_path))

        # Test if observation space is set correctly
        self.assertEqual(self.env.observation_space.shape, (8 + MAX_TRADES, ))
        self.assertEqual(self.env.observation_space.dtype, np.float32)

        # Test if the action space is set correctly
        self.assertTrue(np.array_equal(self.env.action_space.low, np.array([0.0, 0.01, -1.0, -1.0, 0.0], dtype=np.float32)))
        self.assertTrue(np.array_equal(self.env.action_space.high, np.array([1.0, 1.0, 300.0, 300.0, MAX_TRADES - 1], dtype=np.float32)))
        self.assertEqual(self.env.action_space.dtype, np.float32)

        # Test initial agent variables
        self.assertEqual(self.env.current_step, 0)
        self.assertEqual(self.env.current_price, self.env.data.iloc[0]['bid_close'])
        self.assertEqual(self.env.reward, 0.0)
        self.assertEqual(self.env.current_balance, self.env.initial_balance)
        self.assertEqual(self.env.unrealized_profit, 0.0)
        self.assertEqual(self.env.previous_balance, 0.0)

    def test_step(self):
        # Define a sample action
        action = np.array([0.34, 0.5, 100.0, 100.0, 0], np.float32)

        # Perform a step in the environment
        observation, reward, done, _,  info = self.env.step(action)

        self.assertIsInstance(observation, np.ndarray)
        self.assertIsInstance(reward, float)
        self.assertIsInstance(done, bool)
        self.assertIsInstance(info, dict)

        self.assertEqual(len(open_trades), 1)

        # Test if trade opened long
        trade = open_trades[0]
        self.assertEqual(trade.trade_type, TradeType.LONG)
        self.assertEqual(trade.lot_size, 0.5)
        self.assertEqual(trade.stop_loss, 100.0)
        self.assertEqual(trade.take_profit, 100.0)
        self.assertEqual(trade.open_price, self.env.current_price)

        action = np.array([0.69, 0.5, 100.0, 100.0, 0], np.float32)
        self.env.step(action)

        trade = open_trades[1]
        self.assertEqual(trade.trade_type, TradeType.SHORT)
        self.assertEqual(trade.lot_size, 0.5)
        self.assertEqual(trade.stop_loss, 100.0)
        self.assertEqual(trade.take_profit, 100.0)
        self.assertEqual(trade.open_price, self.env.current_price)

        # Close one trade
        action = np.array([1, 0, 0, 0, 0], np.float32)
        self.env.step(action)

        self.assertEqual(len(open_trades), 1)

    def test_rest(self):
        # Reset the environment
        observation, _ = self.env.reset()

        # Check if the rest method returns the correct type
        self.assertIsInstance(observation, np.ndarray)
        self.assertEqual(observation.dtype, np.float32)

        # Check if the environment is reset correctly
        self.assertEqual(self.env.current_step, 0)
        self.assertEqual(self.env.current_price, self.env.data.iloc[0]['bid_close'])
        self.assertEqual(self.env.reward, 0.0)
        self.assertEqual(self.env.current_balance, self.env.initial_balance)
        self.assertEqual(self.env.unrealized_profit, 0.0)
        self.assertEqual(self.env.previous_balance, 0.0)
        self.assertEqual(len(open_trades), 0)

    def test_incorrect_action(self):

        # Attempt to close a none existing trade
        raw_action = np.array([1, 0, 0, 0, 0], np.float32)
        self.env.step(raw_action)

        action = self.env.construct_action(raw_action)
        self.assertEqual(action.action_type, ActionType.DO_NOTHING)


        # Action Long, lot size: 0, stop loss: -10, take profit: -10
        raw_action = np.array([0.34, 0, -10, -10, 0], np.float32)
        self.env.step(raw_action)
        
        trade = open_trades[0]
        self.assertEqual(trade.lot_size, 0.01)
        self.assertIsNone(trade.take_profit)
        self.assertIsNone(trade.stop_loss)


    def test_trade(self):
        self.env.reset()

        raw_action = np.array([0.34, 1, -1, -1, 0], np.float32)

        self.env.step(raw_action)

        self.assertEqual(len(open_trades), 1)

        # do nothing
        raw_action = np.array([0,0,0,0,0], np.float32)
        action = self.env.construct_action(raw_action)
        self.assertEqual(action.action_type, ActionType.DO_NOTHING)
        
        self.env.step(raw_action)

        trade = open_trades[0]
        self.assertNotEqual(trade.ttl, DEFAULT_TRADE_TTL)

        self.env.current_balance += close_trade(trade, self.env.current_price)

        self.assertEqual(len(open_trades), 0)
        self.assertNotEqual(self.env.current_balance, self.env.initial_balance)