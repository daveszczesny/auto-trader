import unittest

import numpy as np
import torch
import gymnasium as gym

import brooksai.models.trade as trade_model
from brooksai.env.simpleforex import SimpleForexEnv


CURRENT_PRICE_INDEX = 6
CURRENT_HIGH_INDEX = 5
CURRENT_LOW_INDEX = 4

class EnvironmentTest(unittest.TestCase):
    def setUp(self):
        self.data_path = "brooksai/tests/data/sample_data.csv"
        self.env = SimpleForexEnv(self.data_path)

    def test_initialization(self):

        # Test if the data is loaded correctly
        self.assertIsInstance(self.env.data, torch.Tensor)
        self.assertIsNotNone(self.env.data)

        # Test if observation space is set correctly
        self.assertEqual(self.env.observation_space.shape, (9, ))
        self.assertIsInstance(self.env.observation_space, gym.spaces.Box)
        self.assertEqual(self.env.observation_space.dtype, np.float32)

        # Test if action space is set correctly
        self.assertEqual(self.env.action_space.shape, (4, ))
        self.assertTrue(np.array_equal(self.env.action_space.low, np.array([0.0, 0, 0, 0], dtype=np.float32)))
        self.assertTrue(np.array_equal(self.env.action_space.high, np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)))
        self.assertEqual(self.env.action_space.dtype, np.float32)

        # Test inital variables
        self.assertEqual(self.env.current_step, 0)
        self.assertEqual(self.env.current_price, self.env.data[0, CURRENT_PRICE_INDEX].item())
        self.assertEqual(self.env.current_high, self.env.data[0, CURRENT_HIGH_INDEX].item())
        self.assertEqual(self.env.current_low, self.env.data[0, CURRENT_LOW_INDEX].item())
        self.assertEqual(self.env.current_emas, (
            self.env.data[0, 10].item(),
            self.env.data[0, 11].item(),
            self.env.data[0, 12].item()
        ))
        self.assertEqual(self.env.reward, 0.0)
        self.assertEqual(self.env.current_balance, self.env.initial_balance)
        self.assertEqual(self.env.unrealised_pnl, 0.0)
        self.assertEqual(self.env.done, False)
        self.assertEqual(len(trade_model.open_trades), 0)

    def test_step(self):
        # Define a sample action
        action = np.array([0.34, 0.5, 0, 0], np.float32)

        # Perform a step in the environment
        obs, reward, done, _, info = self.env.step(action)

        self.assertIsInstance(obs, np.ndarray)
        self.assertIsInstance(reward, float)
        self.assertIsInstance(done, bool)
        self.assertIsInstance(info, dict)

        self.assertEqual(len(trade_model.open_trades), 1)

        # Test if trade opened long
        trade = trade_model.open_trades[0]
        self.assertEqual(trade.trade_type, trade_model.TradeType.LONG)
        self.assertEqual(trade.lot_size, 0.5)
        self.assertEqual(trade.open_price, self.env.current_price)
        self.assertEqual(trade.take_profit, None)
        self.assertEqual(trade.stop_loss, None)

        # close trade
        action = np.array([1, 0, 0, 0], np.float32)
        self.env.step(action)

        self.assertEqual(len(trade_model.open_trades), 0)

        # Test done
        self.assertFalse(self.env.done)

        self.env.current_balance = 0
        self.env.step(action)

        self.assertTrue(self.env.done)

    def test_reset(self):
        self.env.reset()

        self.assertEqual(self.env.current_step, 0)
        self.assertEqual(self.env.current_price, self.env.data[0, CURRENT_PRICE_INDEX].item())
        self.assertEqual(self.env.current_high, self.env.data[0, CURRENT_HIGH_INDEX].item())
        self.assertEqual(self.env.current_low, self.env.data[0, CURRENT_LOW_INDEX].item())
        self.assertEqual(self.env.current_emas, (
            self.env.data[0, 10].item(),
            self.env.data[0, 11].item(),
            self.env.data[0, 12].item()
        ))

        self.assertEqual(self.env.reward, 0.0)
        self.assertEqual(self.env.current_balance, self.env.initial_balance)
        self.assertEqual(self.env.unrealised_pnl, 0.0)
        self.assertEqual(self.env.done, False)
        self.assertEqual(len(trade_model.open_trades), 0)
