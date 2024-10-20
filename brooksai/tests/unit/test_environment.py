import unittest
import torch
import numpy as np

from brooksai.env.simpleforex import SimpleForexEnv
from brooksai.models.trade import open_trades

class EnvironmentTest(unittest.TestCase):
    def setUp(self):
        self.data_path = "brooksai/tests/data/sample_data.csv"
        self.env = SimpleForexEnv(self.data_path)
