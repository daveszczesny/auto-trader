import unittest

from brooksai.env.simpleforex import SimpleForexEnv

class EnvironmentTest(unittest.TestCase):
    def setUp(self):
        self.data_path = "brooksai/tests/data/sample_data.csv"
        self.env = SimpleForexEnv(self.data_path)
