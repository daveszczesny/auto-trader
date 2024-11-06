import unittest
import numpy as np

from utils.constants import StatusCode
from utils.action import construct_action

class ActionTest(unittest.TestCase):

    def test_construct_action(self):
        action = np.array([0.34, 0.5, 0, 0], np.float32)

        response, err, status_code = construct_action(action)

        self.assertIsInstance(response, dict)
        self.assertIsNotNone(response['action'])
        self.assertIsNone(err)
        self.assertEqual(status_code, StatusCode.ACCEPTED)

        self.assertEqual(response['action'], 'LONG')
        self.assertEqual(response['lot_size'], 0.5)
        self.assertIsNone(response['stop_loss'])
        self.assertIsNone(response['take_profit'])

        action = np.array([0.0, 0.0, 0.0, 0.0], np.float32)
        response, _, _ = construct_action(action)
        self.assertEqual(response['action'], 'DO_NOTHING')

        action = np.array([0.69, 0.0, 0.0, 0.0], np.float32)
        response, _, _ = construct_action(action)
        self.assertEqual(response['action'], 'SHORT')

        action = np.array([1.0, 0.0, 0.0, 0.0], np.float32)
        response, _, _ = construct_action(action)
        self.assertEqual(response['action'], 'CLOSE')
