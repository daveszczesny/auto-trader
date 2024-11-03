import unittest

import numpy as np

from utils.ai.observation import construct_observation
from utils.ai.exceptions import ErrorSet, StatusCode
from tests.utils.common import read_json_file

class ObservationTest(unittest.TestCase):
    def test_valid_observation(self):

        payload = read_json_file('brookyapi/tests/data/sample_payload.json')

        observation, err, status_code = construct_observation(payload)
        self.assertIsNone(err)
        self.assertEqual(status_code, StatusCode.OK)
        self.assertIsNotNone(observation)
        self.assertIsInstance(observation, np.ndarray)
        self.assertEqual(observation.shape, (9, ))

        self.assertIsNotNone(
            any(value is None for value in observation)
        )

    def test_invalid_input_observation(self):
        payload = read_json_file('brookyapi/tests/data/sample_payload.json')
        payload.pop('balance')

        observation, err, status_code = construct_observation(payload)
        self.assertEqual(err, ErrorSet.INVALID_INPUT)
        self.assertEqual(status_code, StatusCode.BAD_REQUEST)
        self.assertIsNone(observation)

    def test_invalid_indicators(self):
        payload = read_json_file('brookyapi/tests/data/sample_payload.json')
        payload['indicators'] = []

        observation, err, status_code = construct_observation(payload)
        self.assertEqual(err, ErrorSet.INVALID_INPUT)
        self.assertEqual(status_code, StatusCode.BAD_REQUEST)
        self.assertIsNone(observation)