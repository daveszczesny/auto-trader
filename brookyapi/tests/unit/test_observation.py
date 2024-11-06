import unittest

import numpy as np

from utils.observation import get_observation, get_observation_list
from utils.exceptions import ErrorSet
from utils.constants import StatusCode
from tests.utils.common import read_json_file

class ObservationTest(unittest.TestCase):
    def test_valid_observation(self):

        payload = read_json_file('brookyapi/tests/data/sample_payload.json')

        observation, err, status_code = get_observation(payload)
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

        observation, err, status_code = get_observation(payload)
        self.assertEqual(err, ErrorSet.INVALID_INPUT)
        self.assertEqual(status_code, StatusCode.BAD_REQUEST)
        self.assertIsNone(observation)

    def test_invalid_indicators(self):
        payload = read_json_file('brookyapi/tests/data/sample_payload.json')
        payload['indicators'] = []

        observation, err, status_code = get_observation(payload)
        self.assertEqual(err, ErrorSet.INVALID_INPUT)
        self.assertEqual(status_code, StatusCode.BAD_REQUEST)
        self.assertIsNone(observation)

    
    def test_valid_observation_list(self):
        
        payload = read_json_file('brookyapi/tests/data/sample_warmup_payload.json')

        observation_list, err, status_code = get_observation_list(payload)
        self.assertIsNone(err)
        self.assertEqual(status_code, StatusCode.OK)
        self.assertIsNotNone(observation_list)
        self.assertIsInstance(observation_list, list)
        self.assertIsInstance(observation_list[0], np.ndarray)
        self.assertEqual(observation_list[0].shape, (9, ))

    def test_invalid_input_observation_list(self):
        payload = read_json_file('brookyapi/tests/data/sample_warmup_payload.json')
        payload.pop('balance')

        observation_list, err, status_code = get_observation_list(payload)
        self.assertEqual(err, ErrorSet.INVALID_INPUT)
        self.assertEqual(status_code, StatusCode.BAD_REQUEST)
        self.assertIsNone(observation_list)

    def test_invalid_observation_list_length(self):
        payload = read_json_file('brookyapi/tests/data/sample_warmup_payload.json')
        payload['current_prices'] = [1]

        observation_list, err, status_code = get_observation_list(payload)
        self.assertEqual(err, ErrorSet.INVALID_INPUT)
        self.assertEqual(status_code, StatusCode.BAD_REQUEST)
        self.assertIsNone(observation_list)