import unittest
import requests

from utils.constants import StatusCode
from utils.exceptions import ErrorSet
from tests.utils.common import read_json_file

class TestBrookyAPI(unittest.TestCase):

    BASE_URL: str = 'https://brooky-api-550951781970.europe-west2.run.app'
    BROOKSAI_URL: str = f'{BASE_URL}/brooksai'
    PREDICT_ENDPOINT: str = f'{BROOKSAI_URL}/predict'
    WARMUP_ENDPOINT: str = f'{BROOKSAI_URL}/warmup'
    TIMEOUT = 50

    def test_01_warmup(self):
        payload = read_json_file('brookyapi/tests/data/sample_warmup_payload.json')

        response = requests.post(self.WARMUP_ENDPOINT, json=payload, timeout=self.TIMEOUT)

        self.assertEqual(response.status_code, StatusCode.NO_CONTENT)

    def test_02_predict(self):
        payload = read_json_file('brookyapi/tests/data/sample_payload.json')

        response = requests.post(self.PREDICT_ENDPOINT, json=payload, timeout=self.TIMEOUT)

        self.assertEqual(response.status_code, StatusCode.OK)
        self.assertTrue('action' in response.json())

    def test_03_invalid_payload_predict(self):
        response = requests.post(self.PREDICT_ENDPOINT, json={}, timeout=self.TIMEOUT)

        self.assertEqual(response.status_code, StatusCode.BAD_REQUEST)
        self.assertEqual(response.json()['error'], ErrorSet.INVALID_INPUT.to_dict())
