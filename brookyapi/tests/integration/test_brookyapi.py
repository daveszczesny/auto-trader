import json
from datetime import datetime

import unittest
from unittest.mock import patch, MagicMock
import numpy as np
from flask import Flask
from main import app

from utils.constants import StatusCode
from utils.exceptions import ErrorSet
from tests.utils.common import read_json_file

class BrookyAPITest(unittest.TestCase):
    def setUp(self):
        self.app = app.test_client()
        self.app.testing = True

    @patch('functions.brooksai.brooky.storage.Client')
    @patch('functions.brooksai.brooky._get_model_object')
    def test_predict(self, mock_get_model, mock_storage_client):

        mock_client = MagicMock()
        mock_storage_client.return_value = mock_client

        mock_bucket, mock_blob = MagicMock(), MagicMock()
        mock_client.bucket.return_value = mock_bucket
        mock_bucket.blob.return_value = mock_blob
        mock_blob.download_as_string.return_value = json.dumps(
            read_json_file('brookyapi/tests/data/sample_payload.json')
        ).encode()

        mock_blob.upload_from_string.return_value = MagicMock()

        mock_model_object = MagicMock()
        mock_model_object.predict.return_value = (np.array([0.0, 0.0, 0.0, 0.0]), np.array([0.5]))

        mock_get_model.return_value = (mock_model_object, None, StatusCode.OK)

        payload = read_json_file('brookyapi/tests/data/sample_payload.json')

        response = self.app.post('/brooksai/predict', data=json.dumps(payload), content_type='application/json')

        self.assertEqual(response.status_code, StatusCode.OK)

        response_data = json.loads(response.data)
        self.assertTrue('action' in response_data)

        mock_get_model.assert_called_once()
        mock_model_object.predict.assert_called_once()
    
    def test_predict_invalid_payload(self):
        response = self.app.post('/brooksai/predict', data=json.dumps({}), content_type='application/json')
        resp = response.get_json()

        self.assertEqual(response.status_code, StatusCode.BAD_REQUEST)
        self.assertEqual(resp['error']['code'], ErrorSet.INVALID_INPUT.code)
        self.assertEqual(resp['error']['message'], ErrorSet.INVALID_INPUT.message)

    
    @patch('functions.brooksai.brooky._get_model_object')
    def test_predict_model_not_found(self, mock_get_model):
        mock_get_model.return_value = (None, ErrorSet.MODEL_NOT_FOUND, StatusCode.NOT_FOUND)

        payload = read_json_file('brookyapi/tests/data/sample_payload.json')

        response = self.app.post('/brooksai/predict', data=json.dumps(payload), content_type='application/json')
        resp = response.get_json()

        self.assertEqual(response.status_code, StatusCode.NOT_FOUND)
        self.assertEqual(resp['error']['code'], ErrorSet.MODEL_NOT_FOUND.code)
        self.assertEqual(resp['error']['message'], ErrorSet.MODEL_NOT_FOUND.message)

    @patch('functions.brooksai.brooky.RecurrentPPOAgent')
    @patch('functions.brooksai.brooky.storage.Client')
    def test_ppo_agent_failure(self, mock_agent, mock_storage_client):
        mock_client = MagicMock()
        mock_storage_client.return_value = mock_client

        mock_bucket, mock_blob = MagicMock(), MagicMock()
        mock_client.bucket.return_value = mock_bucket
        mock_bucket.blob.return_value = mock_blob
        mock_blob.download_as_string.return_value = json.dumps(
            read_json_file('brookyapi/tests/data/sample_payload.json')
        ).encode()

        mock_blob.upload_from_string.return_value = MagicMock()

        mock_agent.side_effect = Exception('Failed to load agent')

        payload = read_json_file('brookyapi/tests/data/sample_payload.json')
        response = self.app.post('/brooksai/predict', data=json.dumps(payload), content_type='application/json')
        resp = response.get_json()

        self.assertEqual(response.status_code, StatusCode.INTERNAL_SERVER_ERROR)
        self.assertEqual(resp['error']['code'], ErrorSet.UNKNOWN_ERROR.code)
        self.assertEqual(resp['error']['message'], ErrorSet.UNKNOWN_ERROR.message)


    @patch('functions.brooksai.brooky.patch_instances')
    @patch('functions.brooksai.brooky._get_model_object')
    def test_warmup(self, mock_get_model, mock_patch_instances):

        mock_patch_instances.return_value = (None, StatusCode.OK)

        payload = read_json_file('brookyapi/tests/data/sample_warmup_payload.json')

        mock_model_object = MagicMock()
        mock_model_object.predict.return_value = (np.array([0.0, 0.0, 0.0, 0.0]), np.array([0.5]))

        mock_get_model.return_value = (mock_model_object, None, StatusCode.OK)



        response = self.app.post('/brooksai/warmup', data=json.dumps(payload), content_type='application/json')
        from functions.brooksai.brooky import lstm_states, episode_start

        mock_get_model.assert_called_once()
        self.assertEqual(response.status_code, StatusCode.NO_CONTENT)
        self.assertEqual(mock_model_object.predict.call_count, len(payload['current_prices']))
        self.assertIsNotNone(lstm_states)
        self.assertFalse(episode_start)


    def test_warmup_invalid_payload(self):
        response = self.app.post('/brooksai/warmup', data=json.dumps({}), content_type='application/json')
        resp = response.get_json()

        self.assertEqual(response.status_code, StatusCode.BAD_REQUEST)
        self.assertEqual(resp['error']['code'], ErrorSet.INVALID_INPUT.code)
        self.assertEqual(resp['error']['message'], ErrorSet.INVALID_INPUT.message)