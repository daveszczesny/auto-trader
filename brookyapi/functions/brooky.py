import logging
import json
from typing import Optional, Tuple, Dict, Any

import numpy as np
from flask import jsonify, make_response

from google.cloud import storage
from google.api_core.exceptions import NotFound, Forbidden, Conflict

from stable_baselines3.common.env_util import make_vec_env

from utils import register_env
from utils.ai.action import construct_action
from utils.ai.observation import construct_observation
from utils.ai.exceptions import ErrorEntry, ErrorSet, StatusCode
from brooksai.agent.recurrentppoagent import RecurrentPPOAgent

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('BrookyAPI')

BUCKET_NAME = 'at-brooky-bucket'
DIRECTORY = 'model'
MODEL_FILE = 'ppo_forex.zip'

model: Optional[RecurrentPPOAgent] = None
lstm_states: Optional[Tuple[np.ndarray, ...]] = None
episode_start: Optional[np.ndarray] = None

def predict(request):
    logger.info('Processing prediction request')

    try:
        payload = request.get_json()
        logger.info(f'Received data: {payload}')

        observation, err, status_code = construct_observation(payload)
        if err or status_code != StatusCode.OK:
            return _handle_error(err, status_code)

        model, err, status_code = _get_model_object()
        if err or status_code != StatusCode.OK:
            return _handle_error(err, status_code)

        lstm_states, episode_start, err, status_code = _get_instances()
        if err or status_code != StatusCode.OK:
            return _handle_error(err, status_code)

        action, lstm_states = model.predict(observation, lstm_states, episode_starts=episode_start)
        episode_start = True

        action, err, status_code = construct_action(action)
        if err or status_code != StatusCode.ACCEPTED:
            return _handle_error(err, status_code)

        logger.debug(f'Agent action: {action}')

        err, status_code = patch_instances(lstm_states, episode_start)
        if err or status_code != StatusCode.OK:
            return _handle_error(err, status_code)

        response = jsonify(action)
        return make_response(response, StatusCode.OK)

    except Exception as ex:
        logger.error('Error while processing prediction', exc_info=ex)
        return _handle_error(ErrorSet.UNKNOWN_ERROR, StatusCode.INTERNAL_SERVER_ERROR)


def patch_instances(
        lstm_states: Optional[Tuple[np.ndarray, ...]],
        episode_start: np.ndarray) -> Tuple[Optional[ErrorEntry], int]:
    """
    This method will patch the model, lstm states and episode start
    """
    data: Dict[str, Any] = {
        'lstm_states': [state.tolist() for state in lstm_states] if lstm_states is not None else None,
        'episode_start': episode_start
    }

    try:
        storage_client = storage.Client()
        bucket = storage_client.bucket(BUCKET_NAME)
        blob = bucket.blob('data/instances.json')
        blob.upload_from_string(json.dumps(data))

        logger.info('LSTM states and episode start patched successfully')
        return None, StatusCode.OK

    except NotFound as ex:
        logger.error(f'Instances file or bucket not found, {repr(ex)}', exc_info=True)
        return ErrorSet.INSTANCE_NOT_FOUND, StatusCode.NOT_FOUND
    except Forbidden as ex:
        logger.error(f'Permission denied to update instances, {repr(ex)}', exc_info=True)
        return ErrorSet.INSTANCES_FORBIDDEN, StatusCode.FORBIDDEN
    except Conflict as ex:
        logger.error(f'Failed to update instances due to conflict error, {repr(ex)}', exc_info=True)
        return ErrorSet.INSTANCES_CONFLICT, StatusCode.CONFLICT
    except Exception as ex:
        logger.error(f'Unknown error while updating instances, {repr(ex)}', exc_info=True)
        return ErrorSet.UNKNOWN_ERROR, StatusCode.INTERNAL_SERVER_ERROR


def _get_model_object() -> Tuple[Optional[RecurrentPPOAgent], Optional[ErrorEntry], int]:
    """
    Retrieves model object from GCS
    """

    global model
    if model is not None:
        logger.info('Retrieved model object from previous request')
        return model, None, StatusCode.OK

    logger.info('Downloading model from GCS')
    storage_client = storage.Client()
    bucket = storage_client.bucket(BUCKET_NAME)
    blob = bucket.blob(f'{DIRECTORY}/{MODEL_FILE}')

    try:
        blob.download_to_filename(MODEL_FILE)
        logger.info('Model downloaded successfully')

        logger.info('Setting up environment')
        env = make_vec_env('LayerEnv-v0', n_envs=1)
        logger.info('Environment loaded successfully')

        model = RecurrentPPOAgent.load(MODEL_FILE, env=env)
        logger.info('Successfully retrieved and setup model')

        return model, None, StatusCode.OK

    except NotFound as ex:
        logger.error(f'Model file not found, {repr(ex)}', exc_info=True)
        return None, ErrorSet.MODEL_NOT_FOUND, StatusCode.NOT_FOUND
    except Forbidden as ex:
        logger.error(f'Permission denied to download model, {repr(ex)}', exc_info=True)
        return None, ErrorSet.MODEL_FORBIDDEN, StatusCode.FORBIDDEN
    except Conflict as ex:
        logger.error(f'Failed to download model due to conflict error, {repr(ex)}', exc_info=True)
        return None, ErrorSet.MODEL_CONFLICT, StatusCode.CONFLICT
    except Exception as ex:
        logger.error(f'Unknown error while downloading model: {repr(ex)}', exc_info=True)
        return None, ErrorSet.UNKNOWN_ERROR, StatusCode.INTERNAL_SERVER_ERROR



def _get_instances() -> Tuple[Tuple[np.ndarray], bool, Optional[ErrorEntry], int]:
    """
    Method to retrieve lstm states and episode start from GCS.
    Returns: Tuple of lstm states, episode start and error message
    """

    global lstm_states, episode_start
    if lstm_states is not None and episode_start is not None:
        logger.info('Retrieve instances from previous request')
        return lstm_states, episode_start, None, StatusCode.OK

    logger.info('Downloading instances from GCS')

    storage_client = storage.Client()
    bucket = storage_client.bucket(BUCKET_NAME)
    blob = bucket.blob('data/instances.json')

    try:
        blob.download_to_filename('instances.json')
        logger.info('Instances downloaded successfully')

        with open('instances.json', 'r') as file:
            data = json.load(file)
            logger.info(f'Instances data: {data}')
            data_states = data.get('lstm_states', None)
            if data_states is not None:
                lstm_states = tuple(np.array(state) for state in data_states)
            else:
                lstm_states = None
            episode_start = data.get('episode_start', None)

            return lstm_states, episode_start, None, StatusCode.OK

    except NotFound as ex:
        logger.error(f'Instances file not found, {repr(ex)}', exc_info=True)
        # returning a server error since the instances file should always be present
        return None, None, ErrorSet.INSTANCE_NOT_FOUND, StatusCode.INTERNAL_SERVER_ERROR
    except Forbidden as ex:
        logger.error(f'Permission denied to download instances, {repr(ex)}', exc_info=True)
        return None, None, ErrorSet.INSTANCES_FORBIDDEN, StatusCode.FORBIDDEN
    except Conflict as ex:
        logger.error(f'Failed to download instances due to conflict error, {repr(ex)}', exc_info=True)
        return None, None, ErrorSet.INSTANCES_CONFLICT, StatusCode.CONFLICT
    except Exception as ex:
        logger.error(f'Unknown error while downloading instances: {ex}', exc_info=True)
        return None, None, ErrorSet.UNKNOWN_ERROR, StatusCode.INTERNAL_SERVER_ERROR


def _handle_error(error: ErrorEntry, status_code: StatusCode):
    """
    This method will handle errors and return a response
    """

    response = jsonify({'error': error.to_dict()})
    return make_response(response, status_code)
