import logging
import json

from typing import Optional, Tuple, Dict, Any

from flask import jsonify, make_response
import numpy as np

# pylint: disable=import-error
from google.cloud import storage # pylint: disable=no-name-in-module
from google.api_core.exceptions import NotFound # pylint: disable=no-name-in-module


from stable_baselines3.common.env_util import make_vec_env

# pylint: disable=import-error
# pylint: disable=unused-import
from utils import register_env
from utils.ai.action import construct_action
from utils.ai.observation import construct_observation
from brooksai.agent.recurrentppoagent import RecurrentPPOAgent

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("BrookyAPI")


BUCKET_NAME = 'at-brooky-bucket'
DIRECTORY = 'model'
MODEL_FILE = 'ppo_forex.zip'

model, lstm_states, episode_start = None, None, None


def predict(request):
    global model
    logger.info("Processing prediction request")

    try:
        payload = request.get_json()
        logger.info(f"Received data: {payload}")

        observation, err = construct_observation(payload)
        if err:
            return _handle_bad_request(err)

        model = _get_model_object()
        if not model:
            return _handle_bad_request("Failed to retrieve model")

        lstm_states, episode_start, err = _get_instances()
        if err:
            return _handle_server_error(f'Failed to retrieve instances: {err}')

        action, lstm_states = model.predict(observation, lstm_states, episode_starts=episode_start)
        episode_start = True

        action = construct_action(action)
        logger.debug(f'Agent action: {action}')

        patch_instances(lstm_states, episode_start)

        response = jsonify(action)
        return make_response(response, 200)

    except Exception as ex:
        logger.error('Failed to predict', exc_info=ex)
        return _handle_server_error(ex)


def patch_instances(lstm_states, episode_start):
    """
    This method will patch the model, lstm states and episode start
    """
    data: Dict[str, Any] = {
        "lstm_states": [state.tolist() for state in lstm_states] if lstm_states is not None else None,
        "episode_start": episode_start
    }

    try:
        storage_client = storage.Client()
        bucket = storage_client.bucket(BUCKET_NAME)
        blob = bucket.blob(f"data/instances.json")
        blob.upload_from_string(json.dumps(data))
        logger.info("LSTM states and episode start patched successfully")
        return True
    except Exception as ex:
        logger.error(f"Failed to patch instances: {ex}", exc_info=True)
        return False


def _get_model_object() -> RecurrentPPOAgent:
    """
    Retrieves model object from GCS
    """

    global model
    if model is not None:
        logger.info('Retreived model object from previous request')
        return model

    logger.info("Downloading model from GCS")
    storage_client = storage.Client()
    bucket = storage_client.bucket(BUCKET_NAME)
    blob = bucket.blob(f"{DIRECTORY}/{MODEL_FILE}")

    try:
        blob.download_to_filename(MODEL_FILE)
        logger.info("Model downloaded successfully")

        logger.info("Setting up environment")
        env = make_vec_env('LayerEnv-v0', n_envs=1)
        logger.info("Environment loaded successfully")

        model = RecurrentPPOAgent.load(MODEL_FILE, env=env)

        logger.info("Successfully retrieved and setup model")

    except NotFound as ex:
        logger.error(f'Model file not found, {repr(ex)}', exc_info=True)
        return None
    except Exception as ex:
        logger.error(f'Unknown error while downloading model: {repr(ex)}', exc_info=True)
        return None

    return model



def _get_instances() -> Tuple[Tuple[np.ndarray], bool, Optional[str]]:
    """
    Method to retrieve lstm states and episode start from GCS.
    Returns: Tuple of lstm states, episode start and error message
    """

    global lstm_states, episode_start
    if lstm_states is not None and episode_start is not None:
        logger.info('Retrieve instances from previous request')
        return lstm_states, episode_start, None

    logger.info('Downloading instances from GCS')

    storage_client = storage.Client()
    bucket = storage_client.bucket(BUCKET_NAME)
    blob = bucket.blob(f"data/instances.json")

    try:
        blob.download_to_filename("instances.json")
        logger.info("Instances downloaded successfully")

        with open("instances.json", "r") as file:
            data = json.load(file)
            logger.info(f'Instances data: {data}')
            data_states = data.get('lstm_states', None)
            if lstm_states is not None:
                lstm_states = tuple(np.array(state) for state in data_states)
            else:
                lstm_states = None
            episode_start = data.get("episode_start", None)
    except NotFound as ex:
        logger.error(f'Instances file not found, {repr(ex)}', exc_info=True)
        lstm_states = None
        episode_start = None
        return None, None, 'Instances file not found'
    except Exception as ex:
        logger.error(f'Unknown error while downloading instances: {ex}', exc_info=True)
        return None, None, f'Unknown error while downloading instances: {ex}'


    return lstm_states, episode_start, None


# 400 errors
def _handle_bad_request(message):
    """
    This method will handle bad request
    """

    response = jsonify({'error': message})
    return make_response(response, 400)

# 500 errors
def _handle_server_error(exc: Exception):

    response = jsonify({'exception': repr(exc)})
    return make_response(response, 500)
