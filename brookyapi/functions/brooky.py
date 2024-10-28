import logging
import json

from typing import Dict, Any

from flask import jsonify, make_response
import numpy as np

# pylint: disable=import-error
from google.cloud import storage # pylint: disable=no-nmae-in-module

from stable_baselines3.common.env_util import make_vec_env

# pylint: disable=import-error
# pylint: disable=unused-import
from utils import register_env
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
        context = request.get_json()
        logger.info(f"Received data: {context}")

        # retreive observation
        observation = context.get('observation', None)
        if observation is None:
            return _handle_bad_request("Missing observation from request body")
        observation = np.array(observation)

        if not _validate_observation(observation):
            return _handle_bad_request("Invalid observation")

        model = _get_model_object()
        if not model:
            return _handle_bad_request("Failed to retrieve model")

        lstm_states, episode_start = _get_instances()

        action, lstm_states = model.predict(observation, lstm_states, episode_starts=episode_start)
        episode_start = True

        action = _convert_action(action)
        logger.debug(f'Agent action: {action}')

        patch_instances(lstm_states, episode_start)

        response = jsonify({'action': action.tolist()})
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
    global model
    if model is not None:
        logger.info('Retreived model object from previous request')
        return model

    logger.info("Downloading model from GCS")
    storage_client = storage.Client()
    bucket = storage_client.bucket(BUCKET_NAME)
    blob = bucket.blob(f"{DIRECTORY}/{MODEL_FILE}")
    blob.download_to_filename(MODEL_FILE)
    logger.info("Model downloaded successfully")

    logger.info("Setting up environment")
    env = make_vec_env('LayerEnv-v0', n_envs=1)
    logger.info("Environment loaded successfully")

    model = RecurrentPPOAgent.load(MODEL_FILE, env=env)

    logger.info("Successfully retrieved and setup model")

    return model

def _convert_action(action):
    return action

def _get_instances():
    """
    This method will return the model, lstm states and episode start
    """

    global lstm_states, episode_start
    if lstm_states is not None and episode_start is not None:
        logger.info('Retrieve instances from previous request')
        return lstm_states, episode_start

    logger.info('Downloading instances from GCS')

    storage_client = storage.Client()
    bucket = storage_client.bucket(BUCKET_NAME)
    blob = bucket.blob(f"data/instances.json")
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

    return lstm_states, episode_start

def _validate_observation(observation):
    """
    This method will validate the observation
    """
    return True


def _handle_bad_request(message):
    """
    This method will handle bad request
    """

    response = jsonify({'error': message})
    return make_response(response, 400)

def _handle_server_error(exc: Exception):

    response = jsonify({'exception': repr(exc)})
    return make_response(response, 500)
