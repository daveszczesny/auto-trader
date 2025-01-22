import os
import sys
import logging

import torch
import dask.dataframe as dd
from sklearn.model_selection import KFold

from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecEnv

from brooksai.config_manager import ConfigManager
from brooksai.agent.recurrentppoagent import RecurrentPPOAgent
from brooksai.env.scripts import register_env

sys.setrecursionlimit(10**6)

config = ConfigManager()

CYCLES = config.get('training.cycles', 1_000)
PARTITIONS = config.get('training.partitions', 50)

best_model_base_path: str = config.get('training.best_model_path', 'best_models/')
best_model_path = best_model_base_path + 'best_model_cycle_1.zip'

logging.basicConfig(level=logging.INFO, format='%(name)s - %(message)s')
logger = logging.getLogger('AutoTrader Trainer')

def load_dataset() -> dd.DataFrame:
    return dd.read_csv('resources/training_data2.csv')

def partition_dataset(dataset: dd.DataFrame):
    return dataset.repartition(npartitions=PARTITIONS).to_delayed()

def run_model(agent, env, window, evaluate: bool):
    logger.info(f'Commencing training with {len(window)} data points')
    agent.env = env
    agent.learn(total_timesteps=len(window), evaluate=evaluate)

def load_agent(env):
    no_best_models_saved = len([name for name in os.listdir(best_model_base_path) \
                        if os.path.isfile(os.path.join(best_model_base_path, name)) and \
                            name.endswith('.zip')])

    if no_best_models_saved >= 1:
        best_model_path = best_model_base_path + f'best_model_cycle_{no_best_models_saved}.zip'
    
    if os.path.exists(best_model_path):
        logger.info(f'Found existing agent model at {best_model_path}. Loading...')
        agent = RecurrentPPOAgent.load(best_model_path, env)
    else:
        logger.info('No existing model found. Creating new agent...')
        agent = RecurrentPPOAgent(env)
        agent.saved(best_model_path)

    logger.info('Agent loaded')
    return agent

def configure_env(window):
    window_ = window.select_dtypes(include=[float, int])
    window_ = window_.compute()
    window_ = torch.tensor(window_.values, dtype=torch.float32)

    model_n_steps = config.get('model.n_steps', 1024)
    length = len(window_)
    if length % model_n_steps != 0:
        new_length = (length // model_n_steps) * model_n_steps
        window_ = window_[:new_length]

    logger.info('Creating environment')

    env = make_vec_env(
        'ForexEnv-v0',
        n_envs=1,
        env_kwargs={
            'data': window_,
            'split': True
        },
    )

    return env, window_




logger.info('Loading dataset')
dataset: dd.DataFrame = load_dataset()
logger.info('Dataset loaded')

logger.info('Partitioning dataset')
partionted_dataset = partition_dataset(dataset)
logger.info(f'Created {len(partionted_dataset)} partitions')

# number of folds
n_folds = 5
kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)

for fold, (train_index, val_index) in enumerate(kf.split(partionted_dataset)):
    train_windows = [partionted_dataset[i] for i in train_index]
    val_windows = [partionted_dataset[i] for i in val_index]

    env, _ = configure_env(train_windows[0])
    agent = load_agent(env)

    logger.info(f'Starting fold {fold + 1}')

    for window in train_windows:
        # Train model
        env, window_ = configure_env(window)
        run_model(agent, env, window_, False)

    for window in val_windows:
        env, window_ = configure_env(window)
        run_model(agent, env, window_, True)

