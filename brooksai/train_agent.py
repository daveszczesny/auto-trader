import os
import sys
import logging
import time

from typing import Tuple

import torch
import dask.dataframe as dd

from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecEnv

from brooksai.agent.recurrentppoagent import RecurrentPPOAgent
from brooksai.env.scripts import register_env
from brooksai.utils.format import format_time

# change recursion limit to max
sys.setrecursionlimit(10**6)

CYCLES = 1_000
PARTITIONS = 50

logging.basicConfig(level=logging.INFO, format='%(name)s - %(message)s')
logger = logging.getLogger('AutoTrader')

best_model_base_path: str = 'best_models/'
best_model_path = best_model_base_path + 'best_model_cycle_1.zip'

total_time = 0

def main():
    logger.info(r'''

 $$$$$$\              $$\            $$$$$$$$\                       $$\                           $$\                       $$$$$$$\                                
$$  __$$\             $$ |           \__$$  __|                      $$ |                          $$ |                      $$  __$$\                               
$$ /  $$ |$$\   $$\ $$$$$$\    $$$$$$\  $$ | $$$$$$\  $$$$$$\   $$$$$$$ | $$$$$$\   $$$$$$\        $$$$$$$\  $$\   $$\       $$ |  $$ | $$$$$$\ $$\    $$\  $$$$$$\  
$$$$$$$$ |$$ |  $$ |\_$$  _|  $$  __$$\ $$ |$$  __$$\ \____$$\ $$  __$$ |$$  __$$\ $$  __$$\       $$  __$$\ $$ |  $$ |      $$ |  $$ | \____$$\\$$\  $$  |$$  __$$\ 
$$  __$$ |$$ |  $$ |  $$ |    $$ /  $$ |$$ |$$ |  \__|$$$$$$$ |$$ /  $$ |$$$$$$$$ |$$ |  \__|      $$ |  $$ |$$ |  $$ |      $$ |  $$ | $$$$$$$ |\$$\$$  / $$$$$$$$ |
$$ |  $$ |$$ |  $$ |  $$ |$$\ $$ |  $$ |$$ |$$ |     $$  __$$ |$$ |  $$ |$$   ____|$$ |            $$ |  $$ |$$ |  $$ |      $$ |  $$ |$$  __$$ | \$$$  /  $$   ____|
$$ |  $$ |\$$$$$$  |  \$$$$  |\$$$$$$  |$$ |$$ |     \$$$$$$$ |\$$$$$$$ |\$$$$$$$\ $$ |            $$$$$$$  |\$$$$$$$ |      $$$$$$$  |\$$$$$$$ |  \$  /   \$$$$$$$\ 
\__|  \__| \______/    \____/  \______/ \__|\__|      \_______| \_______| \_______|\__|            \_______/  \____$$ |      \_______/  \_______|   \_/     \_______|
                                                                                                             $$\   $$ |                                              
                                                                                                             \$$$$$$  |                                              
                                                                                                              \______/                                               
''')

    logger.info('Loading dataset')
    dataset = dd.read_csv('resources/training_data2.csv')
    logger.info('Dataset loaded')

    logger.info('Partitioning dataset')
    windows = dataset.repartition(npartitions=PARTITIONS)
    logger.info(f'Created {windows.npartitions} partitions')


    for i in range(CYCLES):
        start_time = time.time()
        logger.info(f'Starting training cycle {i + 1}')

        for window in windows.to_delayed():
            run_model(window, start_time, i)


def run_model(window, start_time, i) -> None:
    global best_model_path, total_time

    for _ in range(5):
        no_best_models_saved = len([name for name in os.listdir(best_model_base_path) \
                        if os.path.isfile(os.path.join(best_model_base_path, name)) and \
                            name.endswith('.zip')])

        if no_best_models_saved >= 1:
            best_model_path = best_model_base_path + f'best_model_cycle_{no_best_models_saved}.zip'

        env, window_ = configure_env(window)

        # Load exiting model or create new one
        if os.path.exists(best_model_path):
            logger.info(f'Existing model found at {best_model_path}... loading')
            model = RecurrentPPOAgent.load(best_model_path, env)
            logger.info('Model loaded')
        else:
            logger.info('No existing model found... '
                        'Creating new model...')

            model = RecurrentPPOAgent(env)
            logger.info('Model created')

            model.save(best_model_path)
            logger.info('Model saved')

        # Commence training
        logger.info(f'Commencing training with {len(window_)} data points')
        model.learn(total_timesteps=len(window_))
        env.close()


    # Calculate time taken
    end_time = time.time()
    cycle_time = end_time - start_time
    total_time += cycle_time
    avg_time_per_cycle = total_time / (i + 1)
    remaining_cycles = CYCLES - (i + 1)
    eta = avg_time_per_cycle * remaining_cycles

    logger.info(f'Cycle {i + 1} out of {CYCLES} took {cycle_time:.2f} seconds')
    logger.info(f'Estimated time remaining: {format_time(eta)}')


def configure_env(window) -> Tuple[VecEnv, torch.Tensor]:
    window_ = window.select_dtypes(include=[float, int])
    window_ = window_.compute()
    window_ = torch.tensor(window_.values, dtype=torch.float32)

    model_n_steps = 1024
    length = len(window_)
    if length % model_n_steps != 0:
        new_length = (length // model_n_steps) * model_n_steps
        window_ = window_[:new_length]
        logger.info(f'Window length adjusted to fit model n_steps, new length is {len(window_)} from {length}')

    logger.info(f'Training window size {len(window_)}')

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


if __name__ == '__main__':
    main()
