import os
import logging
import time

import torch

import dask.dataframe as dd
import numpy as np

from stable_baselines3.common.env_util import make_vec_env

from brooksai.agent.recurrentppoagent import RecurrentPPOAgent
from brooksai.env.scripts import register_env

MODEL_PATH = "ppo_forex.zip"
SAVE_FREQ  = 100_000
CYCLES = 1_000
PARTITIONS = 150

logging.basicConfig(level=logging.INFO, format='%(name)s - %(message)s')
logger = logging.getLogger('AutoTrader')


def format_time(seconds: int) -> str:
    weeks = seconds // (7 * 24 * 3600)
    seconds %= (7 * 24 * 3600)
    days = seconds // (24 * 3600)
    seconds %= (24 * 3600)
    hours = seconds // 3600
    seconds %= 3600
    minutes = seconds // 60
    seconds %= 60
    if weeks > 0:
        return f'{weeks} weeks, {days} days, {hours} hours, {minutes} minutes, {seconds} seconds'
    elif days > 0:
        return f'{days} days, {hours} hours, {minutes} minutes, {seconds} seconds'
    elif hours > 0:
        return f'{hours} hours, {minutes} minutes, {seconds} seconds'
    elif minutes > 0:
        return f'{minutes} minutes, {seconds} seconds'
    else:
        return f'{seconds} seconds'

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

    model = None

    logger.info('Partitioning dataset')
    windows = dataset.repartition(npartitions=PARTITIONS)
    logger.info(f'Created {windows.npartitions} partitions')

    total_time = 0

    for i in range(CYCLES):
        start_time = time.time()
        logger.info(f'Starting training cycle {i + 1}')

        window = np.random.choice(windows.to_delayed())
        window = window.select_dtypes(include=[float, int])
        window = window.compute()
        window = torch.tensor(window.values, dtype=torch.float32)

        logger.info(f'Training window size {len(window)}')

        logger.info('Creating environment')
        env = make_vec_env(
            'ForexEnv-v0',
            n_envs=1,
            env_kwargs={
                'data': window,
                'split': True
                }
            )

        if os.path.exists(MODEL_PATH):
            logger.info('Existing model found... loading')
            model = RecurrentPPOAgent.load(MODEL_PATH, env)
            logger.info('Model loaded')
        else:
            logger.info('No existing model found...')
            logger.info('Creating new model...')

            model = RecurrentPPOAgent(env)
            logger.info('Model created')

            model.save(MODEL_PATH)
            logger.info('Model saved')

        model.learn(total_timesteps=len(window))
        logger.info(f'Finished training cycle {i + 1}')
        model.save(MODEL_PATH)
        logger.info(f'Model {i+1} saved')

        end_time = time.time()
        cycle_time = end_time - start_time
        total_time += cycle_time
        avg_time_per_cycle = total_time / (i + 1)
        remaining_cycles = CYCLES - (i + 1)
        eta = avg_time_per_cycle * remaining_cycles

        logger.info(f'Cycle {i + 1} took {cycle_time:.2f} seconds')
        logger.info(f'Estimated time remaining: {format_time(eta)}')


if __name__ == '__main__':
    main()
