import os
import sys


# change recursion limit to max
sys.setrecursionlimit(10**6)

import logging
import time

import torch

import dask.dataframe as dd
import numpy as np

from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv

from brooksai.agent.recurrentppoagent import RecurrentPPOAgent
from brooksai.utils.action import ActionApply
from brooksai.env.scripts import register_env

from brooksai.utils.format import format_time

CYCLES = 1_000
PARTITIONS=1_800

logging.basicConfig(level=logging.INFO, format='%(name)s - %(message)s')
logger = logging.getLogger('AutoTrader')

best_model_base_path: str = 'best_models/'
best_model_path = best_model_base_path + 'best_model_cycle_1.zip'
best_performance = 800

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


def run_model(window, start_time, i):
    global best_model_path, best_performance, total_time

    for j in range(5):
        no_best_models_saved = len([name for name in os.listdir(best_model_base_path) if os.path.isfile(os.path.join(best_model_base_path, name))])

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
        logger.info(f'\nCommencing training with {len(window_)} data points')
        model.learn(total_timesteps=len(window_))
        logger.info(f'Finished training cycle {j + 1} '
                    'Evaluating Model performance....')

        performance = evaluate_model(env)
        if performance > best_performance:
            best_performance = performance
            best_model_path = best_model_base_path + f'best_model_cycle_{no_best_models_saved+1}.zip'
            model.save(best_model_path)

            logger.info(f'New best model saved at {best_model_path}')

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


def configure_env(window):
    window_ = window.select_dtypes(include=[float, int])
    window_ = window_.compute()
    window_ = torch.tensor(window_.values, dtype=torch.float32)

    model_n_steps = 1024
    length = len(window_)
    if length % model_n_steps != 0:
        new_length = (length // model_n_steps) * model_n_steps
        window_ = window_[:new_length]
        logger.info('Window length adjusted to fit model n_steps')

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


def evaluate_model(env):
    """
    Evaluation of the model's performance in the previous run
    Taking into account the final balance, reward and any unrealized pnl
    """

    balance = env.get_attr('current_balance')[0]
    reward = env.get_attr('reward')[0]
    unrealized_pnl = env.get_attr('unrealised_pnl')[0]
    trades_placed = ActionApply.get_action_tracker('trades_opened')
    times_won = ActionApply.get_action_tracker('times_won')
    win_rate = times_won / trades_placed if trades_placed > 0 else 0

    phi = 1

    # if only one trade was placed the win rate could appear as 1, which is not accurate
    if trades_placed <= 1:
        phi = 0

    alpha = 0.7
    beta = 0.4
    gamma = 0.2
    delta = 0.4

    performance =  (alpha * balance) + (beta * reward) + (gamma * unrealized_pnl) + (delta * win_rate)
    performance = phi * performance # if no trades or one trade were placed the performance is 0

    logger.info(f'Evaluation results - balance: {balance}, reward: {reward}, unrealized_pnl: {unrealized_pnl}, win_rate: {win_rate}, trades placed: {trades_placed}')
    logger.info(f'Calculated performance metric: {performance}')

    return performance


if __name__ == '__main__':
    main()
