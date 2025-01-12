from typing import List

import numpy as np

from sklearn.model_selection import ParameterGrid

from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy

from brooksai.agent.recurrentppoagent import RecurrentPPOAgent
from brooksai.env.scripts import register_env

param_grid = {
    'batch_size': [1024], # adjust to 512 later if needed
    'gamma': [0.95, 0.99],
    'learning_rate': [1e-4], # controls the step size of the optimizer
    'gae_lambda': [0.95, 0.99],
    'ent_coef': [0.15, 0.3, 0.5], # a higher value encourages exploration
    'sde_sample_freq': [16, 32],
    'lstm_hidden_size': [256, 512],
    'n_lstm_layers': [2, 3]
}

num_combinations = len(list(ParameterGrid(param_grid)))
print(f'Total number of combinations to run: {num_combinations}')


param_combinations = list(ParameterGrid(param_grid))

env = make_vec_env('ForexEnv-v0', n_envs=1)


open('results.txt', 'w').close()

for params in param_combinations:
    print(f'Training model with params: {params}')
    agent = RecurrentPPOAgent(
        env,
        log_dir='runs/ppo_recurrent',
        batch_size=params['batch_size'],
        gamma=params['gamma'],
        learning_rate=params['learning_rate'],
        gae_lambda=params['gae_lambda'],
        ent_coef=params['ent_coef'],
        sde_sample_freq=params['sde_sample_freq'],
        lstm_hidden_size=params['lstm_hidden_size'], n_nstm_layers=params['n_lstm_layers'])
    agent.learn(total_timesteps=50_000)

    episode_rewards, episode_lengths = evaluate_policy(agent, env, n_eval_episodes=20, return_episode_rewards=True)

    mean_reward = np.mean(episode_rewards)
    mean_length = np.mean(episode_lengths)

    print(f'Evaluation results for params {params}')
    print(f'Mean reward: {mean_reward}, Mean Episode: {mean_length}\n')

    with open('results.txt', 'a') as f:
        f.write(f'Params: {params}\n')
        f.write(f'Mean reward: {mean_reward}, Mean Episode: {mean_length}\n\n')
