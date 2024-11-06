from typing import List

import numpy as np

from sklearn.model_selection import ParameterGrid

from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy

from brooksai.agent.recurrentppoagent import RecurrentPPOAgent
from brooksai.env.scripts import register_env

param_grid = {
    'batch_size': [256, 512, 1024],
    'gamma': [0.9, 0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99],
    'learning_rate': [1e-4, 3e-4, 5e-4, 7e-4, 9e-4],
    'gae_lambda': [0.8, 0.85, 0.9, 0.95, 0.99],
    'ent_coef': [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1],
    'sde_sample_freq': [8, 16, 32, 64],
    'lstm_hidden_size': [64, 128, 256, 512],
    'n_lstm_layers': [1, 2, 3]
}

num_combinations = len(list(ParameterGrid(param_grid)))
print(f'Total number of combinations to run: {num_combinations}')


param_combinations = list(ParameterGrid(param_grid))

env = make_vec_env('ForexEnv-v0', n_envs=1)

attempts: List[str] = []

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
    agent.learn(total_timesteps=300_000)

    episode_rewards, episode_lengths = evaluate_policy(agent.model, env, n_eval_episodes=20, return_episode_rewards=True)

    mean_reward = np.mean(episode_rewards)
    mean_length = np.mean(episode_lengths)

    print(f'Evaluation results for params {params}')
    print(f'Mean reward: {mean_reward}, Mean Episode: {mean_length}\n')

    attempts.append(
        f'Params: {params}, Mean Reward: {mean_reward}, Mean Episode: {mean_length}'
    )

print('All attempts:')
for attempt in attempts:
    print(attempt)
