
import torch

import gymnasium as gym
from sb3_contrib import RecurrentPPO

class RecurrentPPOAgent:
    def __init__(self, env: gym.Env):
        self.env = env
        self.model = RecurrentPPO(
            "MlpLstmPolicy",
            env,
            verbose=1,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            learning_rate=3e-4,
            clip_range=0.2,
            gae_lambda=0.95,
            ent_coef=0.01,
            vf_coef=0.5,
            max_grad_norm=0.5, # gradient clipping
            use_sde=True,
            sde_sample_freq=4,
            policy_kwargs=dict(lstm_hidden_size=256, n_lstm_layers=2)
            )
        self.num_envs = 1

    def learn(self, total_timesteps: int = 4_000_000, callback=None):
        self.model.learn(total_timesteps, tb_log_name="ppo_recurrent", callback=callback)


    def predict(self, observation, lstm_states, episode_starts):
        raw_action, lstm_states = self.model.predict(
            observation,
            lstm_states,
            episode_starts,
            deterministic=True
        )

        return raw_action, lstm_states


    def save(self, path: str):
        self.model.save(path)

    @staticmethod
    def load(path, env):
        model = RecurrentPPO.load(path, env=env)
        agent = RecurrentPPOAgent(env)
        agent.model = model
        return model