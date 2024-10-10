
import gymnasium as gym
from sb3_contrib import RecurrentPPO

class RecurrentPPOAgent:
    def __init__(self, env: gym.Env):
        self.env = env
        self.model = RecurrentPPO(
            "MlpLstmPolicy",
            env,
            verbose=1,
            policy_kwargs=dict(lstm_hidden_size=385, n_lstm_layers=2)
            )
        self.num_envs = 1

    def learn(self, total_timesteps: int = 4_000_000):
        self.model.learn(total_timesteps, tb_log_name="ppo_recurrent")


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

    def load(self):
        self.model.load("ppo_recurrent")
