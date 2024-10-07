import gymnasium as gym
from stable_baselines3 import PPO


from brooksai.env.models.constants import ApplicationConstants

class PPOAgent:
    def __init__(self, env: gym.Env):
        self.env = env
        self.model = PPO('MlpPolicy', env, device=ApplicationConstants.DEVICE, verbose=1)

    def learn(self, total_timesteps: int = 4_000_000):
        self.model.learn(total_timesteps=total_timesteps)

    def save(self, path: str):
        self.model.save(path)

    def load(self, path):
        self.model = PPO.load(path)

    def predict(self, observation):
        raw_action, _ = self.model.predict(observation)
        return raw_action
