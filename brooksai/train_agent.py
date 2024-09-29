import gymnasium as gym
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3 import PPO

import register_env as register_env

from agent.ppo_agent import PPOAgent

model_path = "ppo_forex.zip"

# Create the environment
print("Creating Environment")
env = make_vec_env('ForexEnv-v0', n_envs=1)
print("Environment created")
# Create agent

try:
    print("Loading existing agent...")
    model = PPO.load(model_path, env=env)
    print("Agent loaded")
except FileNotFoundError:
    print("No agent file found...")
    print("Creating agent")
    model = PPOAgent(env)
    print("Agent created")

# Train the agent
for i in range(10):
    model.learn(total_timesteps=5_000_000)
    model.save('ppo_forex')
