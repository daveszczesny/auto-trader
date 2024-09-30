"""
train_agent.py

This script is used to train a PPO (Proximal Policy Optimization) agent for forex trading.

The script performs the following steps:
1. Attempts to load an existing PPO agent from a specified file.
2. If the agent file is not found, it creates a new PPO agent.
3. Trains the agent for a specified number of iterations.
4. Saves the trained agent to a file after each iteration.

Usage:
    python train_agent.py

Dependencies:
    - stable_baselines3
    - gymnasium
    - numpy

Functions:
    None

Classes:
    None

Exceptions:
    - FileNotFoundError: Raised if the agent file is not found
        when attempting to load an existing agent.

Notes:
    - Ensure that the environment `env` and the model path `model_path` 
        are defined before running this script.
    - The script trains the agent for 10 iterations, each with 5,000,000 timesteps.

Example:
    To run the script, simply execute:
    $ python train_agent.py
"""

from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3 import PPO

# pylint: disable=import-error
from agent.ppo_agent import PPOAgent
# pylint: disable=import-error, disable=unused-import
from brooksai.scripts import register_env


MODEL_PATH = "ppo_forex.zip"

# Create the environment
print("Creating Environment")
env = make_vec_env('ForexEnv-v0', n_envs=1)
print("Environment created")
# Create agent

try:
    print("Loading existing agent...")
    model = PPO.load(MODEL_PATH, env=env)
    print("Agent loaded")
except FileNotFoundError:
    print("No agent file found...")
    print("Creating agent")
    model = PPOAgent(env)
    print("Agent created")

# Train the agent
model.learn(total_timesteps=5_000_000)
model.save('ppo_forex')
