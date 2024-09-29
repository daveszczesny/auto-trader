"""
ppo_agent.py

This module defines the PPOAgent class, which is used to train
    and manage a PPO (Proximal Policy Optimization) agent for forex trading.

Classes:
    PPOAgent: A class that encapsulates the PPO agent, providing methods
        for training, saving, and loading the agent.

Dependencies:
    - gymnasium
    - stable_baselines3
    - typing
    - env.models.constants (ActionType, action_type_mapping)
    - env.models.action (Action, TradeAction)
    - env.models.trade (get_trade_by_id, open_trades, Trade)

Usage:
    from ppo_agent import PPOAgent

    # Initialize the environment
    env = gym.make('YourEnv-v0')

    # Create a PPOAgent instance
    agent = PPOAgent(env)

    # Train the agent
    agent.learn(total_timesteps=4_000_000)

    # Save the agent
    agent.save('path_to_save_model')

    # Load the agent
    agent.load('path_to_load_model')

Classes:
    PPOAgent: A class that encapsulates the PPO agent, providing methods
        for training, saving, and loading the agent.

Methods:
    __init__(self, env: gym.Env): Initializes the PPOAgent with the given environment.
    learn(self, total_timesteps: int = 4_000_000): Trains the PPO agent for
        the specified number of timesteps.
    save(self, path: str): Saves the trained PPO agent to the specified path.
    load(self, path: str): Loads a PPO agent from the specified path.
"""

import gymnasium as gym
from stable_baselines3 import PPO

class PPOAgent:
    """
    A class to encapsulate the PPO agent, providing methods for training, 
        saving, and loading the agent.

    Attributes:
        env (gym.Env): The environment in which the agent will be trained.
        model (PPO): The PPO model used for training the agent.

    Methods:
        __init__(self, env: gym.Env): Initializes the PPOAgent with the given environment.
        learn(self, total_timesteps: int = 4_000_000): Trains the PPO agent for the 
            specified number of timesteps.
        save(self, path: str): Saves the trained PPO agent to the specified path.
        load(self, path: str): Loads a PPO agent from the specified path.
    """

    def __init__(self, env: gym.Env):
        """
        Initializes the PPOAgent with the given environment.

        Args:
            env (gym.Env): The environment in which the agent will be trained.
        """
        self.env = env
        self.model = PPO('MlpPolicy', env, verbose=1)

    def learn(self, total_timesteps: int = 4_000_000):
        """
        Trains the PPO agent for the specified number of timesteps.

        Args:
            total_timesteps (int): The number of timesteps to train the agent.
            Default is 4,000,000.
        """
        self.model.learn(total_timesteps=total_timesteps)

    def save(self, path: str):
        """
        Saves the trained PPO agent to the specified path.

        Args:
            path (str): The path where the model will be saved.
        """
        self.model.save(path)

    def load(self, path):
        """
        Loads a PPO agent from the specified path.

        Args:
            path (str): The path from which the model will be loaded.
        """
        self.model = PPO.load(path)

    def predict(self, observation):
        """
        Predicts the action to take based on the given observation.

        Args:
            observation: The observation from the environment.
        """
        raw_action, _ = self.model.predict(observation)
        return raw_action
