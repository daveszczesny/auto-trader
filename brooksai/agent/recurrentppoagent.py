"""
@article{stable-baselines3,
  author  = {Antonin Raffin and Ashley Hill and Adam Gleave and Anssi Kanervisto and 
    Maximilian Ernestus and Noah Dormann},
  title   = {Stable-Baselines3: Reliable Reinforcement Learning Implementations},
  journal = {Journal of Machine Learning Research},
  year    = {2021},
  volume  = {22},
  number  = {268},
  pages   = {1-8},
  url     = {http://jmlr.org/papers/v22/20-1364.html}
}
"""

try:
    import gymnasium as gym
except Exception as _:
    pass

import logging
from typing import Any

import numpy as np

from sb3_contrib import RecurrentPPO
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList
from stable_baselines3.common.monitor import Monitor

from brooksai.config_manager import ConfigManager
from brooksai.agent.callbacks.performance_callback import EvaluatePerformanceCallback

config = ConfigManager()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RecurrentPPOAgent:
    """
    A class that encapsulates the RecurrentPPO model and provides an interface for training and prediction.

    Controls hyperparameters and model architecture for the RecurrentPPO model.
    """

    def __init__(self,
        env: gym.Env | Any,
        log_dir: str = 'runs/ppo_recurrent',
        lstm_hidden_size: int = config.get('model.lstm_hidden_size', 512),
        n_lstm_layers: int = config.get('model.n_lstm_layers', 2),
        batch_size: int = config.get('model.batch_size', 1024),
        gamma: float = config.get('model.gamma', 0.95),
        learning_rate: float = config.get('model.learning_rate', 1e-4),
        gae_lambda: float = config.get('model.gae_lambda', 0.95),
        ent_coef: float = config.get('model.ent_coef', 0.5),
        sde_sample_freq: int = config.get('model.sde_sample_freq', 16)):

        self.seed = np.random.seed(42)

        self.env = Monitor(env)
        self.model = RecurrentPPO(
            "MlpLstmPolicy",
            env,
            verbose=0,
            n_steps=config.get('model.n_steps', 1024),
            batch_size=batch_size, # larger number reduces variance in learning?
            n_epochs=config.get('model.n_epochs', 10),
            gamma=gamma, # encourages more immediate rewards
            learning_rate=RecurrentPPOAgent.linear_schedule(learning_rate),
            clip_range=config.get('model.clip_range', 0.2),
            gae_lambda=gae_lambda,
            ent_coef=ent_coef, # entropy coefficient
            vf_coef=config.get('model.vf_coef', 0.5), # value function coefficient
            max_grad_norm=config.get('model.max_grad_norm', 0.5), # gradient clipping
            use_sde=True,
            sde_sample_freq=sde_sample_freq, # encourages exploration
            normalize_advantage=True,
            policy_kwargs={"lstm_hidden_size": lstm_hidden_size, "n_lstm_layers": n_lstm_layers},
            tensorboard_log=log_dir,
            seed=self.seed
            )

        self.num_envs = 1
        self.log_dir = log_dir

    def learn(
            self, 
              total_timesteps: int, 
              evaluate: bool = False
            ):
        """
        Train the model on the environment for a specified number of timesteps.
        :param total_timesteps: The total number of timesteps to train the model for.
        """

        if evaluate:
            evaluate_performance_callback = EvaluatePerformanceCallback(
                self.env,
                eval_freq=total_timesteps-1,
                verbose=1
            )

            self.model.learn(total_timesteps, tb_log_name="ppo_recurrent", callback=evaluate_performance_callback)
        else:
            self.model.learn(total_timesteps, tb_log_name="ppo_recurrent")


    def predict(self, observation=None, **kwargs):
        """
        Predict an action given an observation and LSTM states.

        If deterministic is True, the model will return the action with the highest probability,
        otherwise it will sample an action from the probability distribution.
        For training purposes, deterministic should be False,
        and for evaluation (production) deterministic should be True.

        :param observation: The observation to predict an action for.
             If None, it will be retrieved from kwargs.
        :param lstm_states: The LSTM states to use for prediction. 
            Retrieved from kwargs if not provided.
        :param episode_starts: A boolean array indicating whether each episode has started. 
            Retrieved from kwargs if not provided.
        :param deterministic: Whether to use deterministic or stochastic actions. 
            Default is False.
        :return: A tuple containing the raw action predicted by the model and
            the updated LSTM states.
        :raises ValueError: If observation or episode_starts are not provided.
        """


        if observation is None:
            observation = kwargs.get('observation')

        lstm_states = kwargs.get('lstm_states') or kwargs.get('state')
        episode_starts = kwargs.get('episode_starts')
        if episode_starts is None:
             episode_starts = kwargs.get('episode_start')
        deterministic = kwargs.get('deterministic', False)

        if observation is None or episode_starts is None:
            raise ValueError("observation and episode_starts must be provided.")

        raw_action, lstm_states = self.model.predict(
            observation,
            lstm_states,
            episode_starts,
            deterministic=deterministic
        )

        return raw_action, lstm_states


    @staticmethod
    def linear_schedule(initial_value):
        """
        Linear schedule from initial value to 0.
        :param initial_value: The initial value of the schedule.
        :return: The schedule function.
        """
        def schedule(progress_remaining):
            return initial_value * progress_remaining
        return schedule


    def save(self, path: str):
        self.model.save(path)


    @staticmethod
    def load(path: str, env: gym.Env | Any = None):
        """
        Load a model from a specified path.
        :param path: The path to load the model from.
        :param env: The environment to load the model into.
        :return: The loaded model.
        """

        model = RecurrentPPO.load(path, env=env)
        model.lr_schedule = RecurrentPPOAgent.linear_schedule(model.learning_rate)
        agent = RecurrentPPOAgent(env)
        agent.model = model
        return agent
