
try:
    import gymnasium as gym
except Exception as _:
    pass

from typing import Any

from sb3_contrib import RecurrentPPO
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList
from stable_baselines3.common.monitor import Monitor

MODEL_PATH = "ppo_forex.zip"
SAVE_FREQ  = 50_000

class RecurrentPPOAgent:
    """
    A class that encapsulates the RecurrentPPO model and provides an interface for training and prediction.

    Controls hyperparameters and model architecture for the RecurrentPPO model.
    """

    def __init__(self, env: gym.Env | Any, log_dir: str = 'runs/ppo_recurrent'):
        self.env = Monitor(env)
        self.model = RecurrentPPO(
            "MlpLstmPolicy",
            env,
            verbose=1,
            n_steps=2048,
            batch_size=256, # larger number reduces variance in learning?
            n_epochs=10,
            gamma=0.96, # encourages more immediate rewards
            learning_rate=3e-4,
            clip_range=0.2,
            gae_lambda=0.9,
            ent_coef=0.03, # entropy coefficient
            vf_coef=0.5,
            max_grad_norm=0.5, # gradient clipping
            use_sde=True,
            sde_sample_freq=16, # encourages exploration
            normalize_advantage=True,
            policy_kwargs={"lstm_hidden_size":256, "n_lstm_layers": 2},
            tensorboard_log=log_dir
            )

        self.num_envs = 1
        self.log_dir = log_dir

    def learn(self, total_timesteps: int = 5_000_000):
        """
        Train the model on the environment for a specified number of timesteps.
        :param total_timesteps: The total number of timesteps to train the model for.
        """

        # Save the model every SAVE_FREQ timesteps
        checkpoint_callback = CheckpointCallback(save_freq=SAVE_FREQ, save_path='models/', name_prefix='model')
        callback = CallbackList([checkpoint_callback])

        self.model.learn(total_timesteps, tb_log_name="ppo_recurrent", callback=callback)


    def predict(self, observation, lstm_states, episode_starts):
        """
        Predict an action given an observation and LSTM states.
        :param observation: The observation to predict an action for.
        :param lstm_states: The LSTM states to use for prediction.
        :param episode_starts: A boolean array indicating whether each episode has started.
        :return: The raw action predicted by the model.
        """

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
    def load(path: str, env: gym.Env | Any = None):
        """
        Load a model from a specified path.
        :param path: The path to load the model from.
        :param env: The environment to load the model into.
        :return: The loaded model.
        """

        model = RecurrentPPO.load(path, env=env)
        agent = RecurrentPPOAgent(env)
        agent.model = model
        return agent
