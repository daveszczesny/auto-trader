
import gymnasium as gym
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList
from stable_baselines3.common.monitor import Monitor

MODEL_PATH = "ppo_forex.zip"
SAVE_FREQ  = 50_000

class RecurrentPPOAgent:
    def __init__(self, env: gym.Env, log_dir: str = 'runs/ppo_recurrent'):
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

    def learn(self, total_timesteps: int = 4_000_000):
        checkpoint_callback = CheckpointCallback(save_freq=SAVE_FREQ, save_path='models/', name_prefix='model')

        callback = CallbackList([checkpoint_callback])

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
