import os

from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import CheckpointCallback
from brooksai.agent.recurrentppoagent import RecurrentPPOAgent
from brooksai.env.scripts import register_env # pylint: disable=unused-import

MODEL_PATH = "ppo_forex.zip"
SAVE_FREQ  = 100_000


def main():
    env = make_vec_env('ForexEnv-v0', n_envs=1)

    model = None

    # check if model exists
    if os.path.exists(MODEL_PATH):
        model = RecurrentPPOAgent.load(MODEL_PATH, env)
    else:
        model = RecurrentPPOAgent(env)
        model.save(MODEL_PATH)

    checkpoint_callback = CheckpointCallback(
        save_freq=SAVE_FREQ,
        save_path='models/',
        name_prefix='model')

    for _ in range(10):
        model.learn(total_timesteps=5_000_000,
                    callback=checkpoint_callback)
        model.save(MODEL_PATH)

if __name__ == '__main__':
    main()
