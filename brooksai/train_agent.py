import os

from stable_baselines3.common.env_util import make_vec_env

from brooksai.agent.recurrentppoagent import RecurrentPPOAgent
from brooksai.env.scripts import register_env # pylint: disable=unused-import

MODEL_PATH = "ppo_forex.zip"
SAVE_FREQ  = 100_000


def main():
    env = make_vec_env('ForexEnv-v0', n_envs=1)

    model = None

    # check if model exists
    if os.path.exists(MODEL_PATH):
        print('Existing model found... loading')
        model = RecurrentPPOAgent.load(MODEL_PATH, env)
        print('Loaded model')
    else:
        print('No existing model found... creating new model')
        model = RecurrentPPOAgent(env)
        model.save(MODEL_PATH)
        print('Model created and saved')

    for _ in range(10):
        print('Learning...')
        model.learn(total_timesteps=5_000_000)
        model.save(MODEL_PATH)

if __name__ == '__main__':
    main()
