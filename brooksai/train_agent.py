from stable_baselines3.common.env_util import make_vec_env

from brooksai.agent.recurrentppoagent import RecurrentPPOAgent
from brooksai.env.scripts import register_env # pylint: disable=unused-import

MODEL_PATH = "ppo_forex.zip"


def main():
    env = make_vec_env('ForexEnv-v0', n_envs=1)

    model = RecurrentPPOAgent(env)

    model.learn(total_timesteps=2_000_000)
    model.save('ppo_forex')


if __name__ == '__main__':
    main()
