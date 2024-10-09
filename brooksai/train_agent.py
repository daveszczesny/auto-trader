from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3 import PPO


# pylint: disable=import-error
from agent.ppo_agent import PPOAgent
from agent.recurrentppoagent import RecurrentPPOAgent
# pylint: disable=import-error, disable=unused-import
from brooksai.scripts import register_env

MODEL_PATH = "ppo_forex.zip"

# Create the environment
print("Creating Environment")
env = make_vec_env('ForexEnv-v0', n_envs=1)
print("Environment created")
# Create agent

# try:
#     print("Loading existing agent...")
#     model = PPO.load(MODEL_PATH, env=env)
#     print("Agent loaded")
# except FileNotFoundError:
#     print("No agent file found...")
#     print("Creating agent")
#     model = PPOAgent(env)
#     print("Agent created")
# except Exception as _: # pylint: disable=broad-exception-caught
#     print("Error loading agent")
#     print("Creating new agent")
#     model = PPOAgent(env)
#     print("Agent created")


model = RecurrentPPOAgent(env)

# Train the agent
for _ in range(1):
    model.learn(total_timesteps=500_000)
    model.save('ppo_forex')
