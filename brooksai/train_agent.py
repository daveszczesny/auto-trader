import os
import logging

from stable_baselines3.common.env_util import make_vec_env

from brooksai.agent.recurrentppoagent import RecurrentPPOAgent
from brooksai.env.scripts import register_env

MODEL_PATH = "ppo_forex.zip"
SAVE_FREQ  = 100_000
CYCLES = 10
TIMESTEPS_PER_CYCLE = 5_000_000

logging.basicConfig(level=logging.INFO, format='%(name)s - %(message)s')
logger = logging.getLogger('AutoTrader')

def main():
    env = make_vec_env('ForexEnv-v0', n_envs=1)

    model = None

    logger.info(r'''

 $$$$$$\              $$\            $$$$$$$$\                       $$\                           $$\                       $$$$$$$\                                
$$  __$$\             $$ |           \__$$  __|                      $$ |                          $$ |                      $$  __$$\                               
$$ /  $$ |$$\   $$\ $$$$$$\    $$$$$$\  $$ | $$$$$$\  $$$$$$\   $$$$$$$ | $$$$$$\   $$$$$$\        $$$$$$$\  $$\   $$\       $$ |  $$ | $$$$$$\ $$\    $$\  $$$$$$\  
$$$$$$$$ |$$ |  $$ |\_$$  _|  $$  __$$\ $$ |$$  __$$\ \____$$\ $$  __$$ |$$  __$$\ $$  __$$\       $$  __$$\ $$ |  $$ |      $$ |  $$ | \____$$\\$$\  $$  |$$  __$$\ 
$$  __$$ |$$ |  $$ |  $$ |    $$ /  $$ |$$ |$$ |  \__|$$$$$$$ |$$ /  $$ |$$$$$$$$ |$$ |  \__|      $$ |  $$ |$$ |  $$ |      $$ |  $$ | $$$$$$$ |\$$\$$  / $$$$$$$$ |
$$ |  $$ |$$ |  $$ |  $$ |$$\ $$ |  $$ |$$ |$$ |     $$  __$$ |$$ |  $$ |$$   ____|$$ |            $$ |  $$ |$$ |  $$ |      $$ |  $$ |$$  __$$ | \$$$  /  $$   ____|
$$ |  $$ |\$$$$$$  |  \$$$$  |\$$$$$$  |$$ |$$ |     \$$$$$$$ |\$$$$$$$ |\$$$$$$$\ $$ |            $$$$$$$  |\$$$$$$$ |      $$$$$$$  |\$$$$$$$ |  \$  /   \$$$$$$$\ 
\__|  \__| \______/    \____/  \______/ \__|\__|      \_______| \_______| \_______|\__|            \_______/  \____$$ |      \_______/  \_______|   \_/     \_______|
                                                                                                             $$\   $$ |                                              
                                                                                                             \$$$$$$  |                                              
                                                                                                              \______/                                               
''')

    # check if model exists
    if os.path.exists(MODEL_PATH):
        logger.info('Existing model found... loading')
        model = RecurrentPPOAgent.load(MODEL_PATH, env)
        logger.info('Model loaded')
    else:
        logger.info('No existing model found...')
        logger.info('Creating new model...')

        model = RecurrentPPOAgent(env)
        logger.info('Model created')

        model.save(MODEL_PATH)
        logger.info('Model saved')


    logger.info(f'Training model with {CYCLES} cycles, and {TIMESTEPS_PER_CYCLE} total timesteps per cycle...')
    for i in range(CYCLES):
        logger.info(f'Starting training cycle {i + 1}')
        model.learn(total_timesteps=TIMESTEPS_PER_CYCLE)
        logger.info(f'Finished training cycle {i + 1}')
        model.save(MODEL_PATH)
        logger.info(f'Model {i+1} saved')

if __name__ == '__main__':
    main()
