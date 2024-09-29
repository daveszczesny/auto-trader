from gymnasium.envs.registration import register

register(
    id='ForexEnv-v0',
    entry_point='env.forex:ForexEnv',
    kwargs={'data': 'resources/training_data.csv'}
)