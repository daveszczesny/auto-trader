from gymnasium.envs.registration import register

register(
    id='ForexEnv-v0',
    entry_point='env.simpleforex:SimpleForexEnv',
    kwargs={'data': 'resources/data.csv'}
)
