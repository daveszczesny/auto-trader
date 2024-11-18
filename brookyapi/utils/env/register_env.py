from gymnasium.envs.registration import register


register(
    id='LayerEnv-v0',
    entry_point='utils.env.env:LayerEnv'
)
