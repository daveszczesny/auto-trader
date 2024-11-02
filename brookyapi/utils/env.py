from typing import Tuple, Dict, Any

import numpy as np

import gymnasium as gym
from gymnasium import spaces

class LayerEnv(gym.Env):

    def __init__(self, render_mode = None):

        self.render_mode = render_mode

        self.observation_space = spaces.Box(
            low=0.0,
            high=np.inf,
            shape=(9,),
            dtype=np.float32
        )

        self.action_space = spaces.Box(
            low = np.array([0.0, 0, 0, 0], dtype=np.float32),
            high = np.array([1.0, 1.0, 1.0, 1.0],
                            dtype=np.float32),
            dtype=np.float32
        )

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        return self._get_observation(), 0.0, False, False, {}

    def _get_observation(self) -> np.ndarray:
        return np.zeros(9)

    def reset(self, *, seed = None, options = None):
        super().reset(seed=seed, options=options)

        return self._get_observation(), {}

    def render(self, mode: str = 'human') -> None:
        pass
