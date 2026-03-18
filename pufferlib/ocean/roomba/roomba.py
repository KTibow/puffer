import gymnasium
import numpy as np

import pufferlib
from pufferlib.ocean.roomba import binding

WIDTH = 1000  # mm
HEIGHT = 1000  # mm
SPEED = 250  # mm/s
DT = 0.1  # s
TICK_LIMIT = 60 // DT  # s
WHEEL_BASE = 235  # mm


class Roomba(pufferlib.PufferEnv):
    def __init__(
        self, num_envs=1, render_mode=None, log_interval=128, buf=None, seed=0
    ):
        self.single_observation_space = gymnasium.spaces.Box(
            low=0, high=1, shape=(4,), dtype=np.float32
        )
        self.single_action_space = gymnasium.spaces.Box(
            low=-1, high=1, shape=(2,), dtype=np.float32
        )
        self.render_mode = render_mode
        self.num_agents = num_envs

        super().__init__(buf)
        self.c_envs = binding.vec_init(
            self.observations,
            self.actions,
            self.rewards,
            self.terminals,
            self.truncations,
            num_envs,
            seed,
            width=WIDTH,
            height=HEIGHT,
            speed=SPEED,
            dt=DT,
            tick_limit=TICK_LIMIT,
            wheel_base=WHEEL_BASE,
        )

    def reset(self, seed=0):
        binding.vec_reset(self.c_envs, seed)
        return self.observations, []

    def step(self, actions):
        self.actions[:] = actions
        binding.vec_step(self.c_envs)
        info = [binding.vec_log(self.c_envs)]
        return (self.observations, self.rewards, self.terminals, self.truncations, info)

    def render(self):
        binding.vec_render(self.c_envs, 0)

    def close(self):
        binding.vec_close(self.c_envs)
