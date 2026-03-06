"""A minimal template for your own envs."""

import gymnasium
import numpy as np

import pufferlib
from pufferlib.ocean.boustrophedon import binding


class Boustrophedon(pufferlib.PufferEnv):
    def __init__(
        self, num_envs=1, render_mode=None, log_interval=128, buf=None, seed=0
    ):
        self.single_observation_space = gymnasium.spaces.Box(
            low=-1, high=1, shape=(4,), dtype=np.int8
        )
        self.single_action_space = gymnasium.spaces.Discrete(3)
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
