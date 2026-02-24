import functools
from pdb import set_trace as T

import gym
import numpy as np
import pufferlib.postprocess
import pufferlib.utils
import pufferlib.wrappers

import pufferlib
import pufferlib.emulation
import pufferlib.environments


def env_creator(name='zelda'):
    if name == 'zelda':
        name = 'gvgai-zelda-lvl0-v0'
    return functools.partial(make, name)


def make(
    name,
    obs_type='grayscale',
    frameskip=4,
    full_action_space=False,
    repeat_action_probability=0.0,
    render_mode='rgb_array',
    buf=None,
):
    """Atari creation function"""
    pufferlib.environments.try_import('gym_gvgai')
    env = gym.make(name)
    env = pufferlib.wrappers.GymToGymnasium(env)
    env = pufferlib.postprocess.EpisodeStats(env)
    env = pufferlib.emulation.GymnasiumPufferEnv(env=env, buf=buf)
    return env
