from pufferlib.environments.ocean import env_creator

import pufferlib.emulation

env = env_creator('spaces')()
env.reset()
env.step([1, 0])
breakpoint()
