from gym.envs.registration import registry, register, make, spec
from cvar.dqn.ice_lake.ple_env import DiscreteStateEnv, DiscreteVisualEnv, DummyStateEnv
# headless
import os
os.putenv('SDL_VIDEODRIVER', 'fbcon')
os.environ["SDL_VIDEODRIVER"] = "dummy"

register(
    id='IceLake-v0',
    entry_point='cvar.dqn.ice_lake:DiscreteStateEnv',
    kwargs={'game_name': 'IceLake', 'display_screen': False},
    tags={'wrapper_config.TimeLimit.max_episode_steps': 500},
    nondeterministic=False,
)

register(
    id='IceLakeRGB-v0',
    entry_point='cvar.dqn.ice_lake:DiscreteVisualEnv',
    kwargs={'game_name': 'IceLake', 'display_screen': False},
    tags={'wrapper_config.TimeLimit.max_episode_steps': 500},
    nondeterministic=False,
)


register(
    id='DummyIceLake-v0',
    entry_point='cvar.dqn.ice_lake:DummyStateEnv',
    kwargs={'game_name': 'DummyIceLake', 'display_screen': False},
    tags={'wrapper_config.TimeLimit.max_episode_steps': 500},
    nondeterministic=False,
)