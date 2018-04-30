from gym.envs.registration import registry, register, make, spec
from cvar.dqn.ice_lake.ple_env import DiscreteStateEnv


register(
    id='IceLake-v0',
    entry_point='cvar.dqn.ice_lake:DiscreteStateEnv',
    kwargs={'game_name': 'IceLake', 'display_screen': False},
    tags={'wrapper_config.TimeLimit.max_episode_steps': 500},
    nondeterministic=False,
)
