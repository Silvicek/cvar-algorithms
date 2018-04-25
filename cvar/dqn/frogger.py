from gym.envs.registration import register

register(
    id='FroggerNoFrameskip-v4',
    entry_point='gym.envs.atari:AtariEnv',
    kwargs={'game': 'frogger', 'obs_type': 'image', 'frameskip': 1},
    max_episode_steps=100000,
    nondeterministic=False,
)

