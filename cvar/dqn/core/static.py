import os
import gym
import numpy as np
import tensorflow as tf


def parent_path(path):
    if path.endswith('/'):
        path = path[:-1]
    return os.path.join(*os.path.split(path)[:-1])


atari_actions = ['noop', 'fire', 'up', 'right', 'left',
                 'down', 'up-right', 'up-left', 'down-right', 'down-left',
                 'up-fire', 'right-fire', 'left-fire', 'down-fire', 'up-right-fire',
                 'up-left-fire', 'down-right-fire', 'down-left-fire']


def actions_from_env(env):
    """ Propagate through all wrappers to get action indices. """
    while True:
        if isinstance(env, gym.Wrapper):
            env = env.env
        else:
            break
    if isinstance(env, gym.Env):
        if hasattr(env, 'ale'):
            actions = env.ale.getMinimalActionSet()
            return [atari_actions[i] for i in actions]


def make_env_atari(game_name, random_action_eps=0.):
    from baselines.common.atari_wrappers import wrap_deepmind, make_atari
    env = make_atari(game_name + "NoFrameskip-v4")
    if random_action_eps > 0:
        env = ActionRandomizer(env, random_action_eps)
    monitored_env = SimpleMonitor(env)
    env = wrap_deepmind(monitored_env, frame_stack=True, scale=True)
    return env, monitored_env


def make_env_ice(game_name):
    from baselines.common.atari_wrappers import FrameStack, WarpFrame, MaxAndSkipEnv, ScaledFloatFrame
    import gym
    import cvar.dqn.ice_lake

    env = gym.make(game_name)
    env = MaxAndSkipEnv(env, skip=4)
    env = WarpFrame(env)
    env = ScaledFloatFrame(env)
    env = FrameStack(env, 4)
    return env


def make_session(num_cpu):
    tf_config = tf.ConfigProto(
        inter_op_parallelism_threads=num_cpu,
        intra_op_parallelism_threads=num_cpu)
    gpu_frac = 0.4
    tf_config.gpu_options.per_process_gpu_memory_fraction = gpu_frac
    import warnings
    warnings.warn("GPU is using a fixed fraction of memory: %.2f" % gpu_frac)

    return tf.Session(config=tf_config)


class ActionRandomizer(gym.ActionWrapper):

    def __init__(self, env, eps):
        super().__init__(env)
        self.eps = eps

    def _action(self, action):
        if np.random.random() < self.eps:
            # pick action with uniform probability
            return self.action_space.sample()
        else:
            return action

    def _reverse_action(self, action):
        pass


# hard copy from old baselines.common.misc_util
# TODO: remove?
import time
class SimpleMonitor(gym.Wrapper):
    def __init__(self, env):
        """Adds two qunatities to info returned by every step:
            num_steps: int
                Number of steps takes so far
            rewards: [float]
                All the cumulative rewards for the episodes completed so far.
        """
        super().__init__(env)
        # current episode state
        self._current_reward = None
        self._num_steps = None
        # temporary monitor state that we do not save
        self._time_offset = None
        self._total_steps = None
        # monitor state
        self._episode_rewards = []
        self._episode_lengths = []
        self._episode_end_times = []

    def _reset(self):
        obs = self.env.reset()
        # recompute temporary state if needed
        if self._time_offset is None:
            self._time_offset = time.time()
            if len(self._episode_end_times) > 0:
                self._time_offset -= self._episode_end_times[-1]
        if self._total_steps is None:
            self._total_steps = sum(self._episode_lengths)
        # update monitor state
        if self._current_reward is not None:
            self._episode_rewards.append(self._current_reward)
            self._episode_lengths.append(self._num_steps)
            self._episode_end_times.append(time.time() - self._time_offset)
        # reset episode state
        self._current_reward = 0
        self._num_steps = 0

        return obs

    def _step(self, action):
        obs, rew, done, info = self.env.step(action)
        self._current_reward += rew
        self._num_steps += 1
        self._total_steps += 1
        info['steps'] = self._total_steps
        info['rewards'] = self._episode_rewards
        return (obs, rew, done, info)

    def get_state(self):
        return {
            'env_id': self.env.unwrapped.spec.id,
            'episode_data': {
                'episode_rewards': self._episode_rewards,
                'episode_lengths': self._episode_lengths,
                'episode_end_times': self._episode_end_times,
                'initial_reset_time': 0,
            }
        }

    def set_state(self, state):
        assert state['env_id'] == self.env.unwrapped.spec.id
        ed = state['episode_data']
        self._episode_rewards = ed['episode_rewards']
        self._episode_lengths = ed['episode_lengths']
        self._episode_end_times = ed['episode_end_times']

