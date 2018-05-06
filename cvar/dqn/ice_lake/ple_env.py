import gym
from gym import spaces
from ple import PLE
import numpy as np


class LazyDrawPLE(PLE):

    def __init__(self, draw_function, args, **kwargs):
        super().__init__(args, **kwargs)
        self.draw_function = draw_function

    def getScreenRGB(self):
        self.draw_function()
        return super().getScreenRGB()


class Env(gym.Env):
    metadata = {'render.modes': ['human', 'rgb_array']}

    def __init__(self, game_name, display_screen=True):
        # open up a game state to communicate with emulator
        import importlib
        game_module_name = ('cvar.dqn.ice_lake.%s' % game_name).lower()
        game_module = importlib.import_module(game_module_name)
        game = getattr(game_module, game_name)()
        self.game_state = LazyDrawPLE(game.draw, game, fps=30, display_screen=display_screen,
                              state_preprocessor=state_preprocessor)
        self.game_state.init()
        self._action_set = sorted(self.game_state.getActionSet(), key=lambda x: (x is None, x))
        self.screen_width, self.screen_height = self.game_state.getScreenDims()

        self.action_space = None
        self.observation_space = None
        self.viewer = None

    def _step(self, a):
        raise NotImplementedError

    def _get_image(self):
        image_rotated = np.fliplr(np.rot90(self.game_state.getScreenRGB(),3)) # Hack to fix the rotated image returned by ple
        return image_rotated

    @property
    def _n_actions(self):
        return len(self._action_set)

    # return: (states, observations)
    def _reset(self):
        raise NotImplementedError

    def _render(self, mode='human', close=False):
        if close:
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None
            return
        img = self._get_image()
        if mode == 'rgb_array':
            return img
        elif mode == 'human':
            from gym.envs.classic_control import rendering
            if self.viewer is None:
                self.viewer = rendering.SimpleImageViewer()
            self.viewer.imshow(img)

    def _seed(self, seed=0):
        rng = np.random.seed(seed)
        self.game_state.rng = rng
        self.game_state.game.rng = self.game_state.rng

        self.game_state.init()


class DiscreteVisualEnv(Env):

    def __init__(self, game_name, display_screen=True):
        super().__init__(game_name, display_screen)

        self.action_space = spaces.Discrete(len(self._action_set))
        self.observation_space = spaces.Box(low=0, high=255, shape=(self.screen_width, self.screen_height, 3))

    def _step(self, a):
        reward = self.game_state.act(self._action_set[a])
        state = self._get_image()
        terminal = self.game_state.game_over()
        return state, reward, terminal, {}

    # return: (states, observations)
    def _reset(self):
        self.game_state.reset_game()
        state = self._get_image()
        return state


class DiscreteStateEnv(Env):

    def __init__(self, game_name, display_screen=True):
        super().__init__(game_name, display_screen)

        self.action_space = spaces.Discrete(len(self._action_set))
        self.observation_space = spaces.Box(low=-1e3, high=1e3, shape=self.game_state.getGameStateDims())

    def _step(self, a):
        reward = self.game_state.act(self._action_set[a])
        state = self.game_state.getGameState()
        terminal = self.game_state.game_over()
        return state, reward, terminal, {}

    def _reset(self):
        self.game_state.reset_game()
        return self.game_state.getGameState()


class DummyStateEnv(Env):

    def __init__(self, game_name, display_screen=True):
        super().__init__(game_name, display_screen)

        self.action_space = spaces.Discrete(len(self._action_set))
        self.observation_space = spaces.Box(low=0, high=1, shape=(100,))

    def _step(self, a):
        reward = self.game_state.act(self._action_set[a])
        state = self.game_state.getGameState()
        terminal = self.game_state.game_over()
        return state, reward, terminal, {}

    def _reset(self):
        self.game_state.reset_game()
        return self.game_state.getGameState()


def state_preprocessor(s):
    return s
