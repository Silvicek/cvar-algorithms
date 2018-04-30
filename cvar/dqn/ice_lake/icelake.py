import pygame
import sys
import numpy as np

from ple.games.base.pygamewrapper import PyGameWrapper

from pygame.constants import K_w, K_a, K_s, K_d
from ple.games.primitives import Player, Creep
from ple.games.utils import percent_round_int

BG_COLOR = (255, 255, 255)


# class GameObject:
#
#     def __init__(self, position, radius, color):


class IceLake(PyGameWrapper):
    """

    Parameters
    ----------
    width : int
        Screen width.

    height : int
        Screen height, recommended to be same dimension as width.

    """
    actions = {
        "up": K_w,
        "left": K_a,
        "right": K_d,
        "down": K_s
    }

    def __init__(self, width=256, height=256):

        PyGameWrapper.__init__(self, width, height, actions=self.actions)

        self.dx = 0
        self.dy = 0
        self.ticks = 0

        self._game_ended = False

        self.rewards = {
            "tick": -1./30,
            "loss": -100.0,
            "win": 100.0,
            "wall": -100./30
        }

    def _handle_player_events(self):
        self.dx = 0.0
        self.dy = 0.0

        agent_speed = 0.1 * self.width
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

            if event.type == pygame.KEYDOWN:
                key = event.key

                if key == self.actions["left"]:
                    self.dx -= agent_speed

                if key == self.actions["right"]:
                    self.dx += agent_speed

                if key == self.actions["up"]:
                    self.dy -= agent_speed

                if key == self.actions["down"]:
                    self.dy += agent_speed

    def getGameState(self):
        """
            Gets a non-visual state representation of the game.
            XXX: should be a dict, to np in preprocess
        """
        return np.array([self.player.pos.x, self.player.pos.y, self.player.vel.x, self.player.vel.y])

    def getGameStateDims(self):
        """
        Gets the games non-visual state dimensions.

        Returns
        -------
            list of tuples (min, max, discrete_steps) corresponding to each observation index
            TODO: constants
        """
        return np.array([(0, self.width, 50), (0, self.height, 50), (-200, 200, 2), (-200, 200, 2)], dtype=object)

    def getScore(self):
        return self.score

    def game_over(self):
        """
            Return bool if the game has 'finished'
        """
        return self._game_ended

    def init(self):
        """
            Starts/Resets the game to its initial state
        """
        target_radius = percent_round_int(self.width, 0.047)
        self.target = Creep(
            color=(40, 140, 40),
            radius=target_radius,
            pos_init=(self.width-target_radius, target_radius),
            dir_init=(1, 1),
            speed=0.0,
            reward=1.0,
            TYPE="-",
            SCREEN_WIDTH=self.width,
            SCREEN_HEIGHT=self.height,
            jitter_speed=0.0
        )

        self.ice = Creep(
            color=(0, 110, 255),
            radius=percent_round_int(self.width, 0.265),
            pos_init=(self.width/2, self.height/2),
            dir_init=(1, 1),
            speed=0.0,
            reward=1.0,
            TYPE="-",
            SCREEN_WIDTH=self.width,
            SCREEN_HEIGHT=self.height,
            jitter_speed=0.0
        )
        player_radius = percent_round_int(self.width, 0.047)
        self.player = Player(
            radius=player_radius,
            color=(60, 60, 140),
            speed=0.2 * self.width,
            pos_init=(1+player_radius, self.height-1-player_radius),
            SCREEN_WIDTH=self.width,
            SCREEN_HEIGHT=self.height)
        self.playerGroup = pygame.sprite.GroupSingle(self.player)

        self.creeps = pygame.sprite.Group()
        self.creeps.add(self.target)
        self.creeps.add(self.ice)

        self.score = 0
        self.ticks = 0
        self.lives = -1

        self._game_ended = False

    def step(self, dt):
        """
            Perform one step of game emulation.
        """
        dt /= 1000.0
        self.ticks += 1
        self.screen.fill(BG_COLOR)

        self.score += self.rewards["tick"]

        self._handle_player_events()
        self.player.update(self.dx, self.dy, dt)

        if self.wall_collide():
            self.score += self.rewards['wall']

        # if vec2d_distance(self.ice.pos, self.player.pos) < self.ice.radius:
        #     print(vec2d_distance(self.ice.pos, self.player.pos), self.ice.radius)
        #     if np.random.rand() < 0.01:
        #         self.game_ended = True
        #         self.score += self.rewards['loss']
        # elif vec2d_distance(self.target.pos, self.player.pos) < self.target.radius:
        #     self.game_ended = True
        #     self.score += self.rewards['win']
        if pygame.sprite.spritecollide(self.ice, self.playerGroup, False):  # TODO: investigate weird collisions
            if np.random.rand() < 0.1:
                self.game_ended = True
                self.score += self.rewards['loss']
        elif pygame.sprite.spritecollide(self.target, self.playerGroup, False):
            self.game_ended = True
            self.score += self.rewards['win']
        # print(vec2d_distance(self.target.pos, self.player.pos))
        # print((self.target.pos.x, self.target.pos.y), (self.player.pos.x, self.player.pos.y))

        self.player.draw(self.screen)
        self.creeps.draw(self.screen)

    def wall_collide(self):
        x = self.player.pos.x
        y = self.player.pos.y
        return x <= 0 or x >= self.width - self.player.radius * 2 or \
               y <= 0 or y >= self.height - self.player.radius * 2


def vec2d_distance(a, b):
    return np.sqrt((a.x-b.x)**2 + (a.y-b.y)**2)


if __name__ == "__main__":

    pygame.init()
    game = IceLake(width=256, height=256)
    game.screen = pygame.display.set_mode(game.getScreenDims(), 0, 32)
    game.clock = pygame.time.Clock()
    game.rng = np.random.RandomState(24)

    while True:
        game.init()
        while not game.game_over():
            dt = game.clock.tick_busy_loop(30)
            game.step(dt)
            pygame.display.update()
