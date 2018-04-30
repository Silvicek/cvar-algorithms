import pygame
import sys
import numpy as np

from ple.games.base.pygamewrapper import PyGameWrapper

from pygame.constants import K_w, K_a, K_s, K_d
from ple.games.primitives import Player, Creep
from ple.games.utils.vec2d import vec2d
from ple.games.utils import percent_round_int


class PuckCreep(pygame.sprite.Sprite):

    def __init__(self, pos_init, attr, SCREEN_WIDTH, SCREEN_HEIGHT):
        pygame.sprite.Sprite.__init__(self)

        self.pos = vec2d(pos_init)
        self.attr = attr
        self.SCREEN_WIDTH = SCREEN_WIDTH
        self.SCREEN_HEIGHT = SCREEN_HEIGHT

        image = pygame.Surface(
            (self.attr["radius_outer"] * 2,
             self.attr["radius_outer"] * 2))
        image.fill((0, 0, 0, 0))
        image.set_colorkey((0, 0, 0))
        pygame.draw.circle(
            image,
            self.attr["color_outer"],
            (self.attr["radius_outer"], self.attr["radius_outer"]),
            self.attr["radius_outer"],
            0
        )

        image.set_alpha(int(255 * 0.75))

        pygame.draw.circle(
            image,
            self.attr["color_center"],
            (self.attr["radius_outer"], self.attr["radius_outer"]),
            self.attr["radius_center"],
            0
        )

        self.image = image
        self.rect = self.image.get_rect()
        self.rect.center = pos_init

    def update(self, ndx, ndy, dt):
        self.pos.x += ndx * self.attr['speed'] * dt
        self.pos.y += ndy * self.attr['speed'] * dt

        self.rect.center = (self.pos.x, self.pos.y)


class IceLake(PyGameWrapper):
    """

    Parameters
    ----------
    width : int
        Screen width.

    height : int
        Screen height, recommended to be same dimension as width.

    """

    def __init__(self, width=256, height=256):

        actions = {
            "up": K_w,
            "left": K_a,
            "right": K_d,
            "down": K_s
        }

        PyGameWrapper.__init__(self, width, height, actions=actions)

        self.CREEP_BAD = {
            "radius_center": percent_round_int(width, 0.047),
            "radius_outer": percent_round_int(width, 0.265),
            "color_center": (0, 110, 255),
            "color_outer": (0, 110, 255),
            "speed": 0.05 * width,
            "init_position": (self.width/2, self.height/2)
        }

        self.CREEP_GOOD = {
            "radius": percent_round_int(width, 0.047),
            "color": (40, 140, 40),
            "init_position": (self.width-10, 0 + 10)
        }

        self.AGENT_COLOR = (60, 60, 140)
        self.AGENT_SPEED = 0.2 * width
        self.AGENT_RADIUS = percent_round_int(width, 0.047)
        self.AGENT_INIT_POS = (0 + 5, 230)

        self.BG_COLOR = (255, 255, 255)
        self.dx = 0
        self.dy = 0
        self.ticks = 0

        self.game_ended = False

        self.rewards = {
            "tick": -1./30,
            "loss": -100.0,
            "win": 100.0,
            "wall": -100./30
        }

    def _handle_player_events(self):
        self.dx = 0.0
        self.dy = 0.0
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

            if event.type == pygame.KEYDOWN:
                key = event.key

                if key == self.actions["left"]:
                    self.dx -= self.AGENT_SPEED

                if key == self.actions["right"]:
                    self.dx += self.AGENT_SPEED

                if key == self.actions["up"]:
                    self.dy -= self.AGENT_SPEED

                if key == self.actions["down"]:
                    self.dy += self.AGENT_SPEED

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
        return self.game_ended

    def init(self):
        """
            Starts/Resets the game to its initial state
        """

        self.player = Player(
            self.AGENT_RADIUS,
            self.AGENT_COLOR,
            self.AGENT_SPEED,
            self.AGENT_INIT_POS,
            self.width,
            self.height)

        self.playerGroup = pygame.sprite.GroupSingle(self.player)

        self.target = Creep(
            self.CREEP_GOOD['color'],
            self.CREEP_GOOD['radius'],
            self.CREEP_GOOD['init_position'],
            (1, 1),
            0.0,
            1.0,
            "GOOD",
            self.width,
            self.height,
            0.0  # jitter
        )

        self.ice = PuckCreep(
            self.CREEP_BAD['init_position'],
            self.CREEP_BAD,
            self.screen_dim[0] * 0.75,
            self.screen_dim[1] * 0.75)

        self.creeps = pygame.sprite.Group()
        self.creeps.add(self.target)
        self.creeps.add(self.ice)

        self.score = 0
        self.ticks = 0
        self.lives = -1

        self.game_ended = False

    def step(self, dt):
        """
            Perform one step of game emulation.
        """
        dt /= 1000.0
        self.ticks += 1
        self.screen.fill(self.BG_COLOR)

        self.score += self.rewards["tick"]

        self._handle_player_events()
        self.player.update(self.dx, self.dy, dt)

        if self.wall_collide():
            self.score += self.rewards['wall']

        if pygame.sprite.spritecollide(self.ice, self.playerGroup, False):  # TODO: investigate weird collisions
            if np.random.rand() < 0.1:
                self.game_ended = True
                self.score += self.rewards['loss']
        elif pygame.sprite.spritecollide(self.target, self.playerGroup, False):
            self.game_ended = True
            self.score += self.rewards['win']

        self.player.draw(self.screen)
        self.creeps.draw(self.screen)

    def wall_collide(self):
        x = self.player.pos.x
        y = self.player.pos.y
        return x <= 0 or x >= self.width - self.AGENT_RADIUS * 2 or \
               y <= 0 or y >= self.height - self.AGENT_RADIUS * 2


if __name__ == "__main__":

    pygame.init()
    game = IceLake(width=256, height=256)
    game.screen = pygame.display.set_mode(game.getScreenDims(), 0, 32)
    game.clock = pygame.time.Clock()
    game.rng = np.random.RandomState(24)
    game.init()

    while True:
        dt = game.clock.tick_busy_loop(60)
        game.step(dt)
        pygame.display.update()
