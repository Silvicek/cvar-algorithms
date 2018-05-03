import pygame
import sys
import numpy as np

from ple.games.base.pygamewrapper import PyGameWrapper

from pygame.constants import K_w, K_a, K_s, K_d
from ple.games.utils import percent_round_int

BG_COLOR = (255, 255, 255)


class GameObject(pygame.sprite.Sprite):

    def __init__(self, position, radius, color):
        super().__init__()
        self.position = position
        self.velocity = np.zeros(position.shape)
        self.radius = radius
        self.color = color

        image = pygame.Surface([radius * 2, radius * 2])
        image.set_colorkey((0, 0, 0))

        pygame.draw.circle(
            image,
            color,
            (radius, radius),
            radius,
            0
        )

        self.image = image.convert()
        self.rect = self.image.get_rect()

    def draw(self, screen):
        self.rect = pygame.Rect(self.position[0]-self.radius, self.position[1]-self.radius, self.radius, self.radius)
        screen.blit(self.image, self.rect)

    def update(self, velocity, dt):
        self.velocity += velocity

        self.position = self.position + self.velocity * dt

        self.velocity *= 0.975

    @staticmethod
    def distance(a, b):
        return np.sqrt(np.sum(np.square(a.position - b.position)))


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

    rewards = {
        "tick": -10. / 30,
        "ice": -50.0,
        "win": 100.0,
        "wall": 0.,
    }

    def __init__(self, width=84, height=84):

        PyGameWrapper.__init__(self, width, height, actions=self.actions)

        self.dx = 0.
        self.dy = 0.
        self.ticks = 0

        self._game_ended = False

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
        return np.hstack((self.player.position, self.player.velocity))

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
        return self._score

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
        self.target = GameObject(np.array([self.width-target_radius, self.height-target_radius]),
                                 target_radius, (40, 140, 40))

        ice_radius = percent_round_int(self.width, 0.3)
        self.ice = GameObject(np.array([self.width/2, self.height-ice_radius]),
                              ice_radius, (0, 110, 255))

        player_radius = percent_round_int(self.width, 0.047)
        player_position = np.random.rand(2) * np.array([self.width, self.height])
        self.player = GameObject(player_position,
                                 player_radius, (1, 1, 1))
        self.playerGroup = pygame.sprite.GroupSingle(self.player)

        self.creeps = pygame.sprite.Group()
        self.creeps.add(self.target)
        self.creeps.add(self.ice)

        self._score = 0.
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

        self._score += IceLake.rewards["tick"]

        self._handle_player_events()
        self.player.update(np.array([self.dx, self.dy]), dt)

        if GameObject.distance(self.target, self.player) < self.target.radius:
            self._game_ended = True
            self._score += IceLake.rewards['win']
        elif self.wall_collide():
            self._game_ended = False
            self._score += IceLake.rewards['wall']
        elif GameObject.distance(self.ice, self.player) < self.ice.radius:
            if np.random.random() < 0.01:
                self._game_ended = True
                self._score += IceLake.rewards['ice']

    def draw(self):
        self.target.draw(self.screen)
        self.ice.draw(self.screen)
        self.player.draw(self.screen)

    def wall_collide(self):
        x = self.player.position[0]
        y = self.player.position[1]
        r = self.player.radius
        collision = False
        if x <= r:
            self.player.position[0] = r
            self.player.velocity[0] = 0
            collision = True
        elif x >= self.width - r:
            self.player.position[0] = self.width - r
            self.player.velocity[0] = 0
            collision = True
        if y <= r:
            self.player.position[1] = r
            self.player.velocity[1] = 0
            collision = True
        elif y >= self.height - r:
            self.player.position[1] = self.height - r
            self.player.velocity[1] = 0
            collision = True

        return collision


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
            game.draw()
            pygame.display.update()
        print("Episode reward", game.getScore())


