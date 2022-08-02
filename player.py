import random

import numpy as np
import pygame
import torch
from pygame.sprite import Sprite
from torch.optim import Adam

from model import PlayerModel


class Player:
    friction = 0.3
    air_friction = 0.1
    gravity = 2

    def __init__(self, game, screen, size, ai=False):
        self.game = game
        self.x = self.game.width // 2
        self.y = self.game.height // 2
        self.delta_x = 0
        self.delta_y = 0
        self.alive = True

        # Rendering information
        self.screen = screen
        screen_width, screen_height = size
        self.render_width = screen_width / self.game.width
        self.render_height = screen_height / self.game.height

        self.sprite = Sprite()
        self.sprite.image = pygame.image.load("sprites/blob.png")
        self.sprite.image = pygame.transform.scale(self.sprite.image, (self.render_width, self.render_height))
        self.sprite.rect = pygame.Rect(self.x * self.render_width, self.y * self.render_height, self.render_width,
                                       self.render_height)

        self.network = None
        self.is_ai = ai

        if self.is_ai:
            #################################
            # Initialize pytorch stuff
            #################################
            self.network = PlayerModel()
            self.optimizer = Adam(self.network.parameters(), lr=0.05)

            self.action_history = []
            self.reward_history = []
            self.state_history = []

    def reset(self):
        self.x = self.game.width // 2
        self.y = self.game.height // 2
        self.delta_x = 0
        self.delta_y = 0
        self.alive = True

    def jump(self):
        if not self.alive:
            return

        if self.y - int(self.y) > 0.001:
            return
        if np.any(self.game.board[int(self.x) - 1:int(self.x) + 2, int(self.y + 1)]):
            self.delta_y = -1.2

    def left(self):
        self.delta_x -= .2

    def right(self):
        self.delta_x += .2

    def update(self, dt):
        if not self.alive:
            return
        dt /= 1000

        self.delta_y += self.gravity * dt

        self.delta_x *= (1 - self.friction)
        self.delta_y *= (1 - self.air_friction)
        if abs(self.delta_x) < 0.0000001:
            self.delta_x = 0
        if abs(self.delta_y) < 0.0000001:
            self.delta_y = 0

        self.x += self.delta_x
        self.y += self.delta_y
        self.update_rect()

    def update_rect(self):
        self.sprite.rect.x = self.x * self.render_width
        self.sprite.rect.y = self.y * self.render_height

        w, h = self.game.board.shape
        if not (0 < self.x < w) or not (0 < self.y < h):
            self.alive = False

    def render(self):
        if not self.alive:
            return
        self.update_rect()
        self.screen.blit(self.sprite.image, self.sprite.rect)

    def do_ai_action(self):
        tensor = torch.Tensor(self.game.board)
        policy = self.network(torch.unsqueeze(tensor, dim=0))

        if random.random() < 1 - min(self.game.game_tick * 0.01, 0.95):
            action = random.randint(0, 2)
        else:
            action = torch.multinomial(policy, 1).item()

        if action == 0:
            self.jump()
        elif action == 1:
            self.left()
        else:
            self.right()

        self.state_history.append(np.copy(self.game.board))
        self.action_history.append(action)

    def do_ai_reward(self):
        if self.alive:
            self.reward_history.append(1)
        else:
            self.reward_history.append(0)
