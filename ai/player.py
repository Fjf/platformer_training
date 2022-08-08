import random
from collections import deque

import numpy as np
import pygame
import torch
from pygame.sprite import Sprite
from torch.optim import Adam

from ai.experience import ExperienceReplay, Experience
from ai.model import PlayerModel


class Player:
    friction = 0.3
    air_friction = 0.1
    gravity = 2

    def __init__(self, game, screen, size, ai=False, buffer_size=5000):
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
            self.play_ticks = 0
            self.update_ticks = 0

            self.network = PlayerModel()
            self.target_network = PlayerModel()
            self.optimizer = Adam(self.network.parameters(), lr=0.05)

            self.buffer_size = buffer_size
            self.experience = ExperienceReplay(self.buffer_size)
            self.current_action = None
            self.current_reward = None
            self.current_state = None

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
        if not (1 < self.x < w - 1) or not (1 < self.y < h - 1):
            self.alive = False

    def render(self):
        if not self.alive:
            return
        self.update_rect()
        self.screen.blit(self.sprite.image, self.sprite.rect)

    def do_ai_action(self):
        state = self.create_state()
        policy = self.network(torch.unsqueeze(torch.Tensor(state), dim=0))

        epsilon = 1 - min(self.game.game_tick * 0.01, 0.95)
        if random.random() < epsilon:
            action = random.randint(0, 3)
        else:
            _, act_v = torch.max(policy, dim=1)
            action = int(act_v.item())

        if action == 0:
            self.jump()
        elif action == 1:
            self.left()
        elif action == 2:
            self.right()
        else:
            ...  # noop

        self.current_state = np.copy(state)
        self.current_action = action
        self.play_ticks += 1

    def do_ai_reward(self):
        if self.alive:
            reward = self.play_ticks / 100
        else:
            reward = 0

        experience = Experience(
            state=self.current_state,
            action=self.current_action,
            done=not self.alive,
            reward=reward,
            new_state=self.create_state()
        )

        self.experience.append(experience)

    def create_state(self):
        interpolated_player = np.zeros(self.game.board.shape)

        if self.alive:
            x0 = self.x - int(self.x)
            x1 = 1 - x0
            y0 = self.y - int(self.y)
            y1 = 1 - y0
            interpolated_player[int(self.x) + 0, int(self.y) + 0] = x0 * y0
            interpolated_player[int(self.x) + 1, int(self.y) + 0] = x1 * y0
            interpolated_player[int(self.x) + 0, int(self.y) + 1] = x0 * y1
            interpolated_player[int(self.x) + 1, int(self.y) + 1] = x1 * y1

        return np.array([
            self.game.board,
            interpolated_player
        ], dtype=float)
