import math
from typing import List

import numpy as np
import pygame
from pygame.sprite import Sprite

from player import Player


class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y


def distance(rect1, rect2):
    return ((rect1.x - rect2.x) ** 2 + (rect1.y - rect2.y) ** 2) ** .5


class Game:
    width = 20
    height = 20
    n_next = 5

    def __init__(self, screen, size):
        # Rendering information
        self.size = size
        self.game_tick = 0
        self.screen = screen
        screen_width, screen_height = size
        self.render_width = screen_width / self.width
        self.render_height = screen_height / self.height

        self.board = np.zeros((self.width, self.height), int)
        self.last_few = []

        self.tiles: List[Sprite] = []
        self.players: List[Player] = []

        self.reset_board()
        # self.players.append(Player(self, screen, size))

    def reset_board(self):
        self.last_few = []
        self.tiles: List[Sprite] = []
        self.board[:, :] = 0
        # Add a box surrounding the map except for the bottom
        for i in range(0, self.height - 1):
            self.add_block(0, i)
            self.add_block(self.width - 1, i)

        for i in range(1, self.width - 1):
            self.add_block(i, 0)

        self.add_block(self.width // 2, self.height // 2 + 1)

    def _create_block(self):
        sprite = Sprite()
        sprite.image = pygame.image.load("sprites/block.png")
        sprite.image = pygame.transform.scale(sprite.image, (self.render_width, self.render_height))
        sprite.rect = pygame.Rect(0, 0, self.render_width, self.render_height)
        return sprite

    def can_place(self, x, y):
        # Within x range, y range, and board spot is empty
        return (0 <= x < self.width) and (0 <= y < self.height) and (self.board[x, y] == 0)

    def add_block(self, x, y):
        if not self.can_place(x, y):
            print(x, y, self.board)
            raise RuntimeError("Illegal block placement.")

        self.board[x, y] = self.n_next

        # Reduce value on last few placed tiles
        for entry in self.last_few:
            self.board[entry] -= 1

        # Add new tile to last few and cap length to N
        self.last_few.append((x, y))
        if len(self.last_few) == self.n_next:
            self.last_few.pop(0)

        block = self._create_block()
        block.rect.x = x * self.render_width
        block.rect.y = y * self.render_height
        self.tiles.append(block)

    def update(self, dt):
        self.game_tick += 1
        for player in self.players:
            player.update(dt)

            all_collided_id = player.sprite.rect.collidelistall([block.rect for block in self.tiles])
            if len(all_collided_id) == 0:
                continue

            all_collided_id = sorted(all_collided_id,
                                     key=lambda tid: distance(
                                         player.sprite.rect,
                                         self.tiles[tid].rect))
            for block_id in all_collided_id:
                block = self.tiles[block_id]
                if not player.sprite.rect.colliderect(block.rect):
                    # A collision can be resolved by a prior (closer) tile
                    continue

                rect = block.rect
                block_x = rect.x / self.render_width
                block_y = rect.y / self.render_height

                # If the last step the player was not in the block, stop it on that axis
                # TODO: Actually fix this 0.95 weird hack
                horizontal_stop = not (block_x - 0.95 < player.x - player.delta_x < block_x + 1)
                vertical_stop = not (block_y - 1 < player.y - player.delta_y < block_y + 1)

                # If both horizontal and vertical stops are triggered, we hit a corner, check which side of the
                # block we need to be pushed back to
                if vertical_stop and horizontal_stop:
                    block_enter_angle = math.atan2(-(player.y * self.render_height - rect.y),
                                                   player.x * self.render_height - rect.x)
                    movement_angle = math.atan2(-player.delta_y, player.delta_x)
                    block_enter_sharpness = abs(abs(block_enter_angle) - math.pi / 2)
                    movement_sharpness = abs(abs(movement_angle) - math.pi / 2)

                    vertical_stop = block_enter_sharpness < movement_sharpness
                    horizontal_stop = not vertical_stop

                down = player.delta_y >= 0
                right = player.delta_x >= 0

                if vertical_stop and down:
                    player.y = block_y - 1
                    player.delta_y = 0
                if vertical_stop and not down:
                    player.y = block_y + 1
                    player.delta_y = 0
                if horizontal_stop and right:
                    player.x = block_x - 1
                    player.delta_x = 0
                if horizontal_stop and not right:
                    player.x = block_x + 1
                    player.delta_x = 0
                player.update_rect()

    def simulation_ended(self):
        return sum(player.alive for player in self.players) == 0


    def render(self):
        for block in self.tiles:
            self.screen.blit(block.image, block.rect)
        for player in self.players:
            player.render()

    def add_ai(self):
        self.players.append(Player(self, self.screen, self.size, ai=True))

    def reset(self):
        self.game_tick = 0
        self.reset_board()

        for player in self.players:
            player.reset()

    def do_ai_player_action(self):
        ######################
        # Do NN forward pass
        ######################
        for player in self.players:
            if player.is_ai:
                player.do_ai_action()

    def do_ai_reward_players(self):
        for player in self.players:
            if player.is_ai:
                player.do_ai_reward()
