import datetime
import random
import sys, pygame

import torch
from pygame import time
from torch import nn

from game import Game


# TODO: this https://towardsdatascience.com/deep-q-network-dqn-ii-b6bf911b6b2c
def train(game):
    for player in game.players:
        for state, action, reward in zip(player.state_history, player.action_history, player.reward_history):
            policy = player.network(state)
            action = torch.multinomial(policy, 1).item()

            criterion = nn.SmoothL1Loss()
            loss = criterion(action, expected_state_action_values.unsqueeze(1))


def main():
    pygame.init()

    size = 900, 900

    background = (30, 150, 180)
    screen = pygame.display.set_mode(size)
    clock = pygame.time.Clock()
    pygame.key.set_repeat(0, 10)
    game = Game(screen, size)
    game.add_block(11, 10)

    # Add 10 AI players to game
    for i in range(10):
        game.add_ai()

    dt = 0
    while 1:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                sys.exit()
        keys = pygame.key.get_pressed()
        if keys[pygame.K_SPACE]:
            game.players[0].jump()
        if keys[pygame.K_LEFT]:
            game.players[0].left()
        if keys[pygame.K_RIGHT]:
            game.players[0].right()

        screen.fill(background)

        game.do_ai_player_action()
        game.update(dt)
        game.do_ai_reward_players()

        if game.simulation_ended():
            train(game)

        game.render()
        pygame.display.flip()

        # dt = clock.tick(30)
        dt = 30


if __name__ == '__main__':
    main()
