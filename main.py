import datetime
import random
import sys, pygame

import torch
from tqdm import tqdm
from pygame import time
from torch import nn

from game import Game


# TODO: this https://towardsdatascience.com/deep-q-network-dqn-ii-b6bf911b6b2c
def train(game):

    GAMMA = 0.99
    SYNCHRONIZE_INTERVAL = 10
    # Ensure all players have enough experience to start training.

    if sum(int(len(player.experience) == player.buffer_size)
           for player in game.players) != len(game.players):
        return

    for player in tqdm(game.players):
        # Fetch batch and extract values.
        batch = player.experience.sample(256)
        states, actions, rewards, dones, next_states = batch

        # Create tensors out of batch values
        states_v = torch.FloatTensor(states)
        next_states_v = torch.FloatTensor(next_states)
        actions_v = torch.tensor(actions)
        rewards_v = torch.tensor(rewards)
        done_mask = torch.ByteTensor(dones)

        # Forward pass through networks
        state_action_values = player.network(states_v).gather(1, actions_v.unsqueeze(-1)).squeeze(-1)
        next_state_values = player.target_network(next_states_v).max(1)[0]
        next_state_values[done_mask] = 0.0
        next_state_values = next_state_values.detach()

        # Compute loss
        expected_state_action_values = next_state_values * GAMMA + rewards_v
        loss_t = nn.MSELoss()(state_action_values, expected_state_action_values)

        # Apply loss
        player.optimizer.zero_grad()
        loss_t.backward()
        player.optimizer.step()

        player.update_ticks += 1
        # Update main network with target network
        if player.update_ticks % SYNCHRONIZE_INTERVAL == 0:
            print("Updating model")
            player.target_network.load_state_dict(player.network.state_dict())

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
        game.add_ai(buffer_size=5000)

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
            game.reset()

        game.render()
        pygame.display.flip()

        # dt = clock.tick(30)
        dt = 30


if __name__ == '__main__':
    main()
