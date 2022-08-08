import torch
from torch import nn
from torch.nn.modules.module import T


class PlayerModel(nn.Module):
    def __init__(self, input_width=100, input_height=100, action_space=4):
        """
        Action space is 4, because doing nothing is also an option
        :param input_width:
        :param input_height:
        :param action_space:
        """
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(2, 2, (3, 3)),
            nn.Conv2d(2, 2, (3, 3)),
            nn.Conv2d(2, 2, (3, 3)),
            nn.Conv2d(2, 1, (3, 3)),

            nn.Flatten(),
            nn.Linear(12 * 12, action_space)
        )
        self.activation = nn.Softmax()

    def forward(self, batch):
        embedding = self.net(batch)
        activation = self.activation(embedding)
        return activation
