import torch
from torch import nn
from torch.nn.modules.module import T


class PlayerModel(nn.Module):
    def __init__(self, input_width=100, input_height=100):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(1, 1, (3, 3)),
            nn.Conv2d(1, 1, (3, 3)),
            nn.Conv2d(1, 1, (3, 3)),
            nn.Conv2d(1, 1, (3, 3)),

            nn.Flatten(),
            nn.Linear(12 * 12, 3)
        )
        self.activation = nn.Softmax()

    def forward(self, batch):
        embedding = self.net(batch)
        activation = self.activation(embedding)
        return activation
