import torch
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt


class Discriminator(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()

        self.hidden_dim = hidden_dim

        self.fc_1 = nn.Linear(768, hidden_dim)
        self.fc_2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_3 = nn.Linear(hidden_dim, 1)

        self.init_weights()

    def init_weights(self):
        nn.init.normal_(self.fc_1.weight, 0.0, 1 / sqrt(768))
        nn.init.constant_(self.fc_1.bias, 0.0)

        nn.init.normal_(self.fc_2.weight, 0.0, 1 / sqrt(self.hidden_dim))
        nn.init.constant_(self.fc_2.bias, 0.0)

        nn.init.normal_(self.fc_3.weight, 0.0, 1 / sqrt(self.hidden_dim))
        nn.init.constant_(self.fc_3.bias, 0.0)

    def forward(self, x):
        res = F.relu(self.fc_1(torch.Tensor(x)))
        res = F.relu(self.fc_2(res))
        res = self.fc_3(res)

        return torch.Tensor(torch.ravel(res))
