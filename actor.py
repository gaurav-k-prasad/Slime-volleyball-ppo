import torch.nn as nn
import torch.nn.functional as F


class Actor(nn.Module):
    def __init__(self, in_state, out_actions):
        super().__init__()

        self.fc1 = nn.Linear(in_state, 128)
        self.fc2 = nn.Linear(128, 32)
        self.fc3 = nn.Linear(32, out_actions)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = F.softmax(self.fc3(x), dim=-1)
        return x
