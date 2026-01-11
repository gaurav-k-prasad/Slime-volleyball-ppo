import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadActor(nn.Module):
    def __init__(self, in_state: int, move_size=3, jump_size=2):
        super().__init__()
        # Shared Body
        self.fc1 = nn.Linear(in_state, 128)
        self.fc2 = nn.Linear(128, 64)

        # Head 1: Movement (0=None, 1=Away, 2=Toward)
        self.move_head = nn.Linear(64, move_size)

        # Head 2: Jumping (0=No, 1=Yes)
        self.jump_head = nn.Linear(64, jump_size)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))

        # Both return a probability distribution over their respective categories
        move_probs = F.softmax(self.move_head(x), dim=-1)
        jump_probs = F.softmax(self.jump_head(x), dim=-1)

        return move_probs, jump_probs
