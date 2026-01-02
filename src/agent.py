import torch
import torch.nn as nn
import torch.nn.functional as F


class QNetwork(nn.Module):
    """Basic nature DQN agent."""

    def __init__(self, env):
        super().__init__()
        obs_space = env.single_observation_space
        action_dim = int(env.single_action_space.n)
        self.is_image = len(obs_space.shape) == 3

        if self.is_image:
            in_channels = obs_space.shape[0]
            self.conv1 = nn.Conv2d(in_channels, 32, 8, stride=4)
            self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
            self.conv3 = nn.Conv2d(64, 64, 3, stride=1)
            self.fc1 = nn.Linear(self._get_conv_out(obs_space.shape), 512)
            self.q = nn.Linear(512, action_dim)
        else:
            hidden_size = 128
            self.fc1 = nn.Linear(obs_space.shape[0], hidden_size)
            self.fc2 = nn.Linear(hidden_size, hidden_size)
            self.q = nn.Linear(hidden_size, action_dim)

    def _get_conv_out(self, shape):
        with torch.no_grad():
            dummy = torch.zeros(1, *shape)
            x = F.relu(self.conv1(dummy))
            x = F.relu(self.conv2(x))
            x = F.relu(self.conv3(x))
            return int(torch.flatten(x, start_dim=1).shape[1])

    def forward(self, x):
        if self.is_image:
            x = x.float() / 255.0
            x = F.relu(self.conv1(x))
            x = F.relu(self.conv2(x))
            x = F.relu(self.conv3(x))
            x = torch.flatten(x, start_dim=1)
            x = F.relu(self.fc1(x))
            return self.q(x)
        x = x.float()
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.q(x)


def linear_schedule(start_e: float, end_e: float, duration: float, t: int):
    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)
