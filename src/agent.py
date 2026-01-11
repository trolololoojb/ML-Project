import torch
import torch.nn as nn
import torch.nn.functional as F


class QNetwork(nn.Module):
    """Basic nature DQN agent."""

    def __init__(
        self,
        env,
        *,
        use_batch_norm: bool = False,
        bn_eps: float = 1e-5,
        bn_momentum: float = 0.1,
    ):
        super().__init__()
        obs_space = env.single_observation_space
        action_dim = int(env.single_action_space.n)
        self.is_image = len(obs_space.shape) == 3
        self.use_batch_norm = use_batch_norm

        if self.is_image:
            in_channels = obs_space.shape[0]
            self.conv1 = nn.Conv2d(in_channels, 32, 8, stride=4)
            self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
            self.conv3 = nn.Conv2d(64, 64, 3, stride=1)
            if use_batch_norm:
                self.bn1 = nn.BatchNorm2d(32, eps=bn_eps, momentum=bn_momentum)
                self.bn2 = nn.BatchNorm2d(64, eps=bn_eps, momentum=bn_momentum)
                self.bn3 = nn.BatchNorm2d(64, eps=bn_eps, momentum=bn_momentum)
            self.fc1 = nn.Linear(self._get_conv_out(obs_space.shape), 512)
            self.q = nn.Linear(512, action_dim)
        else:
            hidden_size = 128
            self.fc1 = nn.Linear(obs_space.shape[0], hidden_size)
            self.fc2 = nn.Linear(hidden_size, hidden_size)
            if use_batch_norm:
                self.bn1 = nn.BatchNorm1d(hidden_size, eps=bn_eps, momentum=bn_momentum)
                self.bn2 = nn.BatchNorm1d(hidden_size, eps=bn_eps, momentum=bn_momentum)
            self.q = nn.Linear(hidden_size, action_dim)

    def _get_conv_out(self, shape):
        with torch.no_grad():
            dummy = torch.zeros(1, *shape)
            if self.use_batch_norm:
                x = F.relu(self.bn1(self.conv1(dummy)))
                x = F.relu(self.bn2(self.conv2(x)))
                x = F.relu(self.bn3(self.conv3(x)))
            else:
                x = F.relu(self.conv1(dummy))
                x = F.relu(self.conv2(x))
                x = F.relu(self.conv3(x))
            return int(torch.flatten(x, start_dim=1).shape[1])

    def forward(self, x):
        if self.is_image:
            x = x.float() / 255.0
            if self.use_batch_norm:
                x = F.relu(self.bn1(self.conv1(x)))
                x = F.relu(self.bn2(self.conv2(x)))
                x = F.relu(self.bn3(self.conv3(x)))
            else:
                x = F.relu(self.conv1(x))
                x = F.relu(self.conv2(x))
                x = F.relu(self.conv3(x))
            x = torch.flatten(x, start_dim=1)
            x = F.relu(self.fc1(x))
            return self.q(x)
        x = x.float()
        if self.use_batch_norm:
            x = F.relu(self.bn1(self.fc1(x)))
            x = F.relu(self.bn2(self.fc2(x)))
        else:
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
        return self.q(x)


def linear_schedule(start_e: float, end_e: float, duration: float, t: int):
    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)
