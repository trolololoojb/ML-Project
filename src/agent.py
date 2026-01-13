import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-8, affine: bool = True) -> None:
        super().__init__()
        self.eps = eps
        self.affine = affine
        if affine:
            self.weight = nn.Parameter(torch.ones(dim))
        else:
            self.register_parameter("weight", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).sqrt()
        x = x / rms
        if self.affine:
            x = x * self.weight
        return x


class RMSNorm2d(nn.Module):
    def __init__(self, channels: int, eps: float = 1e-8, affine: bool = True) -> None:
        super().__init__()
        self.eps = eps
        self.affine = affine
        if affine:
            self.weight = nn.Parameter(torch.ones(1, channels, 1, 1))
        else:
            self.register_parameter("weight", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.pow(2).mean(dim=(1, 2, 3), keepdim=True).add(self.eps).sqrt()
        x = x / rms
        if self.affine:
            x = x * self.weight
        return x


class QNetwork(nn.Module):
    """Basic nature DQN agent."""

    def __init__(
        self,
        env,
        *,
        use_batch_norm: bool = False,
        bn_eps: float = 1e-5,
        bn_momentum: float = 0.1,
        enable_nap: bool = False,
        nap_norm_type: str = "layer",
        nap_eps: float = 1e-8,
        nap_affine: bool = True,
        nap_remove_bias: bool = False,
    ):
        super().__init__()
        obs_space = env.single_observation_space
        action_dim = int(env.single_action_space.n)
        self.is_image = len(obs_space.shape) == 3
        self.use_batch_norm = use_batch_norm
        self.enable_nap = enable_nap

        if enable_nap and use_batch_norm:
            raise ValueError("enable_nap and use_batch_norm cannot both be True in the same network.")
        if nap_norm_type not in {"rms", "layer"}:
            raise ValueError(f"Unsupported nap_norm_type='{nap_norm_type}'. Use 'rms' or 'layer'.")

        bias = not (enable_nap and nap_remove_bias)

        if self.is_image:
            in_channels = obs_space.shape[0]
            self.conv1 = nn.Conv2d(in_channels, 32, 8, stride=4, bias=bias)
            self.conv2 = nn.Conv2d(32, 64, 4, stride=2, bias=bias)
            self.conv3 = nn.Conv2d(64, 64, 3, stride=1, bias=bias)
            if use_batch_norm:
                self.bn1 = nn.BatchNorm2d(32, eps=bn_eps, momentum=bn_momentum)
                self.bn2 = nn.BatchNorm2d(64, eps=bn_eps, momentum=bn_momentum)
                self.bn3 = nn.BatchNorm2d(64, eps=bn_eps, momentum=bn_momentum)
            if enable_nap:
                if nap_norm_type == "layer":
                    self.n1 = nn.GroupNorm(1, 32, eps=nap_eps, affine=nap_affine)
                    self.n2 = nn.GroupNorm(1, 64, eps=nap_eps, affine=nap_affine)
                    self.n3 = nn.GroupNorm(1, 64, eps=nap_eps, affine=nap_affine)
                else:
                    self.n1 = RMSNorm2d(32, eps=nap_eps, affine=nap_affine)
                    self.n2 = RMSNorm2d(64, eps=nap_eps, affine=nap_affine)
                    self.n3 = RMSNorm2d(64, eps=nap_eps, affine=nap_affine)
            self.fc1 = nn.Linear(self._get_conv_out(obs_space.shape), 512, bias=bias)
            if enable_nap:
                if nap_norm_type == "layer":
                    self.n4 = nn.LayerNorm(512, eps=nap_eps, elementwise_affine=nap_affine)
                else:
                    self.n4 = RMSNorm(512, eps=nap_eps, affine=nap_affine)
            self.q = nn.Linear(512, action_dim, bias=True)
        else:
            hidden_size = 128
            self.fc1 = nn.Linear(obs_space.shape[0], hidden_size, bias=bias)
            self.fc2 = nn.Linear(hidden_size, hidden_size, bias=bias)
            if use_batch_norm:
                self.bn1 = nn.BatchNorm1d(hidden_size, eps=bn_eps, momentum=bn_momentum)
                self.bn2 = nn.BatchNorm1d(hidden_size, eps=bn_eps, momentum=bn_momentum)
            if enable_nap:
                if nap_norm_type == "layer":
                    self.n1 = nn.LayerNorm(hidden_size, eps=nap_eps, elementwise_affine=nap_affine)
                    self.n2 = nn.LayerNorm(hidden_size, eps=nap_eps, elementwise_affine=nap_affine)
                else:
                    self.n1 = RMSNorm(hidden_size, eps=nap_eps, affine=nap_affine)
                    self.n2 = RMSNorm(hidden_size, eps=nap_eps, affine=nap_affine)
            self.q = nn.Linear(hidden_size, action_dim, bias=True)

    def _get_conv_out(self, shape):
        with torch.no_grad():
            dummy = torch.zeros(1, *shape)
            if self.enable_nap:
                x = self.conv1(dummy)
                x = self.n1(x)
                x = F.relu(x)
                x = self.conv2(x)
                x = self.n2(x)
                x = F.relu(x)
                x = self.conv3(x)
                x = self.n3(x)
                x = F.relu(x)
            elif self.use_batch_norm:
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
            if self.enable_nap:
                x = self.conv1(x)
                x = self.n1(x)
                x = F.relu(x)
                x = self.conv2(x)
                x = self.n2(x)
                x = F.relu(x)
                x = self.conv3(x)
                x = self.n3(x)
                x = F.relu(x)
            elif self.use_batch_norm:
                x = F.relu(self.bn1(self.conv1(x)))
                x = F.relu(self.bn2(self.conv2(x)))
                x = F.relu(self.bn3(self.conv3(x)))
            else:
                x = F.relu(self.conv1(x))
                x = F.relu(self.conv2(x))
                x = F.relu(self.conv3(x))
            x = torch.flatten(x, start_dim=1)
            if self.enable_nap:
                x = self.fc1(x)
                x = self.n4(x)
                x = F.relu(x)
            else:
                x = F.relu(self.fc1(x))
            return self.q(x)
        x = x.float()
        if self.enable_nap:
            x = self.fc1(x)
            x = self.n1(x)
            x = F.relu(x)
            x = self.fc2(x)
            x = self.n2(x)
            x = F.relu(x)
        elif self.use_batch_norm:
            x = F.relu(self.bn1(self.fc1(x)))
            x = F.relu(self.bn2(self.fc2(x)))
        else:
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
        return self.q(x)


def linear_schedule(start_e: float, end_e: float, duration: float, t: int):
    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)
