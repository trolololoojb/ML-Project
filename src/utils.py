import math
import typing

import gymnasium as gym
import torch
import torch.nn as nn

from .wrappers import ClipRewardEnv, EpisodicLifeEnv, FireResetEnv, MaxAndSkipEnv, NoopResetEnv


def _is_minigrid(env_id: str) -> bool:
    return env_id.startswith("MiniGrid")


def _is_atari(env_id: str) -> bool:
    return "NoFrameskip" in env_id or env_id.startswith("ALE/")


def make_env(env_id, seed, idx, capture_video, run_name):
    """Helper function to create an environment with some standard wrappers.

    Taken from cleanRL's DQN Atari implementation: https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/dqn_atari.py.
    """

    def thunk():
        render_mode = "rgb_array" if capture_video and idx == 0 else None
        env = gym.make(env_id, render_mode=render_mode) if render_mode else gym.make(env_id)
        if capture_video and idx == 0:
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        env = gym.wrappers.RecordEpisodeStatistics(env)

        if _is_minigrid(env_id):
            from minigrid.wrappers import FlatObsWrapper

            env = FlatObsWrapper(env)
        elif _is_atari(env_id):
            env = NoopResetEnv(env, noop_max=30)
            env = MaxAndSkipEnv(env, skip=4)
            env = EpisodicLifeEnv(env)
            if "FIRE" in env.unwrapped.get_action_meanings():
                env = FireResetEnv(env)
            env = ClipRewardEnv(env)
            env = gym.wrappers.ResizeObservation(env, (84, 84))
            env = gym.wrappers.GrayScaleObservation(env)
            env = gym.wrappers.FrameStack(env, 4)

        env.action_space.seed(seed)

        return env

    return thunk


def set_cuda_configuration(gpu: typing.Any) -> torch.device:
    """Set up the device for the desired GPU or all GPUs."""

    if gpu is None or gpu == -1 or gpu is False:
        device = torch.device("cpu")
    elif isinstance(gpu, int):
        assert gpu <= torch.cuda.device_count(), "Invalid CUDA index specified."
        device = torch.device(f"cuda:{gpu}")
    else:
        device = torch.device("cuda")

    return device


@torch.no_grad()
def lecun_normal_initializer(layer: nn.Module) -> None:
    """
    Initialization according to LeCun et al. (1998).
    See here https://flax.readthedocs.io/en/latest/api_reference/flax.linen/_autosummary/flax.linen.initializers.lecun_normal.html
    and here https://github.com/google/jax/blob/366a16f8ba59fe1ab59acede7efd160174134e01/jax/_src/nn/initializers.py#L460 .
    Initializes bias terms to 0.
    """

    # Catch case where the whole network is passed
    if not isinstance(layer, nn.Linear | nn.Conv2d):
        return

    # For a conv layer, this is num_channels*kernel_height*kernel_width
    fan_in, _ = nn.init._calculate_fan_in_and_fan_out(layer.weight)

    # This implementation follows the jax one
    # https://github.com/google/jax/blob/366a16f8ba59fe1ab59acede7efd160174134e01/jax/_src/nn/initializers.py#L260
    variance = 1.0 / fan_in
    stddev = math.sqrt(variance) / 0.87962566103423978
    torch.nn.init.trunc_normal_(layer.weight)
    layer.weight *= stddev
    if layer.bias is not None:
        torch.nn.init.zeros_(layer.bias)
