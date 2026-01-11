"""
Taken from cleanRL: https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl_utils/evals/dqn_eval.py.
"""

import random
from typing import Callable

import gymnasium as gym
import numpy as np
import torch
from gymnasium.vector import AutoresetMode

def evaluate(
    model_path: str,
    make_env: Callable,
    env_id: str,
    eval_episodes: int,
    run_name: str,
    Model: torch.nn.Module,
    model_kwargs: dict | None = None,
    device: torch.device = torch.device("cpu"),
    epsilon: float = 0.05,
    capture_video: bool = False,
):
    envs = gym.vector.SyncVectorEnv(
        [make_env(env_id, 0, 0, capture_video, run_name)],
        autoreset_mode=AutoresetMode.SAME_STEP,
    )
    model = Model(envs, **(model_kwargs or {})).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    obs, _ = envs.reset()
    episodic_returns = []
    episode_returns = np.zeros(envs.num_envs, dtype=np.float32)
    episode_lengths = np.zeros(envs.num_envs, dtype=np.int32)
    eval_steps = 0
    last_print = 0

    while len(episodic_returns) < eval_episodes:
        if random.random() < epsilon:
            actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
        else:
            q_values = model(torch.as_tensor(obs, device=device, dtype=torch.float32))
            actions = torch.argmax(q_values, dim=1).cpu().numpy()

        next_obs, rewards, terminated, truncated, infos = envs.step(actions)
        eval_steps += envs.num_envs
        episode_returns += rewards
        episode_lengths += 1

        # Fortschritt alle 1000 Schritte
        if eval_steps - last_print >= 1000:
            print(f"[eval] episodes={len(episodic_returns)}/{eval_episodes} steps={eval_steps}", flush=True)
            last_print = eval_steps

        dones = np.logical_or(terminated, truncated)
        if np.any(dones):
            for idx in np.where(dones)[0]:
                episodic_returns.append(float(episode_returns[idx]))
                print(
                    f"eval_episode={len(episodic_returns) - 1} return={episode_returns[idx]} length={episode_lengths[idx]}",
                    flush=True,
                )
                episode_returns[idx] = 0.0
                episode_lengths[idx] = 0

        obs = next_obs

    return episodic_returns
