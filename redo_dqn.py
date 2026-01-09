# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/dqn/#dqn_ataripy
import json
import os
import random
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import tyro
import wandb
from gymnasium.vector import AutoresetMode

from src.agent import QNetwork, linear_schedule
from src.buffer import ReplayBuffer
from src.config import Config
from src.redo import run_redo
from src.utils import lecun_normal_initializer, make_env, set_cuda_configuration


def _sanitize_value(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return value.item()
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: _sanitize_value(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_sanitize_value(val) for val in value]
    return value


class TextLogger:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self.path.open("a", encoding="utf-8")

    def log(self, metrics: dict[str, Any], step: int) -> None:
        payload = {"step": int(step), "time": time.time()}
        payload.update({key: _sanitize_value(val) for key, val in metrics.items()})
        self._fh.write(json.dumps(payload, ensure_ascii=True) + "\n")
        self._fh.flush()

    def close(self) -> None:
        self._fh.close()


def _compute_grad_norm(parameters: list[torch.nn.Parameter]) -> float:
    grads = [p.grad.detach() for p in parameters if p.grad is not None]
    if not grads:
        return 0.0
    return torch.norm(torch.stack([g.norm(2) for g in grads]), 2).item()


def _collect_weight_norms(model: torch.nn.Module) -> dict[str, float]:
    norms: dict[str, float] = {}
    for name, module in model.named_modules():
        if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear)):
            norms[f"weights/{name}_norm"] = module.weight.data.norm(2).item()
    return norms


def _apply_env_preset(cfg: Config) -> None:
    if cfg.env_preset not in {"auto", "minigrid"}:
        return
    if not cfg.env_id.startswith("MiniGrid"):
        return
    defaults = Config()
    overrides = {
        "buffer_size": 100_000,
        "batch_size": 64,
        "learning_starts": 10_000,
        "target_network_frequency": 1_000,
        "exploration_fraction": 0.1,
        "end_e": 0.01,
        "learning_rate": 1e-4,
        "eval_interval": 25_000,
    }
    for key, value in overrides.items():
        if getattr(cfg, key) == getattr(defaults, key):
            setattr(cfg, key, value)


def _evaluate_policy(
    q_network: QNetwork,
    device: torch.device,
    env_id: str,
    eval_episodes: int,
    eval_seed: int,
    epsilon: float,
    run_name: str,
) -> list[float]:
    eval_envs = gym.vector.SyncVectorEnv(
        [make_env(env_id, eval_seed, 0, False, run_name)],
        autoreset_mode=AutoresetMode.SAME_STEP,
    )
    eval_rng = random.Random(eval_seed)
    obs, _ = eval_envs.reset(seed=eval_seed)
    episodic_returns: list[float] = []
    episode_returns = np.zeros(eval_envs.num_envs, dtype=np.float32)
    episode_lengths = np.zeros(eval_envs.num_envs, dtype=np.int32)
    eval_steps = 0
    last_print = 0

    was_training = q_network.training
    q_network.eval()

    with torch.no_grad():
        while len(episodic_returns) < eval_episodes:
            if eval_rng.random() < epsilon:
                actions = np.array([eval_envs.single_action_space.sample() for _ in range(eval_envs.num_envs)])
            else:
                q_values = q_network(torch.Tensor(obs).to(device))
                actions = torch.argmax(q_values, dim=1).cpu().numpy()
            next_obs, rewards, terminated, truncated, infos = eval_envs.step(actions)
            eval_steps += eval_envs.num_envs
            episode_returns += rewards
            episode_lengths += 1
            if eval_steps - last_print >= 1000:
                print(f"[eval] episodes={len(episodic_returns)}/{eval_episodes} steps={eval_steps}", flush=True)
                last_print = eval_steps
            dones = np.logical_or(terminated, truncated)
            if np.any(dones):
                for idx in np.where(dones)[0]:
                    episodic_returns.append(float(episode_returns[idx]))
                    episode_returns[idx] = 0.0
                    episode_lengths[idx] = 0
            obs = next_obs

    if was_training:
        q_network.train()

    eval_envs.close()
    return episodic_returns


def dqn_loss(
    q_network: QNetwork,
    target_network: QNetwork,
    obs: torch.Tensor,
    next_obs: torch.Tensor,
    actions: torch.Tensor,
    rewards: torch.Tensor,
    dones: torch.Tensor,
    gamma: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute the double DQN loss."""
    with torch.no_grad():
        # Get value estimates from the target network
        target_vals = target_network.forward(next_obs)
        # Select actions through the policy network
        policy_actions = q_network(next_obs).argmax(dim=1)
        target_max = target_vals[range(len(target_vals)), policy_actions]
        # Calculate Q-target
        td_target = rewards.flatten() + gamma * target_max * (1 - dones.flatten())

    old_val = q_network(obs).gather(1, actions).squeeze()
    return F.mse_loss(td_target, old_val), old_val


def main(cfg: Config) -> None:
    """Main training method for ReDO DQN."""
    _apply_env_preset(cfg)
    start_stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_name = f"{cfg.env_id}__{cfg.exp_name}__{cfg.seed}__{start_stamp}"
    print("[setup] initializing run", flush=True)

    # initialize wandb
    wandb.init(
        project=cfg.wandb_project_name,
        entity=cfg.wandb_entity,
        config=vars(cfg),
        name=run_name,
        monitor_gym=True,
        save_code=True,
        mode="online" if cfg.track else "disabled",
    )
    txt_logger = None
    if cfg.log_txt:
        run_dir = Path(cfg.log_txt_dir) / run_name
        txt_logger = TextLogger(run_dir / cfg.log_txt_filename)
        txt_logger.log({"event": "start", "env_id": cfg.env_id, "seed": cfg.seed}, step=0)

    if cfg.save_model:
        evaluation_episode = 0
        wandb.define_metric("evaluation_episode")
        wandb.define_metric("eval/episodic_return", step_metric="evaluation_episode")

    # To get deterministic pytorch to work
    if cfg.torch_deterministic:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.use_deterministic_algorithms(True)

    # TRY NOT TO MODIFY: seeding
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.set_float32_matmul_precision("high")

    device = set_cuda_configuration(cfg.gpu)

    # env setup
    print(f"[setup] creating envs for {cfg.env_id}", flush=True)
    envs = gym.vector.SyncVectorEnv(
        [make_env(cfg.env_id, cfg.seed + i, i, cfg.capture_video, run_name) for i in range(cfg.num_envs)],
        autoreset_mode=AutoresetMode.SAME_STEP,
    )
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

    print("[setup] initializing networks and replay buffer", flush=True)
    q_network = QNetwork(envs).to(device)
    if cfg.use_lecun_init:
        # Use the same initialization scheme as jax/flax
        q_network.apply(lecun_normal_initializer)
    optimizer = optim.Adam(q_network.parameters(), lr=cfg.learning_rate, eps=cfg.adam_eps)
    target_network = QNetwork(envs).to(device)
    target_network.load_state_dict(q_network.state_dict())

    rb = ReplayBuffer(
        cfg.buffer_size,
        envs.single_observation_space,
        envs.single_action_space,
        device,
        optimize_memory_usage=True,
        handle_timeout_termination=False,
    )
    start_time = time.time()

    # TRY NOT TO MODIFY: start the game
    print("[train] starting rollout loop", flush=True)
    obs, _ = envs.reset(seed=cfg.seed)
    for global_step in range(cfg.total_timesteps):
        # ALGO LOGIC: put action logic here
        epsilon = linear_schedule(cfg.start_e, cfg.end_e, cfg.exploration_fraction * cfg.total_timesteps, global_step)
        if random.random() < epsilon:
            actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
        else:
            q_values = q_network(torch.Tensor(obs).to(device))
            actions = torch.argmax(q_values, dim=1).cpu().numpy()

        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, rewards, terminated, truncated, infos = envs.step(actions)

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        if "final_info" in infos:
            final_infos = infos["final_info"]
            if isinstance(final_infos, dict):
                final_infos = [final_infos]
            for info in final_infos:
                if not isinstance(info, dict) or "episode" not in info:
                    continue
                epi_return = info["episode"]["r"].item()
                print(f"global_step={global_step}, episodic_return={epi_return}")
                if txt_logger:
                    txt_logger.log(
                        {
                            "train/episodic_return": epi_return,
                            "train/episodic_length": info["episode"]["l"].item(),
                            "train/epsilon": epsilon,
                        },
                        step=global_step,
                    )
                wandb.log(
                    {
                        "charts/episodic_return": epi_return,
                        "charts/episodic_length": info["episode"]["l"].item(),
                        "charts/epsilon": epsilon,
                    },
                    step=global_step,
                )

        # TRY NOT TO MODIFY: save data to reply buffer; handle `final_observation`
        real_next_obs = next_obs.copy()
        if "final_observation" in infos:
            for idx, d in enumerate(truncated):
                if d:
                    real_next_obs[idx] = infos["final_observation"][idx]
        rb.add(obs, real_next_obs, actions, rewards, terminated, infos)

        # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
        obs = next_obs

        # ALGO LOGIC: training.
        if global_step > cfg.learning_starts:
            done_update = False
            if done_update := global_step % cfg.train_frequency == 0:
                data = rb.sample(cfg.batch_size)
                loss, old_val = dqn_loss(
                    q_network=q_network,
                    target_network=target_network,
                    obs=data.observations,
                    next_obs=data.next_observations,
                    actions=data.actions,
                    rewards=data.rewards,
                    dones=data.dones,
                    gamma=cfg.gamma,
                )
                # optimize the model
                optimizer.zero_grad()
                loss.backward()
                grad_norm = _compute_grad_norm(list(q_network.parameters()))
                optimizer.step()

                q_mean = old_val.mean().item()
                q_max = old_val.max().item()
                logs = {
                    "losses/td_loss": loss.item(),
                    "losses/q_values": q_mean,
                    "diagnostics/q_mean": q_mean,
                    "diagnostics/q_max": q_max,
                    "diagnostics/epsilon": epsilon,
                    "diagnostics/grad_norm": grad_norm,
                    "charts/SPS": int(global_step / (time.time() - start_time)),
                }
                logs.update(_collect_weight_norms(q_network))

                if txt_logger:
                    txt_logger.log(logs, step=global_step)

            if global_step % cfg.redo_check_interval == 0:
                print(f"[redo] check at step={global_step}", flush=True)
                redo_samples = rb.sample(cfg.redo_bs)

                # --- wichtig für BatchNorm: Messung soll NICHT die running stats verändern
                was_training = q_network.training
                q_network.eval()

                redo_out = run_redo(
                    redo_samples.observations,
                    model=q_network,
                    optimizer=optimizer,
                    tau=cfg.redo_tau,
                    re_initialize=cfg.enable_redo,
                    use_lecun_init=cfg.use_lecun_init,
                )

                # wieder in den ursprünglichen Modus zurück
                if was_training:
                    q_network.train()

                # Nur bei ReDo wirklich Modell/Optimizer ersetzen
                if cfg.enable_redo:
                    q_network = redo_out["model"]
                    optimizer = redo_out["optimizer"]

                redo_metrics: dict[str, Any] = {
                    f"regularization/dormant_t={cfg.redo_tau}_fraction": redo_out["dormant_fraction"],
                    f"regularization/dormant_t={cfg.redo_tau}_count": redo_out["dormant_count"],
                    "regularization/dormant_t=0.0_fraction": redo_out["zero_fraction"],
                    "regularization/dormant_t=0.0_count": redo_out["zero_count"],
                }
                for name, value in redo_out["dormant_layer_fraction"].items():
                    redo_metrics[f"regularization/dormant_t={cfg.redo_tau}_layer/{name}_fraction"] = value
                for name, value in redo_out["dormant_layer_count"].items():
                    redo_metrics[f"regularization/dormant_t={cfg.redo_tau}_layer/{name}_count"] = value
                for name, value in redo_out["zero_layer_fraction"].items():
                    redo_metrics[f"regularization/dormant_t=0.0_layer/{name}_fraction"] = value
                for name, value in redo_out["zero_layer_count"].items():
                    redo_metrics[f"regularization/dormant_t=0.0_layer/{name}_count"] = value

                if txt_logger:
                    txt_logger.log(redo_metrics, step=global_step)
                wandb.log(redo_metrics, step=global_step)

            if cfg.eval_interval > 0 and global_step % cfg.eval_interval == 0:
                print(f"[eval] starting at step={global_step}", flush=True)
                eval_returns = _evaluate_policy(
                    q_network=q_network,
                    device=device,
                    env_id=cfg.env_id,
                    eval_episodes=cfg.eval_episodes,
                    eval_seed=cfg.eval_seed,
                    epsilon=cfg.eval_epsilon,
                    run_name=f"{run_name}-eval",
                )
                print(f"[eval] completed at step={global_step}", flush=True)
                eval_mean = float(np.mean(eval_returns)) if eval_returns else 0.0
                eval_std = float(np.std(eval_returns)) if eval_returns else 0.0
                eval_metrics = {
                    "eval/episodic_return_mean": eval_mean,
                    "eval/episodic_return_std": eval_std,
                    "eval/episodic_returns": eval_returns,
                }
                if txt_logger:
                    txt_logger.log(eval_metrics, step=global_step)
                wandb.log(eval_metrics, step=global_step)

            if global_step % 100 == 0 and done_update:
                print("SPS:", int(global_step / (time.time() - start_time)))
                wandb.log(logs, step=global_step)

            # update target network
            if global_step % cfg.target_network_frequency == 0:
                for target_network_param, q_network_param in zip(target_network.parameters(), q_network.parameters()):
                    target_network_param.data.copy_(
                        cfg.tau * q_network_param.data + (1.0 - cfg.tau) * target_network_param.data
                    )

    if cfg.save_model:
        model_path = Path(f"runs/{run_name}/{cfg.exp_name}")
        model_path.mkdir(parents=True, exist_ok=True)
        torch.save(q_network.state_dict(), model_path / ".cleanrl_model")
        print(f"model saved to {model_path}")
        from src.evaluate import evaluate

        episodic_returns = evaluate(
            model_path=model_path,
            make_env=make_env,
            env_id=cfg.env_id,
            eval_episodes=10,
            run_name=f"{run_name}-eval",
            Model=QNetwork,
            device=device,
            epsilon=0.05,
            capture_video=False,
        )
        for episodic_return in episodic_returns:
            wandb.log({"evaluation_episode": evaluation_episode, "eval/episodic_return": episodic_return})
            evaluation_episode += 1

    if txt_logger:
        txt_logger.log({"event": "end"}, step=cfg.total_timesteps)
        txt_logger.close()

    envs.close()
    wandb.finish()


if __name__ == "__main__":
    cfg = tyro.cli(Config)
    main(cfg)
