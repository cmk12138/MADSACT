# -*- coding: utf-8 -*-
"""
Main entry point for training and evaluating a MAPPO agent in the
multi-UAV path planning environment. The file mirrors the structure of
``main_DDPG.py`` but swaps in an on-policy multi-agent PPO pipeline so
that downstream tasks (hyper-parameter tuning, agent wiring, logging,
etc.) can be implemented incrementally.
"""
from __future__ import annotations

import argparse
import os
import pickle as pkl
from dataclasses import asdict, dataclass
from typing import Dict, Iterator, Tuple

import numpy as np
#import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

from rl_env.path_env import RlGame


PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DEFAULT_RESULT_DIR = os.path.join(PROJECT_ROOT, "result", "mappo")
DEFAULT_CHECKPOINT = os.path.join(DEFAULT_RESULT_DIR, "mappo_agent.pth")
DEFAULT_STATS = os.path.join(DEFAULT_RESULT_DIR, "mappo_train_stats.pkl")


@dataclass
class MAPPOConfig:
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_coef: float = 0.2
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    rollout_length: int = 256
    ppo_epochs: int = 4
    num_minibatches: int = 4
    policy_lr: float = 3e-4
    value_lr: float = 1e-3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MAPPO trainer for multi-UAV path planning.")
    parser.add_argument("--train-iters", type=int, default=200, help="Number of policy updates.")
    parser.add_argument("--rollout-length", type=int, default=256, help="Timesteps collected per update.")
    parser.add_argument("--render", action="store_true", help="Render the environment during rollout/eval.")
    parser.add_argument("--eval", action="store_true", help="Skip training and run evaluation only.")
    parser.add_argument("--eval-episodes", type=int, default=10, help="Evaluation episodes to run when --eval is set.")
    parser.add_argument("--save-interval", type=int, default=25, help="How often (in updates) to checkpoint.")
    parser.add_argument("--log-dir", type=str, default=DEFAULT_RESULT_DIR, help="Directory used for checkpoints/stats.")
    parser.add_argument("--model-path", type=str, default=DEFAULT_CHECKPOINT, help="Target path for checkpoints.")
    parser.add_argument("--stats-path", type=str, default=DEFAULT_STATS, help="Pickle output containing training curves.")
    parser.add_argument("--resume", type=str, default="", help="Optional checkpoint to resume training from.")
    parser.add_argument("--num-allies", type=int, default=1, help="Number of friendly UAVs.")
    parser.add_argument("--num-enemies", type=int, default=1, help="Number of opponent UAVs.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="torch device to place the networks on.",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def mlp(input_dim: int, output_dim: int, hidden_sizes: Tuple[int, int] = (128, 128)) -> nn.Sequential:
    layers = []
    prev_dim = input_dim
    for hidden in hidden_sizes:
        layers.append(nn.Linear(prev_dim, hidden))
        layers.append(nn.ReLU())
        prev_dim = hidden
    layers.append(nn.Linear(prev_dim, output_dim))
    return nn.Sequential(*layers)


class MAPPOActorCritic(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        global_obs_dim: int,
        action_dim: int,
        action_high: float,
        hidden_sizes: Tuple[int, int] = (128, 128),
    ) -> None:
        super().__init__()
        self.policy_net = mlp(obs_dim, action_dim, hidden_sizes)
        self.value_net = mlp(global_obs_dim, 1, hidden_sizes)
        self.log_std = nn.Parameter(torch.zeros(action_dim))
        self.register_buffer("action_scale", torch.tensor(action_high, dtype=torch.float32))

    def policy_parameters(self):
        return list(self.policy_net.parameters()) + [self.log_std]

    def value_parameters(self):
        return self.value_net.parameters()

    def _distribution(self, obs: torch.Tensor) -> Normal:
        mean = self.policy_net(obs)
        std = torch.exp(self.log_std).expand_as(mean)
        return Normal(mean, std)

    def act(
        self,
        local_obs: torch.Tensor,
        global_obs: torch.Tensor,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        dist = self._distribution(local_obs)
        if deterministic:
            raw_action = dist.mean
            log_prob = torch.zeros(raw_action.size(0), 1, device=raw_action.device)
        else:
            raw_action = dist.rsample()
            log_prob = dist.log_prob(raw_action).sum(dim=-1, keepdim=True)
        scaled_action = torch.tanh(raw_action) * self.action_scale
        values = self.value_net(global_obs)
        return scaled_action, raw_action, log_prob, values

    def evaluate_actions(
        self,
        local_obs: torch.Tensor,
        actions: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        dist = self._distribution(local_obs)
        log_prob = dist.log_prob(actions).sum(dim=-1, keepdim=True)
        entropy = dist.entropy().sum(dim=-1).mean()
        return log_prob, entropy

    def evaluate_values(self, global_obs: torch.Tensor) -> torch.Tensor:
        return self.value_net(global_obs)


class RolloutBuffer:
    def __init__(
        self,
        rollout_length: int,
        num_agents: int,
        obs_dim: int,
        global_obs_dim: int,
        action_dim: int,
    ) -> None:
        self.rollout_length = rollout_length
        self.num_agents = num_agents
        self.obs_dim = obs_dim
        self.global_obs_dim = global_obs_dim
        self.action_dim = action_dim
        self.reset()

    def reset(self) -> None:
        shape_time = (self.rollout_length, self.num_agents)
        self.local_obs = np.zeros(shape_time + (self.obs_dim,), dtype=np.float32)
        self.global_obs = np.zeros(shape_time + (self.global_obs_dim,), dtype=np.float32)
        self.actions = np.zeros(shape_time + (self.action_dim,), dtype=np.float32)
        self.log_probs = np.zeros(shape_time + (1,), dtype=np.float32)
        self.rewards = np.zeros(shape_time, dtype=np.float32)
        self.dones = np.zeros(self.rollout_length, dtype=np.float32)
        self.values = np.zeros(shape_time + (1,), dtype=np.float32)
        self.advantages = np.zeros_like(self.values)
        self.returns = np.zeros_like(self.values)
        self.pos = 0

    def add(
        self,
        local_obs: np.ndarray,
        global_obs: np.ndarray,
        actions: np.ndarray,
        log_probs: np.ndarray,
        rewards: np.ndarray,
        done: bool,
        values: np.ndarray,
    ) -> None:
        assert self.pos < self.rollout_length, "Rollout buffer overflow."
        self.local_obs[self.pos] = local_obs
        self.global_obs[self.pos] = global_obs
        self.actions[self.pos] = actions
        self.log_probs[self.pos] = log_probs
        self.rewards[self.pos] = rewards
        self.dones[self.pos] = float(done)
        self.values[self.pos] = values
        self.pos += 1

    def compute_returns_and_advantages(
        self,
        last_values: np.ndarray,
        last_done: bool,
        gamma: float,
        gae_lambda: float,
    ) -> None:
        last_gae = np.zeros((self.num_agents, 1), dtype=np.float32)
        for step in reversed(range(self.pos)):
            if step == self.pos - 1:
                next_non_terminal = 1.0 - float(last_done)
                next_values = last_values
            else:
                next_non_terminal = 1.0 - self.dones[step + 1]
                next_values = self.values[step + 1]
            delta = (
                self.rewards[step][:, None]
                + gamma * next_values * next_non_terminal
                - self.values[step]
            )
            last_gae = delta + gamma * gae_lambda * next_non_terminal * last_gae
            self.advantages[step] = last_gae
        self.returns[: self.pos] = self.advantages[: self.pos] + self.values[: self.pos]
        adv = self.advantages[: self.pos].reshape(-1, 1)
        self.advantages[: self.pos] = (
            (adv - adv.mean()) / (adv.std() + 1e-8)
        ).reshape(self.advantages[: self.pos].shape)

    def iter_minibatches(
        self,
        num_minibatches: int,
        device: torch.device,
    ) -> Iterator[Dict[str, torch.Tensor]]:
        total_samples = self.pos * self.num_agents
        indices = np.arange(total_samples)
        np.random.shuffle(indices)
        minibatch_size = max(1, total_samples // num_minibatches)

        flat_local_obs = self.local_obs[: self.pos].reshape(total_samples, self.obs_dim)
        flat_global_obs = self.global_obs[: self.pos].reshape(total_samples, self.global_obs_dim)
        flat_actions = self.actions[: self.pos].reshape(total_samples, self.action_dim)
        flat_log_probs = self.log_probs[: self.pos].reshape(total_samples, 1)
        flat_advantages = self.advantages[: self.pos].reshape(total_samples, 1)
        flat_returns = self.returns[: self.pos].reshape(total_samples, 1)

        for start in range(0, total_samples, minibatch_size):
            end = start + minibatch_size
            batch_idx = indices[start:end]
            yield {
                "local_obs": torch.tensor(flat_local_obs[batch_idx], dtype=torch.float32, device=device),
                "global_obs": torch.tensor(flat_global_obs[batch_idx], dtype=torch.float32, device=device),
                "actions": torch.tensor(flat_actions[batch_idx], dtype=torch.float32, device=device),
                "log_probs": torch.tensor(flat_log_probs[batch_idx], dtype=torch.float32, device=device),
                "advantages": torch.tensor(flat_advantages[batch_idx], dtype=torch.float32, device=device),
                "returns": torch.tensor(flat_returns[batch_idx], dtype=torch.float32, device=device),
            }


class MAPPOTrainer:
    def __init__(self, args: argparse.Namespace, config: MAPPOConfig) -> None:
        self.args = args
        self.config = config
        self.device = torch.device(args.device)
        self.env = RlGame(n=args.num_allies, m=args.num_enemies, render=args.render).unwrapped
        os.environ["SDL_VIDEODRIVER"] = "dummy"
        initial_obs = np.asarray(self.env.reset(), dtype=np.float32)
        if initial_obs.ndim != 2:
            raise ValueError("Environment observations must be shaped [num_agents, obs_dim].")
        self.num_agents, self.obs_dim = initial_obs.shape
        self.global_obs_dim = initial_obs.size
        self.action_dim = self.env.action_space.shape[0]
        self.max_action = float(self.env.action_space.high[0])
        config.rollout_length = args.rollout_length

        self.policy = MAPPOActorCritic(
            obs_dim=self.obs_dim,
            global_obs_dim=self.global_obs_dim,
            action_dim=self.action_dim,
            action_high=self.max_action,
        ).to(self.device)
        self.policy_optimizer = torch.optim.Adam(
            self.policy.policy_parameters(),
            lr=self.config.policy_lr,
        )
        self.value_optimizer = torch.optim.Adam(
            self.policy.value_parameters(),
            lr=self.config.value_lr,
        )
        self.buffer = RolloutBuffer(
            rollout_length=self.config.rollout_length,
            num_agents=self.num_agents,
            obs_dim=self.obs_dim,
            global_obs_dim=self.global_obs_dim,
            action_dim=self.action_dim,
        )
        self.current_obs = initial_obs
        self.rollout_rewards = []
        self.history: Dict[str, list] = {"episode_rewards": [], "policy_loss": [], "value_loss": []}
        os.makedirs(self.args.log_dir, exist_ok=True)

        if args.resume:
            self.load(args.resume)

    def _env_step(self, action: np.ndarray) -> Tuple[np.ndarray, np.ndarray, bool, Dict]:
        result = self.env.step(action)
        next_obs = np.asarray(result[0], dtype=np.float32)
        reward = np.asarray(result[1], dtype=np.float32)
        done = bool(result[2])
        info: Dict = {}
        if len(result) > 3:
            info["win"] = result[3]
        if len(result) > 4:
            info["team_counter"] = result[4]
        if len(result) > 5:
            info["extra"] = result[5]
        return next_obs, reward, done, info

    def collect_rollout(self) -> Dict[str, float]:
        self.buffer.reset()
        step = 0
        episode_reward = 0.0
        completed_reward_sum = 0.0
        episode_count = 0
        wins = 0
        obs = self.current_obs
        last_done = False

        while step < self.buffer.rollout_length:
            local_obs = torch.tensor(obs, dtype=torch.float32, device=self.device)
            global_obs_flat = obs.reshape(-1)
            global_obs = np.repeat(global_obs_flat[None, :], self.num_agents, axis=0)
            global_obs_tensor = torch.tensor(global_obs, dtype=torch.float32, device=self.device)
            with torch.no_grad():
                scaled_action, raw_action, log_prob, values = self.policy.act(local_obs, global_obs_tensor)
            next_obs, reward, done, info = self._env_step(scaled_action.cpu().numpy())
            if self.args.render:
                self.env.render()
            self.buffer.add(
                local_obs=local_obs.cpu().numpy(),
                global_obs=global_obs,
                actions=raw_action.cpu().numpy(),
                log_probs=log_prob.cpu().numpy(),
                rewards=reward,
                done=done,
                values=values.cpu().numpy(),
            )
            episode_reward += float(reward.mean())
            step += 1
            obs = next_obs
            last_done = done
            if done:
                if info.get("win"):
                    wins += 1
                self.history["episode_rewards"].append(episode_reward)
                completed_reward_sum += episode_reward
                episode_reward = 0.0
                episode_count += 1
                obs = np.asarray(self.env.reset(), dtype=np.float32)
        self.current_obs = obs
        last_values = values.cpu().numpy()
        self.buffer.compute_returns_and_advantages(
            last_values=last_values,
            last_done=last_done,
            gamma=self.config.gamma,
            gae_lambda=self.config.gae_lambda,
        )
        if episode_count == 0:
            avg_reward = episode_reward
        else:
            avg_reward = completed_reward_sum / max(episode_count, 1)
        return {
            "mean_episode_reward": avg_reward,
            "episodes": episode_count,
            "wins": wins,
        }

    def update_policy(self) -> Dict[str, float]:
        policy_losses, value_losses = [], []
        for _ in range(self.config.ppo_epochs):
            for batch in self.buffer.iter_minibatches(self.config.num_minibatches, self.device):
                new_log_probs, entropy = self.policy.evaluate_actions(batch["local_obs"], batch["actions"])
                ratio = torch.exp(new_log_probs - batch["log_probs"])
                surr1 = ratio * batch["advantages"]
                surr2 = torch.clamp(ratio, 1.0 - self.config.clip_coef, 1.0 + self.config.clip_coef) * batch["advantages"]
                policy_loss = -torch.min(surr1, surr2).mean()
                policy_loss_total = policy_loss - self.config.entropy_coef * entropy
                self.policy_optimizer.zero_grad()
                policy_loss_total.backward()
                nn.utils.clip_grad_norm_(self.policy.policy_parameters(), self.config.max_grad_norm)
                self.policy_optimizer.step()

                values = self.policy.evaluate_values(batch["global_obs"])
                value_loss = F.mse_loss(values, batch["returns"]) * self.config.value_coef
                self.value_optimizer.zero_grad()
                value_loss.backward()
                nn.utils.clip_grad_norm_(self.policy.value_parameters(), self.config.max_grad_norm)
                self.value_optimizer.step()

                policy_losses.append(policy_loss.item())
                value_losses.append(value_loss.item())
        policy_loss_mean = float(np.mean(policy_losses))
        value_loss_mean = float(np.mean(value_losses))
        self.history["policy_loss"].append(policy_loss_mean)
        self.history["value_loss"].append(value_loss_mean)
        return {"policy_loss": policy_loss_mean, "value_loss": value_loss_mean}

    def train(self) -> None:
        for update_idx in range(1, self.args.train_iters + 1):
            rollout_info = self.collect_rollout()
            update_info = self.update_policy()
            if update_idx % self.args.save_interval == 0 or update_idx == self.args.train_iters:
                self.save(self.args.model_path)
            print(
                f"[Update {update_idx}/{self.args.train_iters}] "
                f"policy_loss={update_info['policy_loss']:.4f} "
                f"value_loss={update_info['value_loss']:.4f} "
                f"episodes={rollout_info['episodes']} "
                f"wins={rollout_info['wins']}"
            )
        self._dump_history()

    def evaluate(self, episodes: int) -> None:
        rewards = []
        wins = 0
        for _ in range(episodes):
            obs = np.asarray(self.env.reset(), dtype=np.float32)
            done = False
            ep_reward = 0.0
            while True:
                local_obs = torch.tensor(obs, dtype=torch.float32, device=self.device)
                global_obs_flat = obs.reshape(-1)
                global_obs = np.repeat(global_obs_flat[None, :], self.num_agents, axis=0)
                global_obs_tensor = torch.tensor(global_obs, dtype=torch.float32, device=self.device)
                with torch.no_grad():
                    action, _, _, _ = self.policy.act(local_obs, global_obs_tensor, deterministic=True)
                obs, reward, done, info = self._env_step(action.cpu().numpy())
                ep_reward += float(np.asarray(reward).mean())
                if self.args.render:
                    self.env.render()
                if done:
                    wins += 1 if info.get("win") else 0
                    break
            rewards.append(ep_reward)
        print(
            f"[Evaluation] episodes={episodes} mean_reward={np.mean(rewards):.2f} "
            f"std_reward={np.std(rewards):.2f} win_rate={wins / max(episodes, 1):.2f}"
        )

    def save(self, path: str) -> None:
        state = {
            "model": self.policy.state_dict(),
            "policy_opt": self.policy_optimizer.state_dict(),
            "value_opt": self.value_optimizer.state_dict(),
            "config": asdict(self.config),
            "obs_dim": self.obs_dim,
            "global_obs_dim": self.global_obs_dim,
            "action_dim": self.action_dim,
            "num_agents": self.num_agents,
        }
        torch.save(state, path)

    def load(self, path: str) -> None:
        checkpoint = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(checkpoint["model"])
        self.policy_optimizer.load_state_dict(checkpoint["policy_opt"])
        self.value_optimizer.load_state_dict(checkpoint["value_opt"])
        print(f"Loaded MAPPO checkpoint from {path}")

    def _dump_history(self) -> None:
        with open(self.args.stats_path, "wb") as f:
            pkl.dump(self.history, f, protocol=pkl.HIGHEST_PROTOCOL)


def main() -> None:
    args = parse_args()
    os.makedirs(args.log_dir, exist_ok=True)
    args.model_path = args.model_path or DEFAULT_CHECKPOINT
    args.stats_path = args.stats_path or DEFAULT_STATS
    set_seed(args.seed)
    config = MAPPOConfig(rollout_length=args.rollout_length)
    trainer = MAPPOTrainer(args, config)
    if args.eval:
        if not os.path.exists(args.model_path) and not args.resume:
            raise FileNotFoundError(f"Checkpoint not found at {args.model_path!r}.")
        if not args.resume:
            trainer.load(args.model_path)
        trainer.evaluate(args.eval_episodes)
    else:
        trainer.train()


if __name__ == "__main__":
    main()
