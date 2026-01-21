# -*- coding: utf-8 -*-
"""
marl_ppo.py (replacement)

Multi-agent DSAC-T (a distributional-critic SAC variant) training script
for the continuous-control path planning environment in path_env.py (RlGame).

This file is intended to *replace* the original JAX/PPO implementation.
It follows the workflow style of main_SAC_dsact.py but makes the environment
interaction robust to path_env.py's step() returning 5 or 6 values.

Run:
    python marl_ppo.py

Key points:
- Centralized critic per agent: Q_i(s_global, a_i)
- Decentralized actors per agent: π_i(a_i | o_i)
- Continuous actions with tanh-squash
"""

import os
import sys
import time
import math
import random
from dataclasses import dataclass
from typing import Tuple, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------
# 1) Import environment (robust)
# -----------------------------
def make_env(n_agents: int, m_enemy: int, render: bool):
    """
    Try multiple import paths, because different project layouts exist:
    - main_SAC_dsact.py uses: from rl_env.path_env import RlGame
    - your upload shows path_env.py defines RlGame directly.
    """
    try:
        from rl_env.path_env import RlGame  # type: ignore
    except Exception:
        from path_env import RlGame  # type: ignore

    env = RlGame(n=n_agents, m=m_enemy, render=render)
    # Some gym envs have .unwrapped; keep compatible:
    return getattr(env, "unwrapped", env)


# -----------------------------
# 2) Utils
# -----------------------------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def huber_loss_tensor(pred: torch.Tensor, target: torch.Tensor, delta: float = 1.0) -> torch.Tensor:
    diff = pred - target
    abs_diff = diff.abs()
    quadratic = torch.clamp(abs_diff, max=delta)
    linear = abs_diff - quadratic
    return 0.5 * quadratic.pow(2) + delta * linear


def sample_from_distribution(mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    noise = torch.randn_like(mean)
    noise = torch.clamp(noise, -3.0, 3.0)
    return mean + noise * std


def compute_target_q(
    reward: torch.Tensor,
    done: torch.Tensor,
    current_q: torch.Tensor,
    running_std: torch.Tensor,
    next_q_mean: torch.Tensor,
    next_q_sample: torch.Tensor,
    next_log_prob: torch.Tensor,
    alpha: torch.Tensor,
    gamma: float,
    td_bound_scale: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    DSAC-T style target:
    - target_q uses mean bootstrap
    - target_q_bound uses sampled bootstrap + TD clipping by running std
    """
    target_q = reward + (1.0 - done) * gamma * (next_q_mean - alpha * next_log_prob)
    target_q_sample = reward + (1.0 - done) * gamma * (next_q_sample - alpha * next_log_prob)

    td_bound = td_bound_scale * running_std
    difference = torch.clamp(target_q_sample - current_q, -td_bound, td_bound)
    target_q_bound = current_q + difference
    return target_q.detach(), target_q_bound.detach()


class OrnsteinUhlenbeckNoise:
    """Optional exploration noise (not required for SAC, but kept for compatibility)."""
    def __init__(self, mu: np.ndarray, sigma=0.1, theta=0.1, dt=1e-2, x0=None):
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.x0 = x0
        self.reset()

    def __call__(self):
        x = (
            self.x_prev
            + self.theta * (self.mu - self.x_prev) * self.dt
            + self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        )
        self.x_prev = x
        return x

    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)


# -----------------------------
# 3) Networks
# -----------------------------
class ActorNet(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, max_action: float):
        super().__init__()
        self.max_action = max_action
        self.fc1 = nn.Linear(obs_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.mean_head = nn.Linear(256, act_dim)
        self.log_std_head = nn.Linear(256, act_dim)

        for m in [self.fc1, self.fc2, self.mean_head, self.log_std_head]:
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.1)
                nn.init.zeros_(m.bias)

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = F.relu(self.fc1(obs))
        x = F.relu(self.fc2(x))
        mean = self.max_action * torch.tanh(self.mean_head(x))
        log_std = torch.clamp(self.log_std_head(x), -20, 2)
        std = log_std.exp()
        return mean, std


class CriticNet(nn.Module):
    """
    Distributional Q: outputs (mean,std) for two critics.
    Input: concatenated [s_global, a_i] where a_i is *this agent's action only*.
    """
    def __init__(self, global_state_dim: int, act_dim: int, hidden_size: int = 512):
        super().__init__()
        total_in = global_state_dim + act_dim

        # Q1
        self.q1_fc1 = nn.Linear(total_in, hidden_size)
        self.q1_fc2 = nn.Linear(hidden_size, hidden_size)
        self.q1_mean = nn.Linear(hidden_size, 1)
        self.q1_std = nn.Linear(hidden_size, 1)

        # Q2
        self.q2_fc1 = nn.Linear(total_in, hidden_size)
        self.q2_fc2 = nn.Linear(hidden_size, hidden_size)
        self.q2_mean = nn.Linear(hidden_size, 1)
        self.q2_std = nn.Linear(hidden_size, 1)

        for layer in [
            self.q1_fc1, self.q1_fc2, self.q1_mean, self.q1_std,
            self.q2_fc1, self.q2_fc2, self.q2_mean, self.q2_std
        ]:
            if isinstance(layer, nn.Linear):
                nn.init.normal_(layer.weight, mean=0.0, std=0.1)
                nn.init.zeros_(layer.bias)

    def _branch(self, fc1, fc2, mean_head, std_head, x):
        h = F.relu(fc1(x))
        h = F.relu(fc2(h))
        mean = mean_head(h)
        log_std = torch.clamp(std_head(h), min=-5.0, max=2.0)
        std = F.softplus(log_std) + 1e-4
        return mean, std

    def forward(self, s_global: torch.Tensor, a_i: torch.Tensor):
        x = torch.cat([s_global, a_i], dim=-1)
        q1_mean, q1_std = self._branch(self.q1_fc1, self.q1_fc2, self.q1_mean, self.q1_std, x)
        q2_mean, q2_std = self._branch(self.q2_fc1, self.q2_fc2, self.q2_mean, self.q2_std, x)
        return q1_mean, q1_std, q2_mean, q2_std


# -----------------------------
# 4) Replay Buffer
# -----------------------------
class ReplayBuffer:
    def __init__(self, capacity: int, global_state_dim: int, total_action_dim: int, num_agents: int):
        self.capacity = capacity
        self.num_agents = num_agents
        # store: s, a_all, r_all, done_all, s2
        self.s = np.zeros((capacity, global_state_dim), dtype=np.float32)
        self.a = np.zeros((capacity, total_action_dim), dtype=np.float32)
        self.r = np.zeros((capacity, num_agents), dtype=np.float32)
        self.d = np.zeros((capacity, num_agents), dtype=np.float32)
        self.s2 = np.zeros((capacity, global_state_dim), dtype=np.float32)
        self.ptr = 0
        self.size = 0

    def add(self, s, a_all, r_all, done_all, s2):
        idx = self.ptr
        self.s[idx] = s
        self.a[idx] = a_all
        self.r[idx] = r_all
        self.d[idx] = done_all
        self.s2[idx] = s2
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int):
        assert self.size >= batch_size, "ReplayBuffer: not enough samples"
        idx = np.random.randint(0, self.size, size=batch_size)
        return (
            torch.from_numpy(self.s[idx]),
            torch.from_numpy(self.a[idx]),
            torch.from_numpy(self.r[idx]),
            torch.from_numpy(self.d[idx]),
            torch.from_numpy(self.s2[idx]),
        )


# -----------------------------
# 5) Agent wrappers
# -----------------------------
class SACActor:
    def __init__(self, obs_dim: int, act_dim: int, max_action: float, lr: float, device: torch.device):
        self.net = ActorNet(obs_dim, act_dim, max_action).to(device)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=lr)
        self.device = device
        self.max_action = max_action

    @torch.no_grad()
    def act(self, obs: np.ndarray, deterministic: bool = False) -> np.ndarray:
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        mean, std = self.net(obs_t)
        if deterministic:
            a = torch.tanh(mean)
        else:
            dist = torch.distributions.Normal(mean, std)
            a = torch.tanh(dist.rsample())
        a = torch.clamp(a, -self.max_action, self.max_action)
        return a.squeeze(0).cpu().numpy()

    def evaluate(self, obs_t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Return action and log_prob for SAC loss (tanh-squashed normal).
        obs_t: [B, obs_dim]
        """
        mean, std = self.net(obs_t)
        dist = torch.distributions.Normal(mean, std)
        noise = torch.randn_like(mean)
        pre_tanh = mean + std * noise
        a = torch.tanh(pre_tanh)
        a = torch.clamp(a, -self.max_action, self.max_action)

        # log prob with tanh correction
        log_prob = dist.log_prob(pre_tanh) - torch.log(1 - a.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=1, keepdim=True)
        return a, log_prob

    def update(self, loss: torch.Tensor):
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()


class DSACCritic:
    def __init__(self, global_state_dim: int, act_dim: int, lr: float, tau: float, device: torch.device):
        self.q = CriticNet(global_state_dim, act_dim).to(device)
        self.q_targ = CriticNet(global_state_dim, act_dim).to(device)
        self.q_targ.load_state_dict(self.q.state_dict())
        self.opt = torch.optim.Adam(self.q.parameters(), lr=lr, eps=1e-5)
        self.tau = tau
        self.device = device

        # running std for DSAC-T bounding (EMA)
        self.mean_std1: Optional[torch.Tensor] = None
        self.mean_std2: Optional[torch.Tensor] = None

    @torch.no_grad()
    def soft_update(self):
        for tp, p in zip(self.q_targ.parameters(), self.q.parameters()):
            tp.data.mul_(1.0 - self.tau).add_(p.data, alpha=self.tau)

    def forward(self, s_global, a_i):
        return self.q(s_global, a_i)

    def forward_target(self, s_global, a_i):
        return self.q_targ(s_global, a_i)


class EntropyCoef:
    def __init__(self, act_dim: int, lr: float, device: torch.device):
        self.target_entropy = -float(act_dim)
        self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
        self.opt = torch.optim.Adam([self.log_alpha], lr=lr)

    @property
    def alpha(self) -> torch.Tensor:
        return self.log_alpha.exp()

    def update(self, loss: torch.Tensor):
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()


# -----------------------------
# 6) Config
# -----------------------------
@dataclass
class Config:
    # env
    N_AGENT: int = 1
    M_ENEMY: int = 1
    RENDER: bool = False

    # training
    EP_MAX: int = 1000
    EP_LEN: int = 1000
    GAMMA: float = 0.99

    # SAC/DSAC-T
    ACTOR_LR: float = 3e-4
    CRITIC_LR: float = 3e-4
    ALPHA_LR: float = 1e-4
    TAU: float = 1e-2
    BATCH: int = 128
    REPLAY_SIZE: int = 100000
    START_STEPS: int = 2000            # how many steps to collect before updates
    UPDATE_EVERY: int = 1              # update frequency (steps)
    DELAY_UPDATE: int = 2              # delayed actor updates

    # DSAC-T bounding hyperparams (copied style from main_SAC_dsact.py)
    TAU_B: float = 0.01
    STD_BIAS: float = 0.1
    HUBER_DELTA: float = 50.0
    TD_BOUND_SCALE: float = 3.0

    # exploration noise (optional)
    USE_OU_NOISE: bool = True
    OU_NOISE_EPISODES: int = 20

    # misc
    SEED: int = 42
    SAVE_DIR: str = "./checkpoints_path_dsact"
    SAVE_EVERY_EP: int = 50


# -----------------------------
# 7) Training
# -----------------------------
def unwrap_step(step_out):
    """
    path_env.py has two variants:
    - training: return hero_state, r, done, win, team_counter
    - test:     return hero_state, r, done, win, team_counter, dis_1_agent_0_to_1

    This helper accepts both.
    """
    if isinstance(step_out, (list, tuple)) and len(step_out) >= 5:
        obs2, r, done, win, team_counter = step_out[:5]
        extra = step_out[5:] if len(step_out) > 5 else ()
        return obs2, r, done, win, team_counter, extra
    raise RuntimeError(f"Unexpected env.step() return: {type(step_out)}")


def train(cfg: Config):
    set_seed(cfg.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env = make_env(cfg.N_AGENT, cfg.M_ENEMY, cfg.RENDER)
    obs = env.reset()
    obs = np.asarray(obs, dtype=np.float32)

    num_agents = cfg.N_AGENT + cfg.M_ENEMY
    obs_dim = obs.shape[-1]
    act_dim = int(env.action_space.shape[0])
    max_action = float(env.action_space.high[0])

    global_state_dim = obs_dim * num_agents
    total_action_dim = act_dim * num_agents

    actors: List[SACActor] = []
    critics: List[DSACCritic] = []
    entropies: List[EntropyCoef] = []
    for _ in range(num_agents):
        actors.append(SACActor(obs_dim, act_dim, max_action, cfg.ACTOR_LR, device))
        critics.append(DSACCritic(global_state_dim, act_dim, cfg.CRITIC_LR, cfg.TAU, device))
        entropies.append(EntropyCoef(act_dim, cfg.ALPHA_LR, device))

    buf = ReplayBuffer(cfg.REPLAY_SIZE, global_state_dim, total_action_dim, num_agents)
    ou = OrnsteinUhlenbeckNoise(mu=np.zeros((num_agents, act_dim), dtype=np.float32))

    os.makedirs(cfg.SAVE_DIR, exist_ok=True)

    global_step = 0
    update_step = 0

    ep_returns = []

    for ep in range(cfg.EP_MAX):
        obs = env.reset()
        obs = np.asarray(obs, dtype=np.float32)

        ep_ret = np.zeros((num_agents,), dtype=np.float32)

        for t in range(cfg.EP_LEN):
            global_step += 1

            actions = np.zeros((num_agents, act_dim), dtype=np.float32)
            for i in range(num_agents):
                actions[i] = actors[i].act(obs[i], deterministic=False)

            # optional OU noise in early training
            if cfg.USE_OU_NOISE and ep <= cfg.OU_NOISE_EPISODES:
                actions = actions + ou()
            actions = np.clip(actions, -max_action, max_action)

            step_out = env.step(actions)
            obs2, r, done, win, team_counter, _extra = unwrap_step(step_out)

            obs2 = np.asarray(obs2, dtype=np.float32)
            r = np.asarray(r, dtype=np.float32).reshape(num_agents, -1)
            r = r.squeeze(-1)  # [num_agents]
            done_flag = float(done)
            done_all = np.full((num_agents,), done_flag, dtype=np.float32)

            buf.add(
                obs.reshape(-1),
                actions.reshape(-1),
                r.reshape(-1),
                done_all,
                obs2.reshape(-1),
            )

            ep_ret += r
            obs = obs2

            # update
            if buf.size >= max(cfg.START_STEPS, cfg.BATCH) and (global_step % cfg.UPDATE_EVERY == 0):
                update_step += 1
                update_actor_flag = (update_step % cfg.DELAY_UPDATE == 0)

                b_s, b_a, b_r, b_d, b_s2 = buf.sample(cfg.BATCH)
                b_s = b_s.to(device)
                b_a = b_a.to(device)
                b_r = b_r.to(device)
                b_d = b_d.to(device)
                b_s2 = b_s2.to(device)

                # update each agent (centralized critic, decentralized actor)
                for i in range(num_agents):
                    actor = actors[i]
                    critic = critics[i]
                    ent = entropies[i]

                    # slices
                    s_i = b_s[:, obs_dim * i: obs_dim * (i + 1)]
                    s2_i = b_s2[:, obs_dim * i: obs_dim * (i + 1)]
                    a_i = b_a[:, act_dim * i: act_dim * (i + 1)]
                    r_i = b_r[:, i:i + 1]
                    d_i = b_d[:, i:i + 1]

                    # target action from policy
                    with torch.no_grad():
                        a2_i, logp2_i = actor.evaluate(s2_i)
                        q1n_m, q1n_s, q2n_m, q2n_s = critic.forward_target(b_s2, a2_i)

                        q1n_sample = sample_from_distribution(q1n_m, q1n_s)
                        q2n_sample = sample_from_distribution(q2n_m, q2n_s)

                        q_next_mean = torch.min(q1n_m, q2n_m)
                        q_next_sample = torch.where(q1n_m < q2n_m, q1n_sample, q2n_sample)

                    # current q dist
                    q1_m, q1_s, q2_m, q2_s = critic.forward(b_s, a_i)

                    # running std EMA (detached)
                    std1_mean = torch.mean(q1_s.detach())
                    std2_mean = torch.mean(q2_s.detach())
                    if critic.mean_std1 is None:
                        critic.mean_std1 = std1_mean
                    else:
                        critic.mean_std1 = (1.0 - cfg.TAU_B) * critic.mean_std1 + cfg.TAU_B * std1_mean
                    if critic.mean_std2 is None:
                        critic.mean_std2 = std2_mean
                    else:
                        critic.mean_std2 = (1.0 - cfg.TAU_B) * critic.mean_std2 + cfg.TAU_B * std2_mean

                    alpha_val = ent.alpha.detach()

                    # dsac targets + bounds
                    target_q1, target_q1_bound = compute_target_q(
                        r_i, d_i, q1_m.detach(), critic.mean_std1.detach(),
                        q_next_mean.detach(), q_next_sample.detach(),
                        logp2_i.detach(), alpha_val, cfg.GAMMA, cfg.TD_BOUND_SCALE
                    )
                    target_q2, target_q2_bound = compute_target_q(
                        r_i, d_i, q2_m.detach(), critic.mean_std2.detach(),
                        q_next_mean.detach(), q_next_sample.detach(),
                        logp2_i.detach(), alpha_val, cfg.GAMMA, cfg.TD_BOUND_SCALE
                    )

                    # DSAC-T weighted losses (same style as main_SAC_dsact.py)
                    q1_s_det = torch.clamp(q1_s, min=0.).detach()
                    q2_s_det = torch.clamp(q2_s, min=0.).detach()
                    ratio1 = ((critic.mean_std1.detach() ** 2) / (q1_s_det.pow(2) + cfg.STD_BIAS)).clamp(0.1, 10.0)
                    ratio2 = ((critic.mean_std2.detach() ** 2) / (q2_s_det.pow(2) + cfg.STD_BIAS)).clamp(0.1, 10.0)

                    q1_loss = ratio1 * (
                        huber_loss_tensor(q1_m, target_q1, cfg.HUBER_DELTA)
                        + q1_s * (
                            q1_s_det.pow(2) - huber_loss_tensor(q1_m.detach(), target_q1_bound, cfg.HUBER_DELTA)
                        ) / (q1_s_det + cfg.STD_BIAS)
                    )
                    q2_loss = ratio2 * (
                        huber_loss_tensor(q2_m, target_q2, cfg.HUBER_DELTA)
                        + q2_s * (
                            q2_s_det.pow(2) - huber_loss_tensor(q2_m.detach(), target_q2_bound, cfg.HUBER_DELTA)
                        ) / (q2_s_det + cfg.STD_BIAS)
                    )

                    critic_loss = (q1_loss.mean() + q2_loss.mean())
                    critic.opt.zero_grad()
                    critic_loss.backward()
                    critic.opt.step()

                    if update_actor_flag:
                        a_pi, logp = actor.evaluate(s_i)
                        q1_pi, _, q2_pi, _ = critic.forward(b_s, a_pi)
                        q_pi = torch.min(q1_pi, q2_pi)

                        actor_loss = (alpha_val * logp - q_pi).mean()
                        actor.update(actor_loss)

                        alpha_loss = -(ent.log_alpha * (logp.detach() + ent.target_entropy)).mean()
                        ent.update(alpha_loss)

                if update_actor_flag:
                    for c in critics:
                        c.soft_update()

            if cfg.RENDER:
                env.render()

            if done:
                break

        ep_returns.append(ep_ret.copy())
        print(f"[EP {ep:04d}] return(mean)={ep_ret.mean():.3f}  return(per-agent)={ep_ret}  steps={t+1}")

        # save models
        if (ep + 1) % cfg.SAVE_EVERY_EP == 0:
            ckpt = {
                "cfg": cfg.__dict__,
                "episode": ep,
                "actors": [a.net.state_dict() for a in actors],
                "critics": [c.q.state_dict() for c in critics],
                "critics_target": [c.q_targ.state_dict() for c in critics],
                "log_alpha": [e.log_alpha.detach().cpu().numpy() for e in entropies],
            }
            path = os.path.join(cfg.SAVE_DIR, f"dsact_path_ep{ep+1}.pt")
            torch.save(ckpt, path)
            print(f"Saved checkpoint -> {path}")

    try:
        env.close()
    except Exception:
        pass

    return np.array(ep_returns, dtype=np.float32)


def main():
    cfg = Config()
    train(cfg)


if __name__ == "__main__":
    main()
