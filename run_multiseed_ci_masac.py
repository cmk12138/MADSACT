# -*- coding: utf-8 -*-
"""
run_multiseed_ci_masac.py

Multi-seed runner (train + eval + mean±95%CI) for MASAC as implemented in main_SAC.py.

- Trains MASAC (multi-agent SAC: one SAC per agent) for each seed (optional).
- Evaluates checkpoints with Monte Carlo episodes per seed.
- Aggregates metrics across seeds with mean ± 95% CI (normal approximation).

Metrics (aligned with your DSAC-T runner):
  MCR: win_rate
  FKR: team_counter / steps_taken
  JT : average episode length (steps)
  JS : proxy path length = sum leader speed state[0][2]
  JC : proxy energy = sum(abs(leader_action))

Usage:
  # Train + test (5 seeds)
  python run_multiseed_ci_masac.py --train --seeds 0,1,2,3,4 --followers 1 --ep-max 500 --test-episodes 100
  python run_multiseed_ci_masac.py --train --seeds 2,3,4 --followers 1 --ep-max 500 --test-episodes 10

  # Only test existing checkpoints
  python run_multiseed_ci_masac.py --seeds 0,1,2,3,4 --followers 1 --test-episodes 100
  python run_multiseed_ci_masac.py --seeds 0,1 --followers 1 --test-episodes 100
  python run_multiseed_ci_masac.py --train --seeds 2,3,4 --followers 1 --ep-max 500 --test-episodes 10

Outputs:
  ./multiseed_runs_masac/
    seed_0/checkpoints/*.pth
    seed_0/metrics_seed_0.json
    ...
    summary_across_seeds.csv

Notes:
- This script removes hard-coded Windows save paths in main_SAC.py by saving into output_root.
- It does NOT depend on main_SAC.py at runtime; it reuses the same network/training logic in a parameterized way.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def make_env(n_agent: int, m_enemy: int, render: bool = False):
    """Create env, compatible with both rl_env.path_env and local path_env.py."""
    try:
        from rl_env.path_env import RlGame  # type: ignore
    except Exception:
        from path_env import RlGame  # type: ignore

    env = RlGame(n=n_agent, m=m_enemy, render=render)
    return getattr(env, "unwrapped", env)


def unwrap_step(step_out):
    """Unpack env.step output: (obs, reward, done, win, team_counter, *extra)."""
    if isinstance(step_out, (list, tuple)) and len(step_out) >= 5:
        obs2, reward, done, win, team_counter = step_out[:5]
        extra = step_out[5:] if len(step_out) > 5 else ()
        return obs2, reward, done, win, team_counter, extra
    raise ValueError(f"Unexpected env.step output: {type(step_out)} / {step_out}")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class OrnsteinUhlenbeckNoise:
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


class ActorNet(nn.Module):
    def __init__(self, inp: int, outp: int, max_action: float):
        super().__init__()
        self.max_action = float(max_action)
        self.in_to_y1 = nn.Linear(inp, 256)
        self.y1_to_y2 = nn.Linear(256, 256)
        self.out = nn.Linear(256, outp)
        self.std_out = nn.Linear(256, outp)

        for m in [self.in_to_y1, self.y1_to_y2, self.out, self.std_out]:
            m.weight.data.normal_(0, 0.1)

    def forward(self, s: torch.Tensor):
        x = F.relu(self.in_to_y1(s))
        x = F.relu(self.y1_to_y2(x))
        mean = self.max_action * torch.tanh(self.out(x))
        log_std = self.std_out(x)
        log_std = torch.clamp(log_std, -20, 2)
        std = log_std.exp()
        return mean, std


class CriticNet(nn.Module):
    def __init__(self, state_dim_all: int, act_dim: int):
        super().__init__()
        self.in_to_y1 = nn.Linear(state_dim_all + act_dim, 256)
        self.y1_to_y2 = nn.Linear(256, 256)
        self.out = nn.Linear(256, 1)

        self.q2_in_to_y1 = nn.Linear(state_dim_all + act_dim, 256)
        self.q2_y1_to_y2 = nn.Linear(256, 256)
        self.q2_out = nn.Linear(256, 1)

        for m in [
            self.in_to_y1,
            self.y1_to_y2,
            self.out,
            self.q2_in_to_y1,
            self.q2_y1_to_y2,
            self.q2_out,
        ]:
            m.weight.data.normal_(0, 0.1)

    def forward(self, s_all: torch.Tensor, a_i: torch.Tensor):
        x = torch.cat((s_all, a_i), dim=1)

        q1 = F.relu(self.in_to_y1(x))
        q1 = F.relu(self.y1_to_y2(q1))
        q1 = self.out(q1)

        q2 = F.relu(self.q2_in_to_y1(x))
        q2 = F.relu(self.q2_y1_to_y2(q2))
        q2 = self.q2_out(q2)

        return q1, q2


class Memory:
    def __init__(self, capacity: int, dims: int):
        self.capacity = int(capacity)
        self.mem = np.zeros((capacity, dims), dtype=np.float32)
        self.counter = 0

    def store_transition(self, s: np.ndarray, a: np.ndarray, r: np.ndarray, s_: np.ndarray):
        tran = np.hstack((s, a, r, s_)).astype(np.float32)
        idx = self.counter % self.capacity
        self.mem[idx, :] = tran
        self.counter += 1

    def sample(self, n: int) -> np.ndarray:
        assert self.counter >= self.capacity, "replay buffer not full yet"
        idx = np.random.choice(self.capacity, n)
        return self.mem[idx, :]


class Actor:
    def __init__(self, state_dim: int, act_dim: int, max_action: float, min_action: float, policy_lr: float):
        self.max_action = float(max_action)
        self.min_action = float(min_action)
        self.net = ActorNet(state_dim, act_dim, max_action)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=policy_lr)

    def choose_action(self, s: np.ndarray, deterministic: bool = False) -> np.ndarray:
        s_t = torch.as_tensor(s, dtype=torch.float32).unsqueeze(0)
        mean, std = self.net(s_t)
        if deterministic:
            a = mean
        else:
            dist = torch.distributions.Normal(mean, std)
            a = dist.sample()
        a = torch.clamp(a, self.min_action, self.max_action)
        return a.squeeze(0).detach().cpu().numpy()

    def evaluate(self, s: torch.Tensor):
        mean, std = self.net(s)
        dist = torch.distributions.Normal(mean, std)
        z = torch.distributions.Normal(0, 1).sample(mean.shape).to(mean.device)
        pre_tanh = mean + std * z
        a = torch.tanh(pre_tanh)
        a = torch.clamp(a, self.min_action, self.max_action)
        logp = dist.log_prob(pre_tanh) - torch.log(1 - a.pow(2) + 1e-6)
        logp = logp.sum(dim=1, keepdim=True)
        return a, logp

    def learn(self, loss: torch.Tensor) -> None:
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()


class Entropy:
    def __init__(self, q_lr: float, target_entropy: float = -0.1):
        self.target_entropy = float(target_entropy)
        self.log_alpha = torch.zeros(1, requires_grad=True)
        self.alpha = self.log_alpha.exp().detach()
        self.opt = torch.optim.Adam([self.log_alpha], lr=q_lr)

    def learn(self, loss: torch.Tensor) -> None:
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        self.alpha = self.log_alpha.exp().detach()


class Critic:
    def __init__(self, state_dim_all: int, act_dim: int, value_lr: float, tau: float):
        self.tau = float(tau)
        self.q = CriticNet(state_dim_all, act_dim)
        self.q_targ = CriticNet(state_dim_all, act_dim)
        self.q_targ.load_state_dict(self.q.state_dict())
        self.opt = torch.optim.Adam(self.q.parameters(), lr=value_lr, eps=1e-5)
        self.mse = nn.MSELoss()

    def soft_update(self) -> None:
        for tp, p in zip(self.q_targ.parameters(), self.q.parameters()):
            tp.data.copy_(tp.data * (1.0 - self.tau) + p.data * self.tau)

    def get_q(self, s_all: torch.Tensor, a_i: torch.Tensor):
        return self.q(s_all, a_i)

    def get_q_targ(self, s_all: torch.Tensor, a_i: torch.Tensor):
        return self.q_targ(s_all, a_i)

    def learn(self, q1: torch.Tensor, q2: torch.Tensor, y: torch.Tensor) -> None:
        loss = self.mse(q1, y) + self.mse(q2, y)
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()


@dataclass
class MASACConfig:
    seed: int = 0
    n_agent: int = 1
    m_enemy: int = 1
    render: bool = False

    ep_max: int = 50000
    ep_len: int = 1000

    gamma: float = 0.9
    q_lr: float = 3e-4
    value_lr: float = 3e-3
    policy_lr: float = 1e-3
    batch: int = 128
    tau: float = 1e-2
    memory_capacity: int = 20000
    ou_sigma: float = 0.1
    ou_theta: float = 0.1
    ou_dt: float = 1e-2
    ou_episodes: int = 20

    save_every_ep: int = 500

    device: str = "cpu"


def alpha_tensor(ent: Entropy) -> torch.Tensor:
    return ent.log_alpha.exp()


def train_masac(cfg: MASACConfig, ckpt_dir: str) -> str:
    set_seed(cfg.seed)
    device = torch.device(cfg.device)

    env = make_env(cfg.n_agent, cfg.m_enemy, render=cfg.render)
    obs0 = np.asarray(env.reset(), dtype=np.float32)
    num_agents = obs0.shape[0]
    state_dim = obs0.shape[-1]
    act_dim = int(env.action_space.shape[0])
    max_action = float(env.action_space.high[0])
    min_action = float(env.action_space.low[0])

    state_dim_all = state_dim * num_agents
    mem_dims = 2 * state_dim_all + act_dim * num_agents + 1 * num_agents
    memory = Memory(cfg.memory_capacity, mem_dims)

    actors = [Actor(state_dim, act_dim, max_action, min_action, cfg.policy_lr) for _ in range(num_agents)]
    critics = [Critic(state_dim_all, act_dim, cfg.value_lr, cfg.tau) for _ in range(num_agents)]
    entrs = [Entropy(cfg.q_lr, target_entropy=-0.1) for _ in range(num_agents)]

    for a in actors:
        a.net.to(device)
    for c in critics:
        c.q.to(device)
        c.q_targ.to(device)

    ou_noise = OrnsteinUhlenbeckNoise(
        mu=np.zeros((num_agents, act_dim), dtype=np.float32),
        sigma=cfg.ou_sigma,
        theta=cfg.ou_theta,
        dt=cfg.ou_dt,
    )

    action = np.zeros((num_agents, act_dim), dtype=np.float32)
    last_ckpt_path = ""

    for ep in range(cfg.ep_max):
        obs = np.asarray(env.reset(), dtype=np.float32)
        for _t in range(cfg.ep_len):
            for i in range(num_agents):
                action[i] = actors[i].choose_action(obs[i], deterministic=False)

            noise = ou_noise() if ep <= cfg.ou_episodes else 0.0
            act_noisy = np.clip(action + noise, -max_action, max_action).astype(np.float32)

            obs2, reward, done, win, team_counter, extra = unwrap_step(env.step(act_noisy))
            obs2 = np.asarray(obs2, dtype=np.float32)
            reward = np.asarray(reward, dtype=np.float32).reshape(-1, 1)

            memory.store_transition(obs.flatten(), act_noisy.flatten(), reward.flatten(), obs2.flatten())

            if memory.counter > cfg.memory_capacity:
                batch_mem = memory.sample(cfg.batch)

                b_s = batch_mem[:, :state_dim_all]
                b_a = batch_mem[:, state_dim_all: state_dim_all + act_dim * num_agents]
                b_r = batch_mem[:, -state_dim_all - num_agents: -state_dim_all]
                b_s2 = batch_mem[:, -state_dim_all:]

                b_s = torch.as_tensor(b_s, dtype=torch.float32, device=device)
                b_a = torch.as_tensor(b_a, dtype=torch.float32, device=device)
                b_r = torch.as_tensor(b_r, dtype=torch.float32, device=device)
                b_s2 = torch.as_tensor(b_s2, dtype=torch.float32, device=device)

                for i in range(num_agents):
                    new_a, logp2 = actors[i].evaluate(b_s2[:, state_dim * i: state_dim * (i + 1)])
                    q1_t, q2_t = critics[i].get_q_targ(b_s2, new_a)
                    y = b_r[:, i:(i + 1)] + cfg.gamma * (torch.min(q1_t, q2_t) - entrs[i].alpha.to(device) * logp2)

                    a_i = b_a[:, act_dim * i: act_dim * (i + 1)]
                    q1, q2 = critics[i].get_q(b_s, a_i)
                    critics[i].learn(q1, q2, y.detach())

                    a_new, logp = actors[i].evaluate(b_s[:, state_dim * i: state_dim * (i + 1)])
                    q1_pi, q2_pi = critics[i].get_q(b_s, a_new)
                    q_pi = torch.min(q1_pi, q2_pi)
                    actor_loss = (entrs[i].alpha.to(device) * logp - q_pi).mean()

                    alpha_loss = -(alpha_tensor(entrs[i]) * (logp + entrs[i].target_entropy).detach()).mean()

                    actors[i].learn(actor_loss)
                    entrs[i].learn(alpha_loss)
                    critics[i].soft_update()

            obs = obs2
            if bool(done):
                break

        if ((ep + 1) % cfg.save_every_ep == 0) or (ep + 1 == cfg.ep_max):
            os.makedirs(ckpt_dir, exist_ok=True)
            ckpt_path = os.path.join(ckpt_dir, f"masac_seed{cfg.seed}_ep{ep+1}.pth")
            torch.save(
                {
                    "cfg": cfg.__dict__,
                    "actors": [a.net.state_dict() for a in actors],
                    "ep": ep + 1,
                },
                ckpt_path,
            )
            last_ckpt_path = ckpt_path

    try:
        env.close()
    except Exception:
        pass

    if not last_ckpt_path:
        raise RuntimeError("No checkpoint saved.")
    return last_ckpt_path


def load_actors_from_ckpt(ckpt_path: str, obs_dim: int, act_dim: int, max_action: float, min_action: float, policy_lr: float):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(ckpt_path, map_location=device)
    actor_states = ckpt["actors"]
    actors = []
    for sd in actor_states:
        a = Actor(obs_dim, act_dim, max_action, min_action, policy_lr)
        a.net.load_state_dict(sd)
        a.net.to(device)
        a.net.eval()
        actors.append(a)
    return actors, ckpt


@torch.no_grad()
def evaluate_checkpoint(ckpt_path: str, followers: int, test_episodes: int, ep_len: int, render: bool = False) -> Dict[str, float]:
    env = make_env(1, followers, render=render)
    s0 = np.asarray(env.reset(), dtype=np.float32)
    num_agents = s0.shape[0]
    obs_dim = s0.shape[-1]
    act_dim = int(env.action_space.shape[0])
    max_action = float(env.action_space.high[0])
    min_action = float(env.action_space.low[0])

    actors, ckpt = load_actors_from_ckpt(ckpt_path, obs_dim, act_dim, max_action, min_action, policy_lr=1e-3)

    if len(actors) < num_agents:
        base = actors[0] if len(actors) == 1 else actors[1]
        while len(actors) < num_agents:
            actors.append(base)

    win_times = 0.0
    sum_fkr = 0.0
    sum_jt = 0.0
    sum_js = 0.0
    sum_jc = 0.0

    for _ in range(int(test_episodes)):
        s = np.asarray(env.reset(), dtype=np.float32)
        steps_taken = 0
        last_team_counter = 0.0
        integral_V = 0.0
        integral_U = 0.0
        done = False
        win = False

        for _t in range(int(ep_len)):
            act = np.zeros((num_agents, act_dim), dtype=np.float32)
            for i in range(num_agents):
                act[i] = actors[i].choose_action(s[i], deterministic=True)

            s2, r, done, win, team_counter, extra = unwrap_step(env.step(act))

            integral_V += float(s[0][2])
            integral_U += float(np.sum(np.abs(act[0])))

            steps_taken += 1
            last_team_counter = float(team_counter)
            s = np.asarray(s2, dtype=np.float32)

            if bool(done):
                break

        fkr = last_team_counter / max(1, steps_taken)
        sum_fkr += float(fkr)
        sum_jt += float(steps_taken)
        sum_js += float(integral_V)
        sum_jc += float(integral_U)
        if bool(win):
            win_times += 1.0

    try:
        env.close()
    except Exception:
        pass

    return {
        "mcr": win_times / float(test_episodes),
        "fkr": sum_fkr / float(test_episodes),
        "jt": sum_jt / float(test_episodes),
        "js": sum_js / float(test_episodes),
        "jc": sum_jc / float(test_episodes),
    }


def mean_ci95(values: List[float]) -> Tuple[float, float]:
    n = len(values)
    if n == 0:
        return float("nan"), float("nan")
    mean = float(np.mean(values))
    if n == 1:
        return mean, float("nan")
    std = float(np.std(values, ddof=1))
    se = std / math.sqrt(n)
    ci = 1.96 * se
    return mean, ci


def parse_seeds(s: str) -> List[int]:
    parts = [p.strip() for p in s.split(",") if p.strip()]
    return [int(p) for p in parts]


def find_latest_ckpt(ckpt_dir: str) -> str:
    if not os.path.isdir(ckpt_dir):
        raise FileNotFoundError(f"checkpoint dir not found: {ckpt_dir}")
    pts = [p for p in os.listdir(ckpt_dir) if p.endswith(".pth")]
    if not pts:
        raise FileNotFoundError(f"No checkpoint *.pth found in {ckpt_dir}")
    pts_sorted = sorted(pts, key=lambda n: os.path.getmtime(os.path.join(ckpt_dir, n)))
    return os.path.join(ckpt_dir, pts_sorted[-1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=str, default="0,1,2,3,4")
    ap.add_argument("--followers", type=int, default=1)
    ap.add_argument("--render", action="store_true")
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--ep-max", type=int, default=50000)
    ap.add_argument("--ep-len", type=int, default=1000)
    ap.add_argument("--save-every-ep", type=int, default=None)
    ap.add_argument("--test-episodes", type=int, default=100)
    ap.add_argument("--test-ep-len", type=int, default=1000)
    ap.add_argument("--output-root", type=str, default="./multiseed_runs_masac")
    ap.add_argument("--device", type=str, default="cpu")

    args = ap.parse_args()
    seeds = parse_seeds(args.seeds)

    os.makedirs(args.output_root, exist_ok=True)
    per_seed: Dict[int, Dict[str, float]] = {}

    for seed in seeds:
        run_dir = os.path.join(args.output_root, f"seed_{seed}")
        ckpt_dir = os.path.join(run_dir, "checkpoints")
        os.makedirs(ckpt_dir, exist_ok=True)

        if args.train:
            cfg = MASACConfig(
                seed=int(seed),
                n_agent=1,
                m_enemy=int(args.followers),
                render=False,
                ep_max=int(args.ep_max),
                ep_len=int(args.ep_len),
                save_every_ep=int(args.save_every_ep) if args.save_every_ep is not None else int(args.ep_max),
                device=str(args.device),
            )
            print(f"\n========== TRAIN (MASAC) seed={seed} ==========")
            ckpt_path = train_masac(cfg, ckpt_dir)
        else:
            ckpt_path = find_latest_ckpt(ckpt_dir)

        print(f"\n========== EVAL (MASAC) seed={seed} ==========")
        set_seed(int(seed))
        metrics = evaluate_checkpoint(
            ckpt_path=ckpt_path,
            followers=int(args.followers),
            test_episodes=int(args.test_episodes),
            ep_len=int(args.test_ep_len),
            render=bool(args.render),
        )
        per_seed[int(seed)] = metrics

        with open(os.path.join(run_dir, f"metrics_seed_{seed}.json"), "w", encoding="utf-8") as f:
            json.dump(
                {
                    "seed": int(seed),
                    "ckpt_path": ckpt_path,
                    "followers": int(args.followers),
                    "test_episodes": int(args.test_episodes),
                    "test_ep_len": int(args.test_ep_len),
                    "metrics": metrics,
                },
                f,
                ensure_ascii=False,
                indent=2,
            )

        print(
            f"seed={seed}  MCR={metrics['mcr']:.4f}  FKR={metrics['fkr']:.4f}  "
            f"JT={metrics['jt']:.2f}  JS={metrics['js']:.3f}  JC={metrics['jc']:.3f}"
        )

    keys = ["mcr", "fkr", "jt", "js", "jc"]
    rows = []
    for k in keys:
        vals = [per_seed[s][k] for s in seeds]
        mean, ci = mean_ci95(vals)
        std = float(np.std(vals, ddof=1)) if len(vals) > 1 else float("nan")
        rows.append((k, mean, ci, std, len(vals)))

    print("\n========== SUMMARY (across seeds) ==========")
    name_map = {
        "mcr": "任务完成率(MCR)",
        "fkr": "编队保持率(FKR)",
        "jt": "平均飞行时间(JT, steps)",
        "js": "平均飞行路程(JS, proxy)",
        "jc": "平均能量消耗(JC, proxy)",
    }
    for k, mean, ci, std, n in rows:
        label = name_map.get(k, k)
        if math.isnan(ci):
            print(f"{label}: mean={mean:.6f}  (n={n})")
        else:
            print(f"{label}: mean={mean:.6f} ± {ci:.6f} (95% CI), std={std:.6f} (n={n})")

    csv_path = os.path.join(args.output_root, "summary_across_seeds.csv")
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("metric,mean,ci95_half_width,std,n\n")
        for k, mean, ci, std, n in rows:
            f.write(f"{k},{mean},{ci},{std},{n}\n")
    print(f"\nSaved summary -> {csv_path}")


if __name__ == "__main__":
    main()
