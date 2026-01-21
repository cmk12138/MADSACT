# -*- coding: utf-8 -*-
"""
Test / evaluation script for DSAC-T multi-agent policy trained in path_env.py (RlGame).

Usage (recommended):
    python test_marl_dsact_path_env.py --ckpt ./checkpoints_path_dsact/dsact_path_ep50.pt --episodes 10 --render

If you don't pass --ckpt, it will try to pick the latest *.pt in --save_dir.

What it does:
- Loads actor weights from checkpoint
- Runs episodes with deterministic actions (mean policy) in RlGame
- Prints per-episode leader/follower/total returns and averages

Notes:
- Leader is agent 0, follower is agent 1 (if exists)
- Compatible with env.step() returning 5 or 6 values.
"""

import argparse
import glob
import os
import importlib
from typing import Optional, Tuple

import numpy as np
import torch


def unwrap_step(step_out):
    # Compatible with both 5-return and 6-return variants in path_env.py
    if isinstance(step_out, (list, tuple)) and len(step_out) >= 5:
        obs2, r, done, win, team_counter = step_out[:5]
        extra = step_out[5:] if len(step_out) > 5 else ()
        return obs2, r, done, win, team_counter, extra
    raise RuntimeError(f"Unexpected env.step() return: {type(step_out)}")


def pick_latest_ckpt(save_dir: str) -> Optional[str]:
    cands = sorted(glob.glob(os.path.join(save_dir, "*.pt")))
    return cands[-1] if cands else None


def load_training_module():
    """
    Try to import whichever training file you are using.
    Put this test file in the SAME folder as your training script.
    """
    for name in ["marl_dsact_path_env", "marl_dsact_path_env_mainplot", "marl_dsact_path_env_plotfix"]:
        try:
            return importlib.import_module(name)
        except Exception:
            pass
    raise ImportError(
        "Cannot import training module. Ensure marl_dsact_path_env.py (or *_mainplot.py) is in the same directory."
    )


@torch.no_grad()
def run_episode(env, actors, ep_len: int, device: torch.device, render: bool):
    obs = env.reset()
    obs = np.asarray(obs, dtype=np.float32)
    num_agents = obs.shape[0]

    ep_ret = np.zeros((num_agents,), dtype=np.float32)

    for t in range(ep_len):
        actions = []
        for i in range(num_agents):
            # deterministic=True => use mean policy
            a = actors[i].act(obs[i], deterministic=True)
            actions.append(a)
        actions = np.asarray(actions, dtype=np.float32)

        step_out = env.step(actions)
        obs2, r, done, win, team_counter, _extra = unwrap_step(step_out)

        obs2 = np.asarray(obs2, dtype=np.float32)
        r = np.asarray(r, dtype=np.float32).reshape(num_agents, -1).squeeze(-1)

        ep_ret += r
        obs = obs2

        if render:
            try:
                env.render()
            except Exception:
                pass

        if done:
            break

    return ep_ret, t + 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, default="", help="Path to checkpoint .pt")
    parser.add_argument("--save_dir", type=str, default="./checkpoints_path_dsact", help="Directory to search ckpt")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--ep_len", type=int, default=1000)
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.ckpt:
        ckpt_path = args.ckpt
    else:
        ckpt_path = pick_latest_ckpt(args.save_dir)

    if not ckpt_path or not os.path.exists(ckpt_path):
        raise FileNotFoundError(
            f"Checkpoint not found. Provide --ckpt or put a *.pt in {os.path.abspath(args.save_dir)}"
        )

    print(f"Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device)

    # Import training module to reuse env + network definitions (must match architectures)
    m = load_training_module()

    # Pull config from checkpoint if present (so test uses same n/m)
    cfg_dict = ckpt.get("cfg", {})
    n_agent = int(cfg_dict.get("N_AGENT", 1))
    m_enemy = int(cfg_dict.get("M_ENEMY", 1))

    env = m.make_env(n_agent, m_enemy, render=args.render)

    # Determine dimensions from env
    obs0 = env.reset()
    obs0 = np.asarray(obs0, dtype=np.float32)
    num_agents = obs0.shape[0]
    obs_dim = obs0.shape[-1]
    act_dim = int(env.action_space.shape[0])
    max_action = float(env.action_space.high[0])

    # Build actors and load weights
    actors = []
    for i in range(num_agents):
        actor = m.SACActor(obs_dim, act_dim, max_action, lr=3e-4, device=device)
        actor.net.load_state_dict(ckpt["actors"][i])
        actor.net.eval()
        actors.append(actor)

    leader_returns = []
    follower_returns = []
    total_returns = []

    for ep in range(args.episodes):
        ep_ret, steps = run_episode(env, actors, args.ep_len, device, args.render)
        leader = float(ep_ret[0])
        follower = float(ep_ret[1]) if num_agents > 1 else 0.0
        total = leader + follower

        leader_returns.append(leader)
        follower_returns.append(follower)
        total_returns.append(total)

        print(f"[TEST EP {ep:03d}] steps={steps:4d}  leader={leader: .3f}  follower={follower: .3f}  total={total: .3f}")

    print("\nAverages over episodes:")
    print(f"  leader  mean={np.mean(leader_returns): .3f}  std={np.std(leader_returns): .3f}")
    if num_agents > 1:
        print(f"  follower mean={np.mean(follower_returns): .3f}  std={np.std(follower_returns): .3f}")
    print(f"  total   mean={np.mean(total_returns): .3f}  std={np.std(total_returns): .3f}")

    try:
        env.close()
    except Exception:
        pass


if __name__ == "__main__":
    main()
