# -*- coding: utf-8 -*-
"""
DSAC-T (multi-agent) TEST script for path_env.py (RlGame)
Compatible with training script: marl_dsact_path_env_mainplot.py

Metrics (same as main_SAC_dsact_test.py):
- 任务完成率 (win_times / TEST_EPISODE)
- 平均最大编队保持率 (average_FKR / TEST_EPISODE), FKR = team_counter / timestep
- 平均最短飞行时间 (average_timestep / TEST_EPISODE)
- 平均最短飞行路程 (average_integral_V / TEST_EPISODE), integral_V += state[0][2]
- 平均最小能量损耗 (average_integral_U / TEST_EPISODE), integral_U += abs(action[0]).sum()

Visualization:
- Enable with --render (calls env.render() each step)

Run:
    python test_dsact_marl_mainplot.py --ckpt ./checkpoints_path_dsact/dsact_path_ep50.pt --episodes 100 --render
or:
    python test_dsact_marl_mainplot.py --save_dir ./checkpoints_path_dsact --episodes 100 --render
"""

import os
import glob
import argparse
import numpy as np
import torch
from matplotlib import pyplot as plt  # kept for parity; not required for metrics
import importlib
from rl_env.path_env import RlGame

N_Agent=1
M_Enemy=1
RENDER=True
env = RlGame(n=N_Agent,m=M_Enemy,render=RENDER).unwrapped


def unwrap_step(step_out):
    """Accept env.step() returning 5 or 6 values."""
    if isinstance(step_out, (list, tuple)) and len(step_out) >= 5:
        obs2, r, done, win, team_counter = step_out[:5]
        extra = step_out[5:] if len(step_out) > 5 else ()
        return obs2, r, done, win, team_counter, extra
    raise RuntimeError(f"Unexpected env.step() return: {type(step_out)}")


def pick_latest_ckpt(save_dir: str):
    cands = sorted(glob.glob(os.path.join(save_dir, "*.pt")))
    return cands[-1] if cands else None


@torch.no_grad()
def choose_action_deterministic(actor, obs_1agent: np.ndarray):
    """
    Deterministic policy: use actor mean output directly.
    In marl_dsact_path_env_mainplot.ActorNet, mean is already squashed into [-max_action, max_action].
    """
    device = actor.device
    obs_t = torch.as_tensor(obs_1agent, dtype=torch.float32, device=device).unsqueeze(0)
    mean, std = actor.net(obs_t)
    a = torch.clamp(mean, -actor.max_action, actor.max_action)
    return a.squeeze(0).cpu().numpy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, default="", help="Path to *.pt checkpoint saved by marl_dsact_path_env_mainplot.py")
    parser.add_argument("--save_dir", type=str, default="./checkpoints_path_dsact", help="Directory to search latest *.pt if --ckpt not provided")
    parser.add_argument("--episodes", type=int, default=100, help="Number of test episodes")
    parser.add_argument("--ep_len", type=int, default=1000, help="Max steps per episode")
    parser.add_argument("--render", action="store_true", help="Enable visualization (env.render())")
    args = parser.parse_args()

    # Import training module to reuse env builder + actor class (must match architectures)
    train_mod = None
    for name in ["marl_dsact_path_env_mainplot", "marl_dsact_path_env", "marl_dsact_path_env_plotfix"]:
        try:
            train_mod = importlib.import_module(name)
            break
        except Exception:
            pass
    if train_mod is None:
        raise ImportError("Cannot import training module. Put marl_dsact_path_env_mainplot.py in the same folder.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt_path = args.ckpt if args.ckpt else pick_latest_ckpt(args.save_dir)
    if not ckpt_path or not os.path.exists(ckpt_path):
        raise FileNotFoundError(
            f"Checkpoint not found. Provide --ckpt or put a *.pt in {os.path.abspath(args.save_dir)}"
        )

    print(f"Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device)
    cfg = ckpt.get("cfg", {})

    # Same env params as training (fallback to 1/1)
    N_Agent = int(cfg.get("N_AGENT", 1))
    M_Enemy = int(cfg.get("M_ENEMY", 1))
    RENDER = bool(args.render)

    env = train_mod.make_env(N_Agent, M_Enemy, render=RENDER)

    # Determine dimensions
    obs = np.asarray(env.reset(), dtype=np.float32)
    num_agents = obs.shape[0]
    obs_dim = obs.shape[-1]
    act_dim = int(env.action_space.shape[0])
    max_action = float(env.action_space.high[0])

    # Build actors and load weights
    actors = []
    for i in range(num_agents):
        a = train_mod.SACActor(obs_dim, act_dim, max_action, lr=3e-4, device=device)
        a.net.load_state_dict(ckpt["actors"][i])
        a.net.eval()
        actors.append(a)

    action = np.zeros((num_agents, act_dim), dtype=np.float32)

    win_times = 0
    average_FKR = 0.0
    average_timestep = 0.0
    average_integral_V = 0.0
    average_integral_U = 0.0

    all_ep_V, all_ep_U, all_ep_T, all_ep_F = [], [], [], []

    for ep in range(args.episodes):
        state = np.asarray(env.reset(), dtype=np.float32)

        total_rewards = 0.0
        integral_V = 0.0
        integral_U = 0.0

        last_team_counter = 0.0
        steps_taken = 0

        for t in range(args.ep_len):
            for i in range(num_agents):
                action[i] = choose_action_deterministic(actors[i], state[i])

            new_state, reward, done, win, team_counter, extra = unwrap_step(env.step(action))

            reward = np.asarray(reward, dtype=np.float32).reshape(num_agents, -1).squeeze(-1)
            new_state = np.asarray(new_state, dtype=np.float32)

            if win:
                win_times += 1

            # Metrics aligned with main_SAC_dsact_test.py
            integral_V += float(state[0][2])
            integral_U += float(np.abs(action[0]).sum())
            total_rewards += float(reward.mean())

            state = new_state
            last_team_counter = float(team_counter)
            steps_taken = t + 1

            if RENDER:
                try:
                    env.render()
                except Exception:
                    pass

            if done:
                break

        denom = max(1, steps_taken)
        FKR = last_team_counter / denom

        average_FKR += FKR
        average_timestep += steps_taken
        average_integral_V += integral_V
        average_integral_U += integral_U

        all_ep_V.append(integral_V)
        all_ep_U.append(integral_U)
        all_ep_T.append(steps_taken)
        all_ep_F.append(FKR)

        print(f"[TEST EP {ep:03d}] Score={total_rewards: .3f}  steps={steps_taken:4d}  FKR={FKR: .4f}")

    # Same final prints as main_SAC_dsact_test.py
    print('任务完成率', win_times / args.episodes)
    print('平均最大编队保持率', average_FKR / args.episodes)
    print('平均最短飞行时间', average_timestep / args.episodes)
    print('平均最短飞行路程', average_integral_V / args.episodes)
    print('平均最小能量损耗', average_integral_U / args.episodes)

    print("\n(Extra) mean±std over episodes:")
    print(f"  FKR: {np.mean(all_ep_F):.4f} ± {np.std(all_ep_F):.4f}")
    print(f"  Time(steps): {np.mean(all_ep_T):.2f} ± {np.std(all_ep_T):.2f}")
    print(f"  Distance(integral_V): {np.mean(all_ep_V):.3f} ± {np.std(all_ep_V):.3f}")
    print(f"  Energy(integral_U): {np.mean(all_ep_U):.3f} ± {np.std(all_ep_U):.3f}")

    try:
        env.close()
    except Exception:
        pass


if __name__ == "__main__":
    main()
