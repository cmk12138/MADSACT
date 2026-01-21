# -*- coding: utf-8 -*-
"""
Test script (visual) for DSAC-T multi-agent policy trained by marl_dsact_path_env_mainplot.py

Goal: behave like main_SAC_dsact_test.py (supports 1 leader + 1~4 followers)
- Real-time visualization of UAV motion/trajectory via pygame (env.render()).
- Same performance metrics printed at the end:
  任务完成率、平均最大编队保持率、平均最短飞行时间、平均最短飞行路程、平均最小能量损耗

How to run:
  1) Put this file in the SAME folder as marl_dsact_path_env_mainplot.py
  2) Make sure you have a checkpoint *.pt in ./checkpoints_path_dsact/
  3) Run:
        python test_dsact_marl_mainplot_visual.py
     or specify:
        python test_dsact_marl_mainplot_visual.py --ckpt ./checkpoints_path_dsact/dsact_path_ep50.pt --episodes 100

Tips:
- By default, visualization is ON (RENDER=True) like main_SAC_dsact_test.py.
- If you want to turn it off:
        python test_dsact_marl_mainplot_visual.py --no_render
"""

import os
import glob
import time
import argparse
import importlib
import numpy as np
import torch


def unwrap_step(step_out):
    """
    path_env.py may return 5 or 6 values.
    training/test variants:
      (obs2, r, done, win, team_counter) or
      (obs2, r, done, win, team_counter, dis)
    """
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
    Deterministic evaluation: use actor mean action (no sampling).
    """
    device = actor.device
    obs_t = torch.as_tensor(obs_1agent, dtype=torch.float32, device=device).unsqueeze(0)
    mean, std = actor.net(obs_t)
    a = torch.clamp(mean, -actor.max_action, actor.max_action)
    return a.squeeze(0).cpu().numpy()


def import_train_module():
    """
    Reuse env builder + Actor definition from training script.
    """
    for name in ["marl_dsact_path_env_mainplot", "marl_dsact_path_env", "marl_dsact_path_env_plotfix"]:
        try:
            return importlib.import_module(name)
        except Exception:
            pass
    raise ImportError("Cannot import marl_dsact_path_env_mainplot.py. Put it in the same folder as this test file.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, default="", help="Path to *.pt checkpoint")
    parser.add_argument("--save_dir", type=str, default="./checkpoints_path_dsact", help="Directory to search latest *.pt if --ckpt not provided")
    parser.add_argument("--episodes", type=int, default=100, help="Number of test episodes (TEST_EPIOSDE)")
    parser.add_argument("--followers", type=int, default=0, help="Number of follower UAVs to test (1-4). 0 means use checkpoint config M_ENEMY.")
    parser.add_argument("--ep_len", type=int, default=1000, help="Max steps per episode (EP_LEN)")
    parser.add_argument("--no_render", action="store_true", help="Disable visualization")
    parser.add_argument("--fps", type=float, default=60.0, help="Render FPS limit (for smoother viewing); set 0 for no limit")
    args = parser.parse_args()

    # Like main_SAC_dsact_test.py: visualization ON by default
    RENDER = not args.no_render

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_mod = import_train_module()

    ckpt_path = args.ckpt if args.ckpt else pick_latest_ckpt(args.save_dir)
    if not ckpt_path or not os.path.exists(ckpt_path):
        raise FileNotFoundError(
            f"Checkpoint not found. Provide --ckpt or put a *.pt in {os.path.abspath(args.save_dir)}"
        )

    print(f"Loading checkpoint: {ckpt_path}", flush=True)
    ckpt = torch.load(ckpt_path, map_location=device)
    cfg = ckpt.get("cfg", {})
    follow_used = args.followers if args.followers and args.followers > 0 else int(cfg.get("M_ENEMY", 4))
    print(f"Testing with 1 leader + {follow_used} followers. Render={RENDER}", flush=True)

    # Env params: enforce max 1 leader; followers adjustable 1-4
    N_Agent = 1
    ckpt_followers = int(cfg.get("M_ENEMY", 4))
    M_Enemy = args.followers if args.followers and args.followers > 0 else ckpt_followers
    if M_Enemy < 1 or M_Enemy > 4:
        raise ValueError(f"--followers must be in [1,4], got {M_Enemy}")

    env = train_mod.make_env(N_Agent, M_Enemy, render=RENDER)

    # Determine dims
    state0 = np.asarray(env.reset(), dtype=np.float32)
    num_agents = state0.shape[0]
    obs_dim = state0.shape[-1]
    act_dim = int(env.action_space.shape[0])
    max_action = float(env.action_space.high[0])

    # Load actors (use only required leader+followers)
    if "actors" not in ckpt:
        raise KeyError("Checkpoint missing key 'actors'.")
    if len(ckpt["actors"]) < num_agents:
        raise ValueError(f"Checkpoint has {len(ckpt['actors'])} actors, but env needs {num_agents} (1 leader + {M_Enemy} followers).")
    actors = []
    for i in range(num_agents):
        a = train_mod.SACActor(obs_dim, act_dim, max_action, lr=3e-4, device=device)
        a.net.load_state_dict(ckpt["actors"][i])
        a.net.eval()
        actors.append(a)

    action = np.zeros((num_agents, act_dim), dtype=np.float32)

    # Metrics (same names as main_SAC_dsact_test.py)
    win_times = 0
    average_FKR = 0.0
    average_timestep = 0.0
    average_integral_V = 0.0
    average_integral_U = 0.0

    # Optional per-episode logs
    ep_FKR, ep_T, ep_V, ep_U = [], [], [], []

    for ep in range(args.episodes):
        state = np.asarray(env.reset(), dtype=np.float32)

        integral_V = 0.0
        integral_U = 0.0
        last_team_counter = 0.0
        steps_taken = 0

        for t in range(args.ep_len):
            # deterministic actions
            for i in range(num_agents):
                action[i] = choose_action_deterministic(actors[i], state[i])

            step_out = env.step(action)
            new_state, reward, done, win, team_counter, extra = unwrap_step(step_out)

            new_state = np.asarray(new_state, dtype=np.float32)

            # win count
            if win:
                win_times += 1

            # Metrics accumulation (follow main_SAC_dsact_test.py)
            integral_V += float(state[0][2])            # 飞行路程(积分速度分量)
            integral_U += float(np.abs(action[0]).sum()) # 能量损耗(动作绝对值和)

            state = new_state
            last_team_counter = float(team_counter)
            steps_taken = t + 1

            # Visualization (trajectory) – this is where you see UAV motion
            if RENDER:
                env.render()
                # limit FPS so you can see the trajectory clearly
                if args.fps and args.fps > 0:
                    time.sleep(1.0 / float(args.fps))

            if done:
                break

        denom = max(1, steps_taken)
        FKR = last_team_counter / denom

        average_FKR += FKR
        average_timestep += steps_taken
        average_integral_V += integral_V
        average_integral_U += integral_U

        ep_FKR.append(FKR)
        ep_T.append(steps_taken)
        ep_V.append(integral_V)
        ep_U.append(integral_U)

        print(f"[TEST EP {ep:03d}] steps={steps_taken:4d}  FKR={FKR:.4f}  V={integral_V:.3f}  U={integral_U:.3f}  win={bool(win)}")

    # Final report
    TEST_EPIOSDE = args.episodes
    print('任务完成率', win_times / TEST_EPIOSDE)
    print('平均最大编队保持率', average_FKR / TEST_EPIOSDE)
    print('平均最短飞行时间', average_timestep / TEST_EPIOSDE)
    print('平均最短飞行路程', average_integral_V / TEST_EPIOSDE)
    print('平均最小能量损耗', average_integral_U / TEST_EPIOSDE)

    # Extra stats (useful sanity checks)
    print("\n(Extra) mean±std over episodes:")
    print(f"  FKR: {np.mean(ep_FKR):.4f} ± {np.std(ep_FKR):.4f}")
    print(f"  Time: {np.mean(ep_T):.2f} ± {np.std(ep_T):.2f}")
    print(f"  Distance(V): {np.mean(ep_V):.3f} ± {np.std(ep_V):.3f}")
    print(f"  Energy(U): {np.mean(ep_U):.3f} ± {np.std(ep_U):.3f}")

    try:
        env.close()
    except Exception:
        pass


if __name__ == "__main__":
    main()
