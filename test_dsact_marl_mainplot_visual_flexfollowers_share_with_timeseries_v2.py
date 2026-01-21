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

# --- plotting (for episode time-series curves) ---
import matplotlib
if os.environ.get("DISPLAY", "") == "":
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset


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


def plot_episode_timeseries(t_arr,
                            leader_head_deg, follower_head_deg,
                            leader_vel, follower_vel,
                            dist_m,
                            save_path: str,
                            zoom_len: int = 30):
    """Plot heading angle / velocity / distance vs time for one test episode.

    Notes:
      - heading: degrees
      - velocity: real speed (scaled back)
      - distance: meters (scaled back)

    This is for visualization only; it does NOT affect training/testing results.
    """
    t_arr = np.asarray(t_arr)

    fig = plt.figure(figsize=(14, 4.2))

    # 1) Heading angle (top, wide)
    ax0 = plt.subplot2grid((2, 2), (0, 0), colspan=2)
    ax0.plot(t_arr, leader_head_deg, label="Leader heading angle")
    ax0.plot(t_arr, follower_head_deg, "--", label="Follower heading angle")
    ax0.set_ylabel("Heading angle (deg)")
    ax0.set_xlabel("Time")
    ax0.grid(True, linestyle="--", alpha=0.5)
    ax0.legend(loc="upper right")

    # Inset zoom (optional): default zooms into the last `zoom_len` steps
    if zoom_len and zoom_len > 0 and len(t_arr) > max(10, zoom_len):
        axins = inset_axes(ax0, width="38%", height="55%", loc="upper center")
        z0 = max(0, len(t_arr) - int(zoom_len))
        z1 = len(t_arr) - 1
        axins.plot(t_arr, leader_head_deg)
        axins.plot(t_arr, follower_head_deg, "--")
        axins.set_xlim(t_arr[z0], t_arr[z1])

        y_min = min(float(np.min(leader_head_deg[z0:z1 + 1])),
                    float(np.min(follower_head_deg[z0:z1 + 1])))
        y_max = max(float(np.max(leader_head_deg[z0:z1 + 1])),
                    float(np.max(follower_head_deg[z0:z1 + 1])))
        pad = 0.05 * (y_max - y_min + 1e-6)
        axins.set_ylim(y_min - pad, y_max + pad)
        axins.grid(True, linestyle="--", alpha=0.4)
        mark_inset(ax0, axins, loc1=2, loc2=4, fc="none", ec="0.3")

    # 2) Velocity (bottom-left)
    ax1 = plt.subplot2grid((2, 2), (1, 0))
    ax1.plot(t_arr, leader_vel, label="Leader velocity")
    ax1.plot(t_arr, follower_vel, "--", label="Follower velocity")
    ax1.set_ylabel("Velocity")
    ax1.set_xlabel("Time")
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(loc="lower right")

    # 3) Distance (bottom-right)
    ax2 = plt.subplot2grid((2, 2), (1, 1))
    ax2.plot(t_arr, dist_m, label="Distance between leader and follower")
    ax2.set_ylabel("Distance (m)")
    ax2.set_xlabel("Time")
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(loc="upper right")

    plt.tight_layout()
    plt.savefig(save_path, dpi=200)

    # Avoid blocking in headless environments
    if matplotlib.get_backend().lower() != "agg":
        plt.show()
    plt.close(fig)


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
    parser.add_argument("--share_follower_policy", action="store_true", help="If checkpoint has fewer actors than env needs, reuse the same follower policy for all followers (e.g., actor[1]).")
    parser.add_argument("--ep_len", type=int, default=1000, help="Max steps per episode (EP_LEN)")
    parser.add_argument("--no_render", action="store_true", help="Disable visualization")
    parser.add_argument("--fps", type=float, default=60.0, help="Render FPS limit (for smoother viewing); set 0 for no limit")
    # --- episode time-series plotting (visualization only) ---
    parser.add_argument("--plot_ep", type=int, default=-1,
                        help="Which test episode to plot (-1 means last episode).")
    parser.add_argument("--plot_follower", type=int, default=1,
                        help="Follower index to plot (1..num_agents-1). Leader is 0.")
    parser.add_argument("--plot_save", type=str, default="timeseries_ep.png",
                        help="Path to save the plotted figure.")
    parser.add_argument("--plot_zoom", type=int, default=30,
                        help="Zoom window length for heading inset (steps). 0 disables inset.")
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
    follow_used = args.followers if args.followers and args.followers > 0 else int(cfg.get("M_ENEMY", 1))
    print(f"Testing with 1 leader + {follow_used} followers. Render={RENDER}", flush=True)
    if args.share_follower_policy:
        print("share_follower_policy=ON (followers will reuse the same policy if ckpt actors are insufficient).", flush=True)

    # Env params: enforce max 1 leader; followers adjustable 1-4
    N_Agent = 1
    ckpt_followers = int(cfg.get("M_ENEMY", 1))
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

    # Load actors (supports optional sharing follower policy)
    if "actors" not in ckpt:
        raise KeyError("Checkpoint missing key 'actors'.")
    ckpt_actor_n = len(ckpt["actors"])
    if ckpt_actor_n < 1:
        raise ValueError("Checkpoint has no actors.")
    if ckpt_actor_n < num_agents and (not args.share_follower_policy):
        raise ValueError(
            f"Checkpoint has {ckpt_actor_n} actors, but env needs {num_agents} (1 leader + {M_Enemy} followers). "
            f"Re-train with M_ENEMY={M_Enemy} or run with --share_follower_policy."
        )

    actors = []
    for i in range(num_agents):
        a = train_mod.SACActor(obs_dim, act_dim, max_action, lr=3e-4, device=device)
        # Choose which checkpoint actor to load
        if i < ckpt_actor_n:
            load_idx = i
        else:
            # Share follower policy when ckpt has fewer actors than env requires
            # Prefer ckpt actor[1] as follower template; fallback to actor[0].
            load_idx = 1 if ckpt_actor_n > 1 else 0
        a.net.load_state_dict(ckpt["actors"][load_idx])
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

        # --- timeseries logging (only for one chosen episode) ---
        plot_ep = args.plot_ep if args.plot_ep >= 0 else (args.episodes - 1)
        do_log = (ep == plot_ep)
        fol_idx = int(args.plot_follower)
        if fol_idx < 1 or fol_idx >= num_agents:
            fol_idx = 1  # fallback to follower 1

        t_log = []
        leader_head_log = []
        follower_head_log = []
        leader_vel_log = []
        follower_vel_log = []
        dist_log = []

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

            if do_log:
                # 观测缩放：pos是 /1000，speed是 /30，heading是 (deg/360)
                # env obs: [x/1000, y/1000, speed/30, theta*57.3/360, ...]
                leader_pos_m = new_state[0, 0:2] * 1000.0
                fol_pos_m = new_state[fol_idx, 0:2] * 1000.0
                dist_m = float(np.linalg.norm(leader_pos_m - fol_pos_m))

                leader_vel = float(new_state[0, 2] * 30.0)
                fol_vel = float(new_state[fol_idx, 2] * 30.0)

                leader_head_deg = float(new_state[0, 3] * 360.0)
                fol_head_deg = float(new_state[fol_idx, 3] * 360.0)

                t_log.append(t)
                leader_head_log.append(leader_head_deg)
                follower_head_log.append(fol_head_deg)
                leader_vel_log.append(leader_vel)
                follower_vel_log.append(fol_vel)
                dist_log.append(dist_m)

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

        # Plot/save time-series for the chosen episode
        if do_log and len(t_log) >= 2:
            # ensure parent directory exists (if any)
            _dir = os.path.dirname(args.plot_save)
            if _dir:
                os.makedirs(_dir, exist_ok=True)
            plot_episode_timeseries(
                t_log,
                leader_head_log, follower_head_log,
                leader_vel_log, follower_vel_log,
                dist_log,
                save_path=args.plot_save,
                zoom_len=args.plot_zoom,
            )
            print(f"[PLOT] Saved time-series figure to: {os.path.abspath(args.plot_save)}")

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
