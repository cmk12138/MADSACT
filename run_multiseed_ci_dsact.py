# -*- coding: utf-8 -*-
"""
run_multiseed_ci_dsact.py

One-file runner for:
  1) multi-seed training (re-train for each seed)
  2) evaluation (Monte Carlo episodes per seed)
  3) mean ± 95% confidence interval (across seeds) for 5 metrics:
        MCR (任务完成率), FKR (编队保持率), JT (平均飞行时间/步数),
        JS (平均飞行路程 proxy), JC (平均能量损耗 proxy)

This script reuses your existing source code:
  - marl_dsact_path_env_3_smooth.py  (training + model definitions)
  - path_env.py                     (environment)

It is designed to be "drop-in runnable":
  python run_multiseed_ci_dsact.py

Tips:
- Training can take a long time. Start with a small --ep-max to verify the pipeline.
- For a paper-style report, use 3~10 seeds and 100 test episodes per seed.

Example:
  python run_multiseed_ci_dsact.py --seeds 0,1,2,3,4 --followers 2 --ep-max 5000 --test-episodes 100

Outputs:
  ./multiseed_runs/
    seed_0/  (checkpoints + metrics_seed_0.json)
    ...
    summary_across_seeds.csv
"""

import argparse
import json
import math
import os
import random
from dataclasses import asdict
from typing import Dict, List, Tuple

import numpy as np
import torch

# Import your training module (contains Config, train(), make_env(), SACActor, unwrap_step, etc.)

# import sys
# import rl_env.path_env_patched as _patched_env
# sys.modules["rl_env.path_env"] = _patched_env

#import marl_dsact_path_env_3_smooth as train_mod
import marl_dsact_path_env_3_smooth as train_mod

def set_all_seeds(seed: int) -> None:
    """Seed python/numpy/torch and (optionally) make CUDA a bit more deterministic."""
    train_mod.set_seed(seed)

    # Optional: make runs more repeatable on CUDA (may slow down a bit)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def try_env_seed(env, seed: int) -> None:
    """
    Best-effort seeding for env.
    Your env may or may not support these; we try common conventions without failing.
    """
    try:
        env.reset(seed=seed)
        return
    except Exception:
        pass
    try:
        env.seed(seed)
        return
    except Exception:
        pass
    # If env doesn't support explicit seeding, that's ok — you'll still get multi-seed variation
    # from network init/replay sampling, but test randomness will be less controlled.


@torch.no_grad()
def evaluate_checkpoint(
    ckpt_path: str,
    followers: int,
    test_episodes: int,
    ep_len: int,
    render: bool = False,
    share_follower_policy: bool = True,
) -> Dict[str, float]:
    """
    Evaluate a checkpoint using the SAME metric logic as your visual test script,
    but without pygame visualization.

    Returns dict with:
      mcr, fkr, jt, js, jc
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(ckpt_path, map_location=device)
    cfg_from_ckpt = ckpt.get("cfg", {}) or {}

    N_Agent = 1
    M_Enemy = int(followers)

    env = train_mod.make_env(N_Agent, M_Enemy, render=render)
    try_env_seed(env, seed=int(cfg_from_ckpt.get("SEED", 0)))

    state0 = np.asarray(env.reset(), dtype=np.float32)
    num_agents = state0.shape[0]
    obs_dim = state0.shape[-1]
    act_dim = int(env.action_space.shape[0])
    max_action = float(env.action_space.high[0])

    if "actors" not in ckpt:
        raise KeyError("Checkpoint missing key 'actors'.")

    ckpt_actor_n = len(ckpt["actors"])
    if ckpt_actor_n < 1:
        raise ValueError("Checkpoint has no actors.")

    if ckpt_actor_n < num_agents and (not share_follower_policy):
        raise ValueError(
            f"Checkpoint has {ckpt_actor_n} actors, but env needs {num_agents}. "
            f"Re-train with M_ENEMY={M_Enemy} or run with share_follower_policy."
        )

    # Build actors and load weights (followers can share a policy if needed)
    actors = []
    for i in range(num_agents):
        a = train_mod.SACActor(obs_dim, act_dim, max_action, lr=3e-4, device=device)
        if i < ckpt_actor_n:
            load_idx = i
        else:
            load_idx = 1 if ckpt_actor_n > 1 else 0
        a.net.load_state_dict(ckpt["actors"][load_idx])
        a.net.eval()
        actors.append(a)

    win_times = 0.0
    sum_fkr = 0.0
    sum_jt = 0.0
    sum_js = 0.0
    sum_jc = 0.0

    for ep in range(test_episodes):
        state = np.asarray(env.reset(), dtype=np.float32)
        action = np.zeros((num_agents, act_dim), dtype=np.float32)

        integral_V = 0.0
        integral_U = 0.0
        steps_taken = 0
        last_team_counter = 0.0

        done = False
        win = False

        for t in range(ep_len):
            for i in range(num_agents):
                action[i] = actors[i].act(state[i], deterministic=True)

            step_out = env.step(action)
            # Use the same robust unpacking logic as training script
            new_state, reward, done, win, team_counter, extra = train_mod.unwrap_step(step_out)

            new_state = np.asarray(new_state, dtype=np.float32)

            # --- Metrics proxy (same as your test script) ---
            # JS proxy: accumulate normalized speed (state[0][2]) for leader
            # JC proxy: accumulate L1 action magnitude for leader
            integral_V += float(state[0][2])
            integral_U += float(np.sum(np.abs(action[0])))

            steps_taken += 1
            state = new_state

            # team_counter can be cumulative; we convert to "this episode's count"
            # Your visual test computes FKR = team_counter / steps_taken (team_counter is cumulative count)
            # We'll follow that logic exactly.
            last_team_counter = float(team_counter)

            if done:
                break

        FKR = (last_team_counter / max(1, steps_taken))

        if bool(win):
            win_times += 1.0

        sum_fkr += float(FKR)
        sum_jt += float(steps_taken)
        sum_js += float(integral_V)
        sum_jc += float(integral_U)

    mcr = win_times / float(test_episodes)
    fkr = sum_fkr / float(test_episodes)
    jt = sum_jt / float(test_episodes)
    js = sum_js / float(test_episodes)
    jc = sum_jc / float(test_episodes)

    try:
        env.close()
    except Exception:
        pass

    return {"mcr": mcr, "fkr": fkr, "jt": jt, "js": js, "jc": jc}


def mean_ci95(values: List[float]) -> Tuple[float, float]:
    """
    Mean and 95% CI half-width using normal approximation:
      CI = 1.96 * std/sqrt(n)
    """
    n = len(values)
    if n == 0:
        return float("nan"), float("nan")
    mean = float(np.mean(values))
    if n == 1:
        return mean, float("nan")
    std = float(np.std(values, ddof=1))
    #95% 置信区间（CI）计算，标准误差（SE）= 标准差（STD）/ sqrt(n)
    se = std / math.sqrt(n)
    ci = 1.96 * se
    return mean, ci


def parse_seeds(s: str) -> List[int]:
    if not s:
        return [0, 1, 2, 3, 4]
    parts = [p.strip() for p in s.split(",") if p.strip()]
    return [int(p) for p in parts]


def find_latest_ckpt(save_dir: str) -> str:
    """
    Find latest checkpoint by episode number in filenames like dsact_path_epXXXX.pt.
    Fallback to most recently modified *.pt.
    """
    if not os.path.isdir(save_dir):
        raise FileNotFoundError(f"save_dir not found: {save_dir}")
    pts = [p for p in os.listdir(save_dir) if p.endswith(".pt")]
    if not pts:
        raise FileNotFoundError(f"No checkpoint *.pt found in {save_dir}")

    def ep_num(name: str) -> int:
        import re
        m = re.search(r"_ep(\d+)\.pt$", name)
        return int(m.group(1)) if m else -1

    pts_sorted = sorted(pts, key=lambda n: (ep_num(n), os.path.getmtime(os.path.join(save_dir, n))))
    return os.path.join(save_dir, pts_sorted[-1])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=str, default="0,1,2,3,4", help="Comma-separated seeds, e.g. 0,1,2,3,4")
    parser.add_argument("--followers", type=int, default=1, help="Number of followers (M_ENEMY) in [1,4]")
    parser.add_argument("--render", action="store_true", help="Render during evaluation (slow)")
    parser.add_argument("--share-follower-policy", action="store_true", help="Allow sharing follower policy if ckpt has fewer actors than env needs")

    # Training controls
    parser.add_argument("--train", action="store_true", help="Actually train per seed. If not set, only evaluate existing ckpts.")
    parser.add_argument("--ep-max", type=int, default=None, help="Override training EP_MAX")
    parser.add_argument("--ep-len", type=int, default=None, help="Override training EP_LEN")
    parser.add_argument("--save-every-ep", type=int, default=None, help="Override SAVE_EVERY_EP. If not set, saves at end of training.")

    # Evaluation controls
    parser.add_argument("--test-episodes", type=int, default=100, help="Monte Carlo test episodes per seed")
    parser.add_argument("--test-ep-len", type=int, default=1000, help="Max steps per test episode")
    parser.add_argument("--output-root", type=str, default="./multiseed_runs", help="Output root folder")

    args = parser.parse_args()

    seeds = parse_seeds(args.seeds)
    if args.followers < 1 or args.followers > 4:
        raise ValueError("--followers must be in [1,4]")

    os.makedirs(args.output_root, exist_ok=True)

    per_seed_metrics: Dict[int, Dict[str, float]] = {}

    for seed in seeds:
        run_dir = os.path.join(args.output_root, f"seed_{seed}")
        os.makedirs(run_dir, exist_ok=True)
        ckpt_dir = os.path.join(run_dir, "checkpoints")
        os.makedirs(ckpt_dir, exist_ok=True)

        ckpt_path = None

        if args.train:
            # --- Train ---
            print(f"\n========== TRAIN seed={seed} ==========")
            set_all_seeds(seed)

            cfg = train_mod.Config()  # default config from your training script
            cfg.SEED = int(seed)
            cfg.N_AGENT = 1
            cfg.M_ENEMY = int(args.followers)
            cfg.RENDER = False
            cfg.SAVE_DIR = ckpt_dir

            if args.ep_max is not None:
                cfg.EP_MAX = int(args.ep_max)
            if args.ep_len is not None:
                cfg.EP_LEN = int(args.ep_len)

            # Ensure we save a checkpoint at the end (or use user-provided)
            if args.save_every_ep is not None:
                cfg.SAVE_EVERY_EP = int(args.save_every_ep)
            else:
                cfg.SAVE_EVERY_EP = int(cfg.EP_MAX)  # save at the end

            # Run training
            train_mod.train(cfg)

            # Find latest checkpoint
            ckpt_path = find_latest_ckpt(ckpt_dir)
        else:
            # --- Only evaluate existing ckpt ---
            ckpt_path = find_latest_ckpt(ckpt_dir)

        print(f"\n========== EVAL seed={seed} ==========")
        # Seed evaluation randomness as well (for repeatability of test episodes)
        set_all_seeds(seed)

        metrics = evaluate_checkpoint(
            ckpt_path=ckpt_path,
            followers=args.followers,
            test_episodes=args.test_episodes,
            ep_len=args.test_ep_len,
            render=args.render,
            share_follower_policy=args.share_follower_policy,
        )
        per_seed_metrics[seed] = metrics

        # Save per-seed metrics
        with open(os.path.join(run_dir, f"metrics_seed_{seed}.json"), "w", encoding="utf-8") as f:
            json.dump(
                {
                    "seed": seed,
                    "ckpt_path": ckpt_path,
                    "followers": args.followers,
                    "test_episodes": args.test_episodes,
                    "test_ep_len": args.test_ep_len,
                    "metrics": metrics,
                },
                f,
                ensure_ascii=False,
                indent=2,
            )

        print(f"seed={seed}  MCR={metrics['mcr']:.4f}  FKR={metrics['fkr']:.4f}  JT={metrics['jt']:.2f}  JS={metrics['js']:.3f}  JC={metrics['jc']:.3f}")

    # --- Aggregate across seeds ---
    keys = ["mcr", "fkr", "jt", "js", "jc"]
    summary_rows = []
    for k in keys:
        vals = [per_seed_metrics[s][k] for s in seeds]
        mean, ci = mean_ci95(vals)
        summary_rows.append((k, mean, ci, float(np.std(vals, ddof=1)) if len(vals) > 1 else float("nan"), len(vals)))

    # Print summary
    print("\n========== SUMMARY (across seeds) ==========")
    name_map = {
        "mcr": "任务完成率(MCR)",
        "fkr": "编队保持率(FKR)",
        "jt": "平均飞行时间(JT, steps)",
        "js": "平均飞行路程(JS, proxy)",
        "jc": "平均能量消耗(JC, proxy)",
    }
    for k, mean, ci, std, n in summary_rows:
        label = name_map.get(k, k)
        if math.isnan(ci):
            print(f"{label}: mean={mean:.6f}  (n={n})")
        else:
            print(f"{label}: mean={mean:.6f} ± {ci:.6f} (95% CI), std={std:.6f} (n={n})")

    # Save CSV
    csv_path = os.path.join(args.output_root, "summary_across_seeds.csv")
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("metric,mean,ci95_half_width,std,n\n")
        for k, mean, ci, std, n in summary_rows:
            f.write(f"{k},{mean},{ci},{std},{n}\n")
    print(f"\nSaved summary -> {csv_path}")


if __name__ == "__main__":
    main()
