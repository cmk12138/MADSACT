# -*- coding: utf-8 -*-
"""
run_ablation_suite_dsact.py

一键跑“消融实验（ablation）” + “多随机种子 + 95%CI”。

默认设置（用户要求的默认）：
  - seeds: 0,1,2,3,4
  - followers: 1   (1 leader + 1 follower)
  - test_episodes: 100
  - ep_len: 1000
  - variants:
      full
      no_td_bound
      no_uncertainty_weight
      no_action_penalty
      no_progress_reward
      no_formation_bonus

它会自动：
  1) 为每个 variant 生成一份“打补丁”的源码副本（不会污染你的原始文件）
  2) 对每个 seed：重新训练（可关）-> 自动选最新 checkpoint -> 测试 -> 保存 metrics.json
  3) 汇总每个 variant 的 mean ± 95%CI（跨 seeds），保存到 CSV

放置方式：
  把本文件放到与你的代码同一目录（应当能找到）：
    - marl_dsact_path_env_3_smooth.py
    - path_env.py

运行示例：
  # 快速验证（少训练点）
  python run_ablation_suite_dsact.py --train --ep-max 2000 --test-episodes 10 --seeds 0,1

  # 论文口径（训练很慢，按需调整）
  python run_ablation_suite_dsact.py --train --seeds 0,1,2,3,4 --test-episodes 100

  # 只评测已有 ckpt（不重新训练）
  python run_ablation_suite_dsact.py --no-train

输出目录：
  ./ablation_runs/
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


# -----------------------------
# Helpers: CI, ckpt selection
# -----------------------------
def mean_ci95(values: List[float]) -> Tuple[float, float]:
    """Return (mean, half_width_of_95CI). Uses normal approx 1.96 * std/sqrt(n)."""
    n = len(values)
    if n == 0:
        return float("nan"), float("nan")
    if n == 1:
        return float(values[0]), 0.0
    arr = np.asarray(values, dtype=np.float64)
    mean = float(arr.mean())
    std = float(arr.std(ddof=1))
    ci = 1.96 * std / math.sqrt(n)
    return mean, float(ci)


def pick_latest_ckpt(save_dir: Path) -> Path | None:
    """Pick newest *.pt in save_dir (by mtime)."""
    pts = sorted(save_dir.glob("*.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
    return pts[0] if pts else None


# -----------------------------
# Patching source files
# -----------------------------
def read_text(p: Path) -> str:
    return p.read_text(encoding="utf-8", errors="ignore")


def write_text(p: Path, s: str) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(s, encoding="utf-8")


def patch_train_no_uncertainty_weight(train_src: str) -> str:
    """
    Replace DSAC-T uncertainty weighting ratio1/ratio2 with 1.
    We only patch the specific ratio lines; everything else stays untouched.
    """
    # Keep indentation
    pattern1 = r"(?m)^(?P<indent>\s*)ratio1\s*=\s*\(\(\s*critic\.mean_std1\.detach\(\)\s*\*\*\s*2\s*\)\s*/\s*\(q1_s_det\.pow\(2\)\s*\+\s*cfg\.STD_BIAS\)\)\.clamp\(0\.1,\s*10\.0\)\s*$"
    pattern2 = r"(?m)^(?P<indent>\s*)ratio2\s*=\s*\(\(\s*critic\.mean_std2\.detach\(\)\s*\*\*\s*2\s*\)\s*/\s*\(q2_s_det\.pow\(2\)\s*\+\s*cfg\.STD_BIAS\)\)\.clamp\(0\.1,\s*10\.0\)\s*$"

    def repl1(m: re.Match) -> str:
        ind = m.group("indent")
        return f"{ind}ratio1 = torch.ones_like(q1_m)"

    def repl2(m: re.Match) -> str:
        ind = m.group("indent")
        return f"{ind}ratio2 = torch.ones_like(q2_m)"

    out = re.sub(pattern1, repl1, train_src)
    out2 = re.sub(pattern2, repl2, out)

    # Sanity check: ensure replacement happened at least once
    if out2 == train_src:
        # fallback: looser patterns
        out2 = re.sub(r"(?m)^\s*ratio1\s*=.*STD_BIAS.*clamp\(0\.1,\s*10\.0\).*$",
                      "                    ratio1 = torch.ones_like(q1_m)", out2)
        out2 = re.sub(r"(?m)^\s*ratio2\s*=.*STD_BIAS.*clamp\(0\.1,\s*10\.0\).*$",
                      "                    ratio2 = torch.ones_like(q2_m)", out2)

    return out2


def patch_env_reward(env_src: str, *, k_progress=None, formation_bonus=None, w_act=None) -> str:
    """Patch reward shaping scalars inside path_env.py (step())."""
    out = env_src
    if k_progress is not None:
        out = re.sub(r"(?m)^\s*k_progress\s*=\s*[-+0-9.eE]+\s*#",
                     f"        k_progress = {float(k_progress):.10g}         #", out)
    if formation_bonus is not None:
        out = re.sub(r"(?m)^\s*formation_bonus\s*=\s*[-+0-9.eE]+\s*#",
                     f"        formation_bonus = {float(formation_bonus):.10g}    #", out)
    if w_act is not None:
        out = re.sub(r"(?m)^\s*w_act\s*=\s*[-+0-9.eE]+\s*#",
                     f"        w_act = {float(w_act):.10g}              #", out)
    return out


def prepare_variant_src(
    variant: str,
    base_dir: Path,
    out_src_dir: Path,
) -> None:
    """
    Create patched source copies under out_src_dir:
      - marl_dsact_path_env_3_smooth.py
      - path_env.py
    """
    out_src_dir.mkdir(parents=True, exist_ok=True)

    train_in = base_dir / "marl_dsact_path_env_3_smooth.py"
    env_in = base_dir / "path_env.py"

    if not train_in.exists():
        raise FileNotFoundError(f"Missing {train_in}. Put this runner next to your training script.")
    if not env_in.exists():
        raise FileNotFoundError(f"Missing {env_in}. Put this runner next to your environment file.")

    train_src = read_text(train_in)
    env_src = read_text(env_in)

    # Apply variant-specific patches
    if variant == "no_uncertainty_weight":
        train_src = patch_train_no_uncertainty_weight(train_src)

    if variant == "no_action_penalty":
        env_src = patch_env_reward(env_src, w_act=0.0)
    elif variant == "no_progress_reward":
        env_src = patch_env_reward(env_src, k_progress=0.0)
    elif variant == "no_formation_bonus":
        env_src = patch_env_reward(env_src, formation_bonus=0.0)

    # Write out
    write_text(out_src_dir / "marl_dsact_path_env_3_smooth.py", train_src)
    write_text(out_src_dir / "path_env.py", env_src)

    # Also copy anything else in the folder that might be required (optional)
    # (Usually not needed; training script imports only path_env.)
    return




def cfg_snapshot(cfg) -> Dict:
    """Config class is not a dataclass; collect uppercase fields."""
    out: Dict = {}
    for k in dir(cfg):
        if not k.isupper():
            continue
        try:
            v = getattr(cfg, k)
        except Exception:
            continue
        if callable(v):
            continue
        out[k] = v
    return out

# -----------------------------
# Worker: train + eval one run
# -----------------------------
def worker_run(
    src_dir: Path,
    seed: int,
    variant: str,
    followers: int,
    ep_max: int,
    ep_len: int,
    test_episodes: int,
    test_ep_len: int,
    do_train: bool,
    seed_out_dir: Path,
    share_follower_policy: bool = True,
) -> Dict[str, float]:
    """
    Run one (variant, seed): optional train then eval.
    """
    # Import patched module from src_dir
    sys.path.insert(0, str(src_dir))
    import importlib

    train_mod = importlib.import_module("marl_dsact_path_env_3_smooth")

    # Seed everything (python/numpy/torch/cuda)
    train_mod.set_seed(int(seed))

    cfg = train_mod.Config()
    cfg.SEED = int(seed)
    cfg.N_AGENT = 1
    cfg.M_ENEMY = int(followers)
    cfg.EP_MAX = int(ep_max)
    cfg.EP_LEN = int(ep_len)
    cfg.RENDER = False

    ckpt_dir = seed_out_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    cfg.SAVE_DIR = str(ckpt_dir)

    # Variant override via cfg (no need to patch file)
    if variant == "no_td_bound":
        cfg.TD_BOUND_SCALE = 0.0

    if do_train:
        train_mod.train(cfg)

    ckpt_path = pick_latest_ckpt(ckpt_dir)
    if ckpt_path is None or (not ckpt_path.exists()):
        raise FileNotFoundError(f"No checkpoint found in {ckpt_dir}. (Variant={variant}, seed={seed})")

    metrics = evaluate_checkpoint_local(
        train_mod=train_mod,
        ckpt_path=ckpt_path,
        followers=int(followers),
        test_episodes=int(test_episodes),
        ep_len=int(test_ep_len),
        share_follower_policy=share_follower_policy,
    )

    # Save metrics + config snapshot
    payload = {
        "variant": variant,
        "seed": int(seed),
        "followers": int(followers),
        "ckpt_path": str(ckpt_path),
        "metrics": metrics,
        "cfg_used": cfg_snapshot(cfg),
    }
    write_text(seed_out_dir / "metrics.json", json.dumps(payload, ensure_ascii=False, indent=2))
    return metrics


# -----------------------------
# Local evaluation (no pygame)
# -----------------------------
def _build_actors_from_ckpt(train_mod, ckpt: Dict, obs_dim: int, act_dim: int, max_action: float, device) -> List:
    actors = []
    ckpt_actor_n = len(ckpt["actors"])
    num_agents = 1 + int(ckpt.get("cfg", {}).get("M_ENEMY", 1))

    # We'll build exactly (1 + followers) actors for evaluation; map ckpt actors accordingly.
    # If ckpt has fewer follower actors, we can share one follower policy if share_follower_policy=True.
    for i in range(num_agents):
        a = train_mod.SACActor(obs_dim, act_dim, max_action, lr=1e-6, device=device)
        if i < ckpt_actor_n:
            load_idx = i
        else:
            # share follower policy
            load_idx = 1 if ckpt_actor_n > 1 else 0
        a.net.load_state_dict(ckpt["actors"][load_idx])
        a.net.eval()
        actors.append(a)
    return actors


def evaluate_checkpoint_local(
    train_mod,
    ckpt_path: Path,
    followers: int,
    test_episodes: int,
    ep_len: int,
    share_follower_policy: bool = True,
) -> Dict[str, float]:
    """
    Metric logic follows your existing visual test script:
      - MCR: win_times / episodes
      - FKR: team_counter / steps_taken
      - JT: steps_taken
      - JS: sum of leader speed state[0][2] (proxy)
      - JC: sum(abs(leader_action))
    """
    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(str(ckpt_path), map_location=device)

    if "actors" not in ckpt:
        raise KeyError("Checkpoint missing key 'actors'.")

    env = train_mod.make_env(1, followers, render=False)
    obs = np.asarray(env.reset(), dtype=np.float32)
    num_agents = 1 + int(followers)
    obs_dim = int(obs.shape[-1])
    act_dim = int(env.action_space.shape[0])
    max_action = float(env.action_space.high[0])

    # build actors: use ckpt cfg if available for number of follower policies
    # Here we always evaluate 1+followers agents.
    actors = []
    ckpt_actor_n = len(ckpt["actors"])
    for i in range(num_agents):
        a = train_mod.SACActor(obs_dim, act_dim, max_action, lr=1e-6, device=device)
        if i < ckpt_actor_n:
            load_idx = i
        else:
            if not share_follower_policy:
                raise ValueError(f"Checkpoint has {ckpt_actor_n} actors < needed {num_agents}.")
            load_idx = 1 if ckpt_actor_n > 1 else 0
        a.net.load_state_dict(ckpt["actors"][load_idx])
        a.net.eval()
        actors.append(a)

    win_times = 0.0
    sum_fkr = 0.0
    sum_jt = 0.0
    sum_js = 0.0
    sum_jc = 0.0

    for _ in range(int(test_episodes)):
        state = np.asarray(env.reset(), dtype=np.float32)
        integral_V = 0.0
        integral_U = 0.0
        steps_taken = 0
        last_team_counter = 0.0
        win = 0.0

        for t in range(int(ep_len)):
            actions = []
            for i in range(num_agents):
                ai = actors[i].act(state[i], deterministic=True)
                actions.append(ai)
            action = np.asarray(actions, dtype=np.float32)

            step_out = env.step(action)
            obs2, r, done, win, team_counter, extra = train_mod.unwrap_step(step_out)

            steps_taken += 1
            # proxies matching your test script
            integral_V += float(state[0][2])
            integral_U += float(np.sum(np.abs(action[0])))

            state = np.asarray(obs2, dtype=np.float32)
            last_team_counter = float(team_counter)

            if bool(done):
                break

        fkr = (last_team_counter / max(1, steps_taken))
        sum_fkr += float(fkr)
        sum_jt += float(steps_taken)
        sum_js += float(integral_V)
        sum_jc += float(integral_U)
        if bool(win):
            win_times += 1.0

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


# -----------------------------
# Master orchestration
# -----------------------------
DEFAULT_VARIANTS = [
    "full",
    "no_td_bound",
    "no_uncertainty_weight",
    "no_action_penalty",
    "no_progress_reward",
    "no_formation_bonus",
]


def run_master(args: argparse.Namespace) -> None:
    base_dir = Path(__file__).resolve().parent
    root = Path(args.out_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)

    seeds = [int(s) for s in args.seeds.split(",") if s.strip() != ""]
    variants = DEFAULT_VARIANTS if not args.variants else [v.strip() for v in args.variants.split(",") if v.strip()]
    do_train = bool(args.train)

    all_rows = []

    for variant in variants:
        variant_dir = root / variant
        src_dir = variant_dir / "_patched_src"
        if args.force_patch or (not (src_dir / "marl_dsact_path_env_3_smooth.py").exists()):
            print(f"[patch] Preparing patched source for variant={variant} -> {src_dir}", flush=True)
            prepare_variant_src(variant, base_dir, src_dir)

        for seed in seeds:
            seed_dir = variant_dir / f"seed_{seed}"
            seed_dir.mkdir(parents=True, exist_ok=True)

            cmd = [
                sys.executable,
                str(Path(__file__).resolve()),
                "--worker",
                "--src", str(src_dir),
                "--variant", variant,
                "--seed", str(seed),
                "--followers", str(args.followers),
                "--ep-max", str(args.ep_max),
                "--ep-len", str(args.ep_len),
                "--test-episodes", str(args.test_episodes),
                "--test-ep-len", str(args.test_ep_len),
                "--seed-out", str(seed_dir),
            ]
            if do_train:
                cmd.append("--train")

            print(f"[run] variant={variant} seed={seed} train={do_train}", flush=True)
            proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            (seed_dir / "worker.log").write_text(proc.stdout, encoding="utf-8", errors="ignore")
            if proc.returncode != 0:
                print(proc.stdout)
                raise RuntimeError(f"Worker failed for variant={variant}, seed={seed}. See {seed_dir/'worker.log'}")

            metrics_path = seed_dir / "metrics.json"
            payload = json.loads(metrics_path.read_text(encoding="utf-8"))
            metrics = payload["metrics"]
            all_rows.append({
                "variant": variant,
                "seed": seed,
                **metrics,
                "ckpt_path": payload.get("ckpt_path", ""),
            })

    # Save raw
    import pandas as pd
    raw_df = pd.DataFrame(all_rows)
    raw_csv = root / "ablation_raw.csv"
    raw_df.to_csv(raw_csv, index=False, encoding="utf-8-sig")

    # Summary per variant
    summary_rows = []
    for variant in variants:
        dfv = raw_df[raw_df["variant"] == variant]
        row = {"variant": variant, "n_seeds": len(dfv)}
        for k in ["mcr", "fkr", "jt", "js", "jc"]:
            vals = dfv[k].astype(float).tolist()
            mean, ci = mean_ci95(vals)
            row[f"{k}_mean"] = mean
            row[f"{k}_ci95"] = ci
        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    summary_csv = root / "summary_across_variants.csv"
    summary_df.to_csv(summary_csv, index=False, encoding="utf-8-sig")

    print("\n=== Ablation Summary (mean ± 95%CI across seeds) ===")
    for _, r in summary_df.iterrows():
        print(f"\n[{r['variant']}]  n={int(r['n_seeds'])}")
        for k, name in [("mcr","MCR"), ("fkr","FKR"), ("jt","JT"), ("js","JS"), ("jc","JC")]:
            print(f"  {name}: {r[f'{k}_mean']:.6f} ± {r[f'{k}_ci95']:.6f}")

    print(f"\nSaved raw -> {raw_csv}")
    print(f"Saved summary -> {summary_csv}")


def run_worker(args: argparse.Namespace) -> None:
    src_dir = Path(args.src).resolve()
    seed_out_dir = Path(args.seed_out).resolve()
    seed_out_dir.mkdir(parents=True, exist_ok=True)

    metrics = worker_run(
        src_dir=src_dir,
        seed=int(args.seed),
        variant=str(args.variant),
        followers=int(args.followers),
        ep_max=int(args.ep_max),
        ep_len=int(args.ep_len),
        test_episodes=int(args.test_episodes),
        test_ep_len=int(args.test_ep_len),
        do_train=bool(args.train),
        seed_out_dir=seed_out_dir,
        share_follower_policy=True,
    )
    print(json.dumps(metrics, ensure_ascii=False), flush=True)


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", type=str, default="./ablation_runs", help="Output root directory")
    p.add_argument("--seeds", type=str, default="0,1,2,3,4", help="Comma-separated seeds")
    p.add_argument("--followers", type=int, default=1, help="Number of followers (M_ENEMY)")
    p.add_argument("--variants", type=str, default="", help="Comma-separated variants (default: all)")
    p.add_argument("--force-patch", action="store_true", help="Regenerate patched sources even if exist")

    p.add_argument("--train", action="store_true", help="Run training for each (variant,seed)")
    p.add_argument("--no-train", action="store_true", help="Do NOT train; only evaluate existing ckpts (ignored if --train given)")

    p.add_argument("--ep-max", type=int, default=50000, help="Training EP_MAX")
    p.add_argument("--ep-len", type=int, default=1000, help="Training EP_LEN")
    p.add_argument("--test-episodes", type=int, default=100, help="Eval episodes")
    p.add_argument("--test-ep-len", type=int, default=1000, help="Eval max steps per episode")

    # worker-only args
    p.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--src", type=str, default="", help=argparse.SUPPRESS)
    p.add_argument("--variant", type=str, default="full", help=argparse.SUPPRESS)
    p.add_argument("--seed", type=int, default=0, help=argparse.SUPPRESS)
    p.add_argument("--seed-out", type=str, default="", help=argparse.SUPPRESS)

    return p


def main():
    args = build_argparser().parse_args()
    # normalize train flag
    if args.no_train and (not args.train):
        args.train = False

    if args.worker:
        run_worker(args)
    else:
        run_master(args)


if __name__ == "__main__":
    main()
