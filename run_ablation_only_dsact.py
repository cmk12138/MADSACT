# -*- coding: utf-8 -*-
"""run_ablation_only_dsact.py

Ablation ONLY runner (no multi-seed, no CI).

Place this file next to:
  - marl_dsact_path_env_3_smooth.py
  - path_env.py

Quick sanity run:
  python run_ablation_only_dsact.py --train --ep-max 2000 --test-episodes 10

Full (still single-seed) run:
  python run_ablation_only_dsact.py --train --ep-max 50000 --test-episodes 100

If you already trained and just want evaluation on existing ckpt folders:
  python run_ablation_only_dsact.py --no-train

Output:
  ./ablation_only_runs/
    <variant>/
      _patched_src/   # patched code copy (original files untouched)
      seed_<seed>/
        checkpoints/
        metrics.json
        worker.log
  ./ablation_only_runs/ablation_results.csv

Notes:
  - Your training script only saves checkpoints when (ep+1) % SAVE_EVERY_EP == 0.
    To guarantee there is at least one checkpoint, this runner forces SAVE_EVERY_EP = EP_MAX.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np


DEFAULT_VARIANTS = [
    "full",
    "no_td_bound",
    "no_uncertainty_weight",
    "no_action_penalty",
    "no_progress_reward",
    "no_formation_bonus",
]


# -----------------------------
# small utilities
# -----------------------------

def read_text(p: Path) -> str:
    return p.read_text(encoding="utf-8", errors="ignore")


def write_text(p: Path, s: str) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(s, encoding="utf-8")


def pick_latest_ckpt(save_dir: Path) -> Path | None:
    pts = sorted(save_dir.glob("*.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
    return pts[0] if pts else None


# -----------------------------
# patching
# -----------------------------

def patch_train_no_uncertainty_weight(train_src: str) -> str:
    """Force ratio1/ratio2 = 1 (turn off uncertainty weighting)."""

    def repl_ratio(line: str, which: str) -> str:
        # Keep indentation from original line
        m = re.match(r"^(\s*)", line)
        ind = m.group(1) if m else ""
        if which == "ratio1":
            return f"{ind}ratio1 = torch.ones_like(q1_m)"
        return f"{ind}ratio2 = torch.ones_like(q2_m)"

    out_lines = []
    for line in train_src.splitlines():
        if re.search(r"\bratio1\s*=.*clamp\(0\.1,\s*10\.0\)", line):
            out_lines.append(repl_ratio(line, "ratio1"))
        elif re.search(r"\bratio2\s*=.*clamp\(0\.1,\s*10\.0\)", line):
            out_lines.append(repl_ratio(line, "ratio2"))
        else:
            out_lines.append(line)
    return "\n".join(out_lines) + "\n"


def patch_env_reward(env_src: str, *, k_progress=None, formation_bonus=None, w_act=None) -> str:
    """Patch reward shaping scalars inside path_env.py (step())."""
    out = env_src
    if k_progress is not None:
        out = re.sub(
            r"(?m)^\s*k_progress\s*=\s*[-+0-9.eE]+\s*#",
            f"        k_progress = {float(k_progress):.10g}         #",
            out,
        )
    if formation_bonus is not None:
        out = re.sub(
            r"(?m)^\s*formation_bonus\s*=\s*[-+0-9.eE]+\s*#",
            f"        formation_bonus = {float(formation_bonus):.10g}    #",
            out,
        )
    if w_act is not None:
        out = re.sub(
            r"(?m)^\s*w_act\s*=\s*[-+0-9.eE]+\s*#",
            f"        w_act = {float(w_act):.10g}              #",
            out,
        )
    return out


def prepare_variant_src(variant: str, base_dir: Path, out_src_dir: Path) -> None:
    """Create patched source copies under out_src_dir (original files unchanged)."""
    out_src_dir.mkdir(parents=True, exist_ok=True)

    train_in = base_dir / "marl_dsact_path_env_3_smooth.py"
    env_in = base_dir / "path_env.py"
    if not train_in.exists():
        raise FileNotFoundError(f"Missing {train_in}. Put this runner next to your training script.")
    if not env_in.exists():
        raise FileNotFoundError(f"Missing {env_in}. Put this runner next to your environment file.")

    train_src = read_text(train_in)
    env_src = read_text(env_in)

    if variant == "no_uncertainty_weight":
        train_src = patch_train_no_uncertainty_weight(train_src)

    if variant == "no_action_penalty":
        env_src = patch_env_reward(env_src, w_act=0.0)
    elif variant == "no_progress_reward":
        env_src = patch_env_reward(env_src, k_progress=0.0)
    elif variant == "no_formation_bonus":
        env_src = patch_env_reward(env_src, formation_bonus=0.0)

    write_text(out_src_dir / "marl_dsact_path_env_3_smooth.py", train_src)
    write_text(out_src_dir / "path_env.py", env_src)


# -----------------------------
# local evaluation (no pygame)
# -----------------------------

def evaluate_checkpoint_local(train_mod, ckpt_path: Path, followers: int, test_episodes: int, ep_len: int) -> Dict[str, float]:
    """Evaluate using the same metric logic as your visual test script.

    - MCR: win_times / episodes
    - FKR: team_counter / steps_taken
    - JT : steps_taken
    - JS : sum of leader speed state[0][2]  (proxy, consistent with your current test)
    - JC : sum(abs(leader_action))          (proxy)
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

    # Build actors and load weights.
    actors = []
    ckpt_actor_n = len(ckpt["actors"])
    for i in range(num_agents):
        a = train_mod.SACActor(obs_dim, act_dim, max_action, lr=1e-6, device=device)
        load_idx = i if i < ckpt_actor_n else (1 if ckpt_actor_n > 1 else 0)  # share follower policy if needed
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

        for _t in range(int(ep_len)):
            actions = []
            for i in range(num_agents):
                actions.append(actors[i].act(state[i], deterministic=True))
            action = np.asarray(actions, dtype=np.float32)

            step_out = env.step(action)
            obs2, r, done, win, team_counter, extra = train_mod.unwrap_step(step_out)

            steps_taken += 1
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


# -----------------------------
# worker (per variant)
# -----------------------------

def worker_run(src_dir: Path, out_dir: Path, variant: str, seed: int, followers: int, ep_max: int, ep_len: int, test_episodes: int, test_ep_len: int, do_train: bool) -> Dict[str, float]:
    """Run one variant: optional train, then eval."""

    sys.path.insert(0, str(src_dir))
    import importlib

    train_mod = importlib.import_module("marl_dsact_path_env_3_smooth")

    # Set random seeds (torch/numpy/python)
    train_mod.set_seed(int(seed))

    cfg = train_mod.Config()
    cfg.SEED = int(seed)
    cfg.N_AGENT = 1
    cfg.M_ENEMY = int(followers)
    cfg.EP_MAX = int(ep_max)
    cfg.EP_LEN = int(ep_len)
    cfg.RENDER = False

    # guarantee a checkpoint exists
    cfg.SAVE_EVERY_EP = int(ep_max)

    ckpt_dir = out_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    cfg.SAVE_DIR = str(ckpt_dir)

    if variant == "no_td_bound":
        cfg.TD_BOUND_SCALE = 0.0

    if do_train:
        train_mod.train(cfg)

    ckpt_path = pick_latest_ckpt(ckpt_dir)
    if ckpt_path is None or (not ckpt_path.exists()):
        raise FileNotFoundError(f"No checkpoint found in {ckpt_dir}. Did you forget --train? Variant={variant}")

    metrics = evaluate_checkpoint_local(train_mod, ckpt_path, followers=int(followers), test_episodes=int(test_episodes), ep_len=int(test_ep_len))

    payload = {
        "variant": variant,
        "seed": int(seed),
        "followers": int(followers),
        "ckpt_path": str(ckpt_path),
        "metrics": metrics,
    }
    write_text(out_dir / "metrics.json", json.dumps(payload, ensure_ascii=False, indent=2))
    return metrics


# -----------------------------
# master orchestration
# -----------------------------

def run_master(args: argparse.Namespace) -> None:
    base_dir = Path(__file__).resolve().parent
    root = Path(args.out_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)

    variants = DEFAULT_VARIANTS if not args.variants else [v.strip() for v in args.variants.split(",") if v.strip()]

    # Prepare output rows
    rows: List[Dict[str, object]] = []

    for variant in variants:
        variant_dir = root / variant
        src_dir = variant_dir / "_patched_src"
        if args.force_patch or (not (src_dir / "marl_dsact_path_env_3_smooth.py").exists()):
            print(f"[patch] variant={variant} -> {src_dir}", flush=True)
            prepare_variant_src(variant, base_dir, src_dir)

        out_dir = variant_dir / f"seed_{args.seed}"
        out_dir.mkdir(parents=True, exist_ok=True)

        cmd = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--worker",
            "--src",
            str(src_dir),
            "--out",
            str(out_dir),
            "--variant",
            variant,
            "--seed",
            str(args.seed),
            "--followers",
            str(args.followers),
            "--ep-max",
            str(args.ep_max),
            "--ep-len",
            str(args.ep_len),
            "--test-episodes",
            str(args.test_episodes),
            "--test-ep-len",
            str(args.test_ep_len),
        ]
        if args.train and (not args.no_train):
            cmd.append("--train")

        print(f"[run] variant={variant} seed={args.seed} train={args.train and (not args.no_train)}", flush=True)
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        write_text(out_dir / "worker.log", proc.stdout)
        if proc.returncode != 0:
            print(proc.stdout)
            raise RuntimeError(f"Worker failed for variant={variant}. See {out_dir/'worker.log'}")

        payload = json.loads(read_text(out_dir / "metrics.json"))
        m = payload["metrics"]
        row = {"variant": variant, "seed": int(args.seed), **m, "ckpt_path": payload.get("ckpt_path", "")}
        rows.append(row)

    # Save CSV summary
    try:
        import pandas as pd

        df = pd.DataFrame(rows)
        df_path = root / "ablation_results.csv"
        df.to_csv(df_path, index=False, encoding="utf-8-sig")
    except Exception:
        # Fallback if pandas not installed
        df_path = root / "ablation_results.csv"
        header = ["variant", "seed", "mcr", "fkr", "jt", "js", "jc", "ckpt_path"]
        lines = [",".join(header)]
        for r in rows:
            lines.append(",".join(
                [
                    str(r.get("variant", "")),
                    str(r.get("seed", "")),
                    f"{float(r.get('mcr', 0.0)):.8f}",
                    f"{float(r.get('fkr', 0.0)):.8f}",
                    f"{float(r.get('jt', 0.0)):.8f}",
                    f"{float(r.get('js', 0.0)):.8f}",
                    f"{float(r.get('jc', 0.0)):.8f}",
                    str(r.get("ckpt_path", "")),
                ]
            ))
        write_text(df_path, "\n".join(lines) + "\n")

    # Print nice summary
    print("\n=== Ablation results (single seed) ===")
    for r in rows:
        print(
            f"[{r['variant']}] MCR={r['mcr']:.6f}  FKR={r['fkr']:.6f}  JT={r['jt']:.2f}  JS={r['js']:.6f}  JC={r['jc']:.6f}"
        )

    print(f"\nSaved -> {df_path}")


def run_worker(args: argparse.Namespace) -> None:
    metrics = worker_run(
        src_dir=Path(args.src).resolve(),
        out_dir=Path(args.out).resolve(),
        variant=str(args.variant),
        seed=int(args.seed),
        followers=int(args.followers),
        ep_max=int(args.ep_max),
        ep_len=int(args.ep_len),
        test_episodes=int(args.test_episodes),
        test_ep_len=int(args.test_ep_len),
        do_train=bool(args.train),
    )
    print(json.dumps(metrics, ensure_ascii=False))


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", type=str, default="./ablation_only_runs", help="Output root")
    p.add_argument("--seed", type=int, default=42, help="Single seed for ablation")
    p.add_argument("--followers", type=int, default=1, help="Number of followers (M_ENEMY)")
    p.add_argument("--variants", type=str, default="", help="Comma-separated variants (default: all)")
    p.add_argument("--force-patch", action="store_true", help="Regenerate patched source copies")

    p.add_argument("--train", action="store_true", help="Train each variant before evaluating")
    p.add_argument("--no-train", action="store_true", help="Do NOT train; only evaluate existing ckpts")

    p.add_argument("--ep-max", type=int, default=50000, help="Training EP_MAX")
    p.add_argument("--ep-len", type=int, default=1000, help="Training EP_LEN")
    p.add_argument("--test-episodes", type=int, default=100, help="Eval episodes")
    p.add_argument("--test-ep-len", type=int, default=1000, help="Eval max steps per episode")

    # worker args
    p.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--src", type=str, default="", help=argparse.SUPPRESS)
    p.add_argument("--out", type=str, default="", help=argparse.SUPPRESS)
    p.add_argument("--variant", type=str, default="full", help=argparse.SUPPRESS)

    return p


def main() -> None:
    args = build_argparser().parse_args()

    if args.worker:
        run_worker(args)
    else:
        # If user didn't explicitly ask train, keep default False.
        # But ablation is usually with retraining, so we set train=True unless --no-train.
        # User can still call with --no-train.
        if (not args.train) and (not args.no_train):
            args.train = True
        run_master(args)


if __name__ == "__main__":
    main()
