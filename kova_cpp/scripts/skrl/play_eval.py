# Copyright (c) 2026, KOVA Project.
# SPDX-License-Identifier: BSD-3-Clause

"""Unattended evaluation of a KOVA policy.

Runs the deterministic policy (mean actions) with the Layer-2 escape active over
`--num_episodes` episodes, with NO real-time throttle (finishes fast, headless),
and writes:
    <run>/play_eval/kova_play_log_<ts>.csv      one row per episode
    <run>/play_eval/kova_play_summary_<ts>.csv  aggregate stats

Per-episode columns: termination reason, final coverage %, completed flag, cells
covered / total free, steps, duration (s), time remaining (s), budget (s),
collision wall/obstacle label, escape engagements + steps, no-progress counter.

DEPLOYMENT NOTE: the training terminations `no_progress` (150 steps w/o new cell)
and `stuck_in_place` (30 steps w/o moving) are DISABLED here. They are training
episode-shorteners; in deployment `stuck_in_place` fires while the escape is
turning in place toward a frontier and kills the episode before it can recover.
Collision / time_out / coverage_complete stay on (real deployment outcomes).

Example:
    python scripts/skrl/play_eval.py \
        --task Isaac-Coverage-KOVA-Play-v0 \
        --num_envs 1 --headless --num_episodes 20 \
        --checkpoint logs/skrl/kova_cpp/<run>/checkpoints/best_agent.pt
"""

import argparse
import csv
import math
import os
import sys
import time

from isaaclab.app import AppLauncher

# ---------------------------------------------------------------- CLI / launch
parser = argparse.ArgumentParser(description="Evaluate a KOVA policy over N episodes and log to CSV.")
parser.add_argument("--video", action="store_true", default=False, help="Record a video of the run.")
parser.add_argument("--video_length", type=int, default=400, help="Length of the recorded video (steps).")
parser.add_argument("--num_envs", type=int, default=1, help="Number of envs (keep 1 for clean per-episode logging).")
parser.add_argument("--task", type=str, default=None, help="Task name (use the -Play variant).")
parser.add_argument("--checkpoint", type=str, default=None, help="Path to the model checkpoint (e.g. best_agent.pt).")
parser.add_argument("--use_pretrained_checkpoint", action="store_true", default=False, help="Use a published checkpoint.")
parser.add_argument("--seed", type=int, default=None, help="Seed (default: random).")
parser.add_argument("--num_episodes", type=int, default=20, help="How many episodes to evaluate.")
parser.add_argument("--stuck_steps", type=int, default=20, help="No-progress steps before the escape engages.")
parser.add_argument("--real_time", action="store_true", default=False, help="Throttle to real-time (for watching).")
parser.add_argument("--keep_train_terminations", action="store_true", default=False,
                    help="Do NOT disable no_progress/stuck_in_place (debug only; the escape will be preempted).")
parser.add_argument("--ml_framework", type=str, default="torch", choices=["torch", "jax", "jax-numpy"])
parser.add_argument("--algorithm", type=str, default="PPO", choices=["AMP", "PPO", "IPPO", "MAPPO"])
parser.add_argument("--agent", type=str, default=None, help="Name of the RL agent configuration entry point.")
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
if args_cli.video:
    args_cli.enable_cameras = True
# clear out the hydra args so the rest of the script doesn't see them
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ------------------------------------------------------------------- imports
import random  # noqa: E402
import torch  # noqa: E402

import skrl  # noqa: E402
from packaging import version  # noqa: E402

SKRL_VERSION = "1.4.3"
if version.parse(skrl.__version__) < version.parse(SKRL_VERSION):
    skrl.logger.error(f"Unsupported skrl version: {skrl.__version__}. Install >= {SKRL_VERSION}.")
    exit()

if args_cli.ml_framework.startswith("torch"):
    from skrl.utils.runner.torch import Runner
elif args_cli.ml_framework.startswith("jax"):
    from skrl.utils.runner.jax import Runner

import gymnasium as gym  # noqa: E402
from isaaclab.envs import (  # noqa: E402
    DirectMARLEnv,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.dict import print_dict  # noqa: E402
from isaaclab.utils.pretrained_checkpoint import get_published_pretrained_checkpoint  # noqa: E402
from isaaclab_rl.skrl import SkrlVecEnvWrapper  # noqa: E402

import isaaclab_tasks  # noqa: F401, E402
from isaaclab_tasks.utils import get_checkpoint_path  # noqa: E402
from isaaclab_tasks.utils.hydra import hydra_task_config  # noqa: E402

import kova_cpp.tasks  # noqa: F401, E402
from kova_cpp.tasks.manager_based.kova_cpp.mdp.action_masking import apply_dijkstra_escape  # noqa: E402


# config shortcuts
if args_cli.agent is None:
    algorithm = args_cli.algorithm.lower()
    agent_cfg_entry_point = "skrl_cfg_entry_point" if algorithm in ["ppo"] else f"skrl_{algorithm}_cfg_entry_point"
else:
    agent_cfg_entry_point = args_cli.agent
    algorithm = agent_cfg_entry_point.split("_cfg")[0].split("skrl_")[-1].lower()


def _term_value(tm, name):
    """Return the [N] done buffer for termination term `name` from the last step."""
    try:
        return tm.get_term(name)
    except Exception:
        return tm._term_dones[name]


@hydra_task_config(args_cli.task, agent_cfg_entry_point)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg, experiment_cfg: dict):
    """Evaluate the policy and log to CSV."""
    task_name = args_cli.task.split(":")[-1]
    train_task_name = task_name.replace("-Play", "")

    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # --- DEPLOYMENT: disable the training-only "give up" terminations -------------------
    # no_progress (150 steps w/o new cell) and stuck_in_place (30 steps w/o moving) are
    # training episode-shorteners. In deployment they abort the episode the instant the
    # escape pauses to turn toward a frontier -- before it can drive the robot out. With
    # them on, the escape can never finish a rescue (this was the 0/20 completion bug).
    if not args_cli.keep_train_terminations:
        for _term in ("no_progress", "stuck_in_place"):
            if getattr(env_cfg.terminations, _term, None) is not None:
                setattr(env_cfg.terminations, _term, None)
                print(f"[INFO] Disabled termination for deployment: {_term}")

    if args_cli.ml_framework.startswith("jax"):
        skrl.config.jax.backend = "jax" if args_cli.ml_framework == "jax" else "numpy"

    if args_cli.seed == -1 or args_cli.seed is None:
        args_cli.seed = random.randint(0, 10000)
    experiment_cfg["seed"] = args_cli.seed
    env_cfg.seed = experiment_cfg["seed"]

    # resolve checkpoint
    log_root_path = os.path.abspath(os.path.join("logs", "skrl", experiment_cfg["agent"]["experiment"]["directory"]))
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    if args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("skrl", train_task_name)
        if not resume_path:
            print("[INFO] No pre-trained checkpoint available for this task.")
            return
    elif args_cli.checkpoint:
        resume_path = os.path.abspath(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(
            log_root_path, run_dir=f".*_{algorithm}_{args_cli.ml_framework}", other_dirs=["checkpoints"]
        )
    log_dir = os.path.dirname(os.path.dirname(resume_path))
    env_cfg.log_dir = log_dir

    # build env
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    if isinstance(env.unwrapped, DirectMARLEnv) and algorithm in ["ppo"]:
        env = multi_agent_to_single_agent(env)

    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play_eval"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording a video of the evaluation.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    env = SkrlVecEnvWrapper(env, ml_framework=args_cli.ml_framework)

    experiment_cfg["trainer"]["close_environment_at_exit"] = False
    experiment_cfg["agent"]["experiment"]["write_interval"] = 0
    experiment_cfg["agent"]["experiment"]["checkpoint_interval"] = 0
    runner = Runner(env, experiment_cfg)

    print(f"[INFO] Loading model checkpoint from: {resume_path}")
    runner.agent.load(resume_path)
    runner.agent.set_running_mode("eval")

    # ------------------------------------------------------------- eval setup
    base_env = env.unwrapped
    cmap = base_env.coverage_map
    tm = base_env.termination_manager
    max_steps = int(base_env.max_episode_length)
    step_dt = float(base_env.step_dt)

    completion_term = next((c for c in ("coverage_complete", "completion", "success") if c in tm.active_terms), None)

    NUM_EPISODES = int(args_cli.num_episodes)
    STUCK = int(args_cli.stuck_steps)

    obs, _ = env.reset()

    # room interior bounding box (geometry is fixed per level -> compute once)
    free0 = cmap.free_mask.reshape(cmap.num_envs, cmap.H, cmap.W)[0]
    ys, xs = torch.where(free0)
    if ys.numel() > 0:
        room_rmin, room_rmax = int(ys.min()), int(ys.max())
        room_cmin, room_cmax = int(xs.min()), int(xs.max())
    else:
        room_rmin = room_cmin = 0
        room_rmax, room_cmax = cmap.H - 1, cmap.W - 1
    wall_band = int(math.ceil(cmap.robot_radius / cmap.cell_size)) + 2

    rows = []
    ep = 0
    ep_len = 0
    peak_cov = 0.0
    last_visited = 0
    last_total = int(cmap.total_free_cells.reshape(-1)[0].item())
    last_rr = last_rc = 0
    last_ssnc = 0
    esc_steps = 0
    esc_engagements = 0
    prev_esc = False

    print(f"[INFO] Evaluating {NUM_EPISODES} episodes (stuck_steps={STUCK}, budget={max_steps} steps "
          f"= {max_steps * step_dt:.0f}s). Real-time={args_cli.real_time}.")

    while ep < NUM_EPISODES and simulation_app.is_running():
        t0 = time.time()
        with torch.inference_mode():
            # snapshot the ongoing episode's state (reflects the end of the previous step)
            cov = float(cmap.coverage_pct().reshape(-1)[0].item())
            last_ssnc = int(cmap.steps_since_new_cell.reshape(-1)[0].item())
            last_visited = int(cmap.visited_free_cells.reshape(-1)[0].item())
            last_total = int(cmap.total_free_cells.reshape(-1)[0].item())
            last_rr = int(cmap.robot_row.reshape(-1)[0].item())
            last_rc = int(cmap.robot_col.reshape(-1)[0].item())
            peak_cov = max(peak_cov, cov)

            esc_active = last_ssnc >= STUCK
            if esc_active:
                esc_steps += 1
                if not prev_esc:
                    esc_engagements += 1
            prev_esc = esc_active

            # deterministic policy + escape
            outputs = runner.agent.act(obs, timestep=0, timesteps=0)
            actions = outputs[-1].get("mean_actions", outputs[0])
            actions = apply_dijkstra_escape(base_env, actions, stuck_steps=STUCK)

            obs, _, terminated, truncated, _ = env.step(actions)
            ep_len += 1

            term_flat = terminated.reshape(-1)
            trunc_flat = truncated.reshape(-1)
            done = bool(term_flat[0].item()) or bool(trunc_flat[0].item())

        if done:
            fired = [n for n in tm.active_terms if bool(_term_value(tm, n).reshape(-1)[0].item())]
            if len(fired) > 1 and "time_out" in fired:
                fired = [n for n in fired if n != "time_out"]
            reason = fired[0] if fired else ("time_out" if bool(trunc_flat[0].item()) else "unknown")

            completed = completion_term is not None and reason == completion_term
            final_cov = max(peak_cov, 0.95) if completed else peak_cov

            collision_with = ""
            if "collision" in reason:
                at_edge = (
                    (last_rr - room_rmin <= wall_band) or (room_rmax - last_rr <= wall_band)
                    or (last_rc - room_cmin <= wall_band) or (room_cmax - last_rc <= wall_band)
                )
                collision_with = "wall" if at_edge else "obstacle"

            rows.append(
                {
                    "episode": ep + 1,
                    "terminated_by": reason,
                    "completed": int(completed),
                    "final_coverage_pct": round(100.0 * final_cov, 2),
                    "cells_covered": last_visited,
                    "total_free_cells": last_total,
                    "steps": ep_len,
                    "duration_s": round(ep_len * step_dt, 1),
                    "time_remaining_s": round((max_steps - ep_len) * step_dt, 1),
                    "budget_s": round(max_steps * step_dt, 1),
                    "collision_with": collision_with,
                    "escape_engagements": esc_engagements,
                    "escape_steps": esc_steps,
                    "no_progress_steps_at_end": last_ssnc,
                }
            )
            print(
                f"[ep {ep + 1:>2}/{NUM_EPISODES}] {reason:<16} "
                f"cov={100.0 * final_cov:5.1f}%  steps={ep_len:>5}  "
                f"t_left={(max_steps - ep_len) * step_dt:5.1f}s  "
                f"escapes={esc_engagements}  hit={collision_with or '-'}"
            )

            ep += 1
            ep_len = 0
            peak_cov = 0.0
            esc_steps = 0
            esc_engagements = 0
            prev_esc = False

        if args_cli.real_time:
            sleep = step_dt - (time.time() - t0)
            if sleep > 0:
                time.sleep(sleep)

    # --------------------------------------------------------------- write CSV
    out_dir = os.path.join(log_dir, "play_eval")
    os.makedirs(out_dir, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")

    fieldnames = [
        "episode", "terminated_by", "completed", "final_coverage_pct",
        "cells_covered", "total_free_cells", "steps", "duration_s",
        "time_remaining_s", "budget_s", "collision_with",
        "escape_engagements", "escape_steps", "no_progress_steps_at_end",
    ]
    log_path = os.path.join(out_dir, f"kova_play_log_{stamp}.csv")
    with open(log_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    # summary
    n = len(rows)
    n_complete = sum(r["completed"] for r in rows)
    covs = [r["final_coverage_pct"] for r in rows]
    noncomplete = [r["final_coverage_pct"] for r in rows if not r["completed"]]
    reason_hist: dict[str, int] = {}
    for r in rows:
        reason_hist[r["terminated_by"]] = reason_hist.get(r["terminated_by"], 0) + 1

    summary = [
        ("episodes", n),
        ("completion_rate", f"{n_complete}/{n}" if n else "0/0"),
        ("completion_pct", round(100.0 * n_complete / n, 1) if n else 0.0),
        ("mean_final_coverage_pct", round(sum(covs) / n, 2) if n else 0.0),
        ("min_final_coverage_pct", min(covs) if covs else 0.0),
        ("max_final_coverage_pct", max(covs) if covs else 0.0),
        ("mean_coverage_noncompleting_pct", round(sum(noncomplete) / len(noncomplete), 2) if noncomplete else "n/a"),
        ("mean_escape_engagements", round(sum(r["escape_engagements"] for r in rows) / n, 2) if n else 0.0),
        ("mean_steps", round(sum(r["steps"] for r in rows) / n, 1) if n else 0.0),
    ]
    for reason, count in sorted(reason_hist.items()):
        summary.append((f"terminated_by[{reason}]", count))

    summary_path = os.path.join(out_dir, f"kova_play_summary_{stamp}.csv")
    with open(summary_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["metric", "value"])
        w.writerows(summary)

    print("\n================ SUMMARY ================")
    for k, v in summary:
        print(f"  {k:<34} {v}")
    print(f"\n[INFO] Per-episode log: {log_path}")
    print(f"[INFO] Summary:         {summary_path}")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()