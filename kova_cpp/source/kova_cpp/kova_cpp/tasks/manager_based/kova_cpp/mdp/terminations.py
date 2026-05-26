# Copyright (c) 2025, KOVA Project.
# SPDX-License-Identifier: BSD-3-Clause

"""Termination functions for KOVA Coverage Path Planning."""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def no_progress(
    env: "ManagerBasedRLEnv",
    max_steps_without_new_cell: int = 500,
) -> torch.Tensor:
    """Failure if the robot has gone too long without covering a new cell."""
    return env.coverage_map.steps_since_new_cell >= max_steps_without_new_cell


def stuck_in_place(
    env: "ManagerBasedRLEnv",
    max_steps_without_moving: int = 30,
) -> torch.Tensor:
    """Failure if the robot is physically stuck (wall-hugging or frozen).

    Uses displacement tracking from the coverage map: counts consecutive steps
    where the robot moved less than half a cell. Catches the two observed
    failure modes (driving into a wall in a loop, or freezing in the center)
    even when no_progress would not fire because a cell was occasionally clipped.
    """
    return env.coverage_map.steps_since_moved >= max_steps_without_moving


def collision_termination(
    env: "ManagerBasedRLEnv",
    force_threshold: float = 0.5,
    startup_grace_steps: int = 30,
    sensor_cfg: SceneEntityCfg | None = None,
) -> torch.Tensor:
    """Failure on contact force exceeding the threshold (after grace period)."""
    sensor = env.scene[sensor_cfg.name]
    forces = sensor.data.net_forces_w  # [N, B, 3]
    if forces is None:
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    horizontal = torch.linalg.norm(forces[..., :2], dim=-1)
    any_hit = (horizontal > force_threshold).any(dim=-1)
    grace = env.episode_length_buf < startup_grace_steps
    return any_hit & ~grace


def coverage_complete(
    env: "ManagerBasedRLEnv",
    coverage_threshold: float = 0.98,
) -> torch.Tensor:
    """Optional success termination once near-total coverage is achieved.

    Disabled by default (the env keeps running so the policy can keep collecting
    completion bonuses or trailing reward). Provided here for convenience.
    """
    return env.coverage_map.coverage_pct() >= coverage_threshold


def robot_out_of_bounds(
    env: "ManagerBasedRLEnv",
    max_height: float = 1.5,
    max_xy_from_origin: float = 30.0,
) -> torch.Tensor:
    """Failure guard against physics instability.

    Catches the documented Isaac Lab failure where an unstable articulation
    "flies away" (position goes huge or NaN) and then vanishes from the scene,
    silently stalling training. By terminating these envs we force a clean
    reset before the bad state propagates into the coverage map / policy.

    Triggers when, relative to the env origin, the robot:
      * has a non-finite root position (NaN / inf), OR
      * is higher than ``max_height`` m (launched into the air), OR
      * is farther than ``max_xy_from_origin`` m horizontally (flung away).
    """
    robot = env.scene["robot"]
    pos_w = robot.data.root_pos_w  # [N, 3] world frame
    origins = env.scene.env_origins  # [N, 3]
    local = pos_w - origins

    non_finite = ~torch.isfinite(pos_w).all(dim=-1)
    too_high = local[:, 2] > max_height
    too_far = torch.linalg.norm(local[:, :2], dim=-1) > max_xy_from_origin

    return non_finite | too_high | too_far