# Copyright (c) 2026, KOVA Project.
# SPDX-License-Identifier: BSD-3-Clause

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

    return env.coverage_map.steps_since_moved >= max_steps_without_moving


def collision_termination(
    env: "ManagerBasedRLEnv",
    force_threshold: float = 0.5,
    startup_grace_steps: int = 30,
    sensor_cfg: SceneEntityCfg | None = None,
) -> torch.Tensor:
    """Failure on contact force exceeding the threshold """
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

    return env.coverage_map.coverage_pct() >= coverage_threshold


def robot_out_of_bounds(
    env: "ManagerBasedRLEnv",
    max_height: float = 1.5,
    max_xy_from_origin: float = 30.0,
) -> torch.Tensor:

    robot = env.scene["robot"]
    pos_w = robot.data.root_pos_w
    origins = env.scene.env_origins  
    local = pos_w - origins

    non_finite = ~torch.isfinite(pos_w).all(dim=-1)
    too_high = local[:, 2] > max_height
    too_far = torch.linalg.norm(local[:, :2], dim=-1) > max_xy_from_origin

    return non_finite | too_high | too_far