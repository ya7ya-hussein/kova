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
