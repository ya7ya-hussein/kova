# Copyright (c) 2025, KOVA Project.
# SPDX-License-Identifier: BSD-3-Clause

"""KOVA MDP functions. Re-exports standard Isaac Lab mdp helpers plus our own."""

# Standard Isaac Lab MDP helpers (time_out, action_rate_l2, base_lin_vel, ...)
from isaaclab.envs.mdp import *  # noqa: F401, F403

# KOVA-specific terms
from .actions import DifferentialDriveAction, DifferentialDriveActionCfg
from .observations import (
    CoverageMapObs,
    action_history_obs,
    lidar_obs,
    nearest_uncovered_distance,
)
from .rewards import (
    blocking_penalty,
    collision_penalty,
    completion_bonus,
    direction_change_penalty,
    distance_guidance_reward,
    new_cell_reward,
    step_penalty,
    total_variation_reward,
)
from .terminations import collision_termination, coverage_complete, no_progress
from .events import randomize_room, CURRICULUM_TABLE, MAX_OBSTACLES
from .action_masking import apply_dijkstra_escape
