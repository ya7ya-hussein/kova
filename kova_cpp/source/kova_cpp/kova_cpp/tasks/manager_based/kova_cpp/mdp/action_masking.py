# Copyright (c) 2026, KOVA Project.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math
import torch
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def apply_dijkstra_escape(
    env: "ManagerBasedRLEnv",
    actions: torch.Tensor,
    stuck_threshold: int = 7,
    v_cmd: float = 0.5,
    w_gain: float = 1.0,
) -> torch.Tensor:
    cmap = env.coverage_map
    stuck = cmap.steps_since_new_cell >= stuck_threshold  
    if not stuck.any():
        return actions

    stuck_ids = stuck.nonzero(as_tuple=False).squeeze(-1)

    step_dirs = cmap.bfs_first_step_to_nearest_uncovered(stuck_ids)
    d_row = step_dirs[:, 0].float()
    d_col = step_dirs[:, 1].float()

    target_yaw = torch.atan2(d_row, d_col)  # [k]

    no_target = (d_row == 0) & (d_col == 0)

    current_yaw = cmap.robot_yaw[stuck_ids]
    err = _wrap_to_pi(target_yaw - current_yaw)  

    # Build escape actions
    w_norm = torch.clamp(w_gain * err / math.pi, -1.0, 1.0) 

    v_norm = v_cmd * torch.cos(err).clamp(min=0.0)

    new_actions = actions.clone()
    new_v = torch.where(no_target, new_actions[stuck_ids, 0], v_norm)
    new_w = torch.where(no_target, new_actions[stuck_ids, 1], w_norm)
    new_actions[stuck_ids, 0] = new_v
    new_actions[stuck_ids, 1] = new_w

    # Reset stuck counter for rescued envs to avoid override every step
    rescued = stuck_ids[~no_target]
    if rescued.numel() > 0:
        cmap.steps_since_new_cell[rescued] = 0
    return new_actions


def _wrap_to_pi(a: torch.Tensor) -> torch.Tensor:
    return (a + math.pi) % (2.0 * math.pi) - math.pi
