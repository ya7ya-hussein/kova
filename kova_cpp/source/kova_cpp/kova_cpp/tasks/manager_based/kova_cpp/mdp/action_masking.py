# Copyright (c) 2025, KOVA Project.
# SPDX-License-Identifier: BSD-3-Clause

"""Action masking for KOVA.

Two layers:

* **Layer 1 — Blocking (reward shaping):** the policy's commanded heading is
  evaluated against the 4-cardinal validity vector from ``CoverageMap``; if it
  points at an already-covered / blocked direction while uncovered options
  exist, a penalty is paid. This is implemented in ``mdp/rewards.py`` as
  ``blocking_penalty`` and lives in the reward graph rather than the action graph.

* **Layer 2 — Dijkstra escape:** when an env's ``steps_since_new_cell ≥ 7`` we
  run a batched BFS over its coverage grid (see ``coverage_map.bfs_first_step_
  to_nearest_uncovered``) and override the policy action for that env for one
  step, steering toward the nearest uncovered cell. ``steps_since_new_cell`` is
  reset by the coverage-map update on the step that records a new cell.

The function ``apply_dijkstra_escape`` is intended to be called by the
training/play loop AFTER ``policy.act(obs)`` and BEFORE ``env.step(actions)``.
This keeps PPO internals untouched — the override is invisible to the
optimiser. See the docstring on that function for the integration recipe.
"""

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
    """Override ``actions`` for stuck envs with a heading toward the nearest
    uncovered cell (one step). Returns the modified action tensor.

    Args
    ----
    env: the manager-based env (with ``env.coverage_map`` attached).
    actions: [N, 2] normalised policy outputs (v_norm, w_norm).
    stuck_threshold: number of consecutive no-progress steps that triggers the
        override.
    v_cmd: normalised linear velocity to command during the escape (in [0, 1]).
    w_gain: gain that maps heading error (rad) to normalised angular velocity.

    Implementation
    --------------
    1. Identify envs where ``steps_since_new_cell >= stuck_threshold``.
    2. Run a batched BFS for those envs only (see ``CoverageMap.bfs_first_step``).
    3. Convert the (d_row, d_col) first-step direction into a world-frame yaw,
       then compute the heading error vs the robot's current yaw.
    4. Set the action to (v_cmd, clip(w_gain * heading_err, -1, 1)) for those envs.
    5. Reset ``steps_since_new_cell`` for the rescued envs so they're not
       repeatedly forced into escape every step.
    """
    cmap = env.coverage_map
    stuck = cmap.steps_since_new_cell >= stuck_threshold  # [N]
    if not stuck.any():
        return actions

    stuck_ids = stuck.nonzero(as_tuple=False).squeeze(-1)
    # First step directions in grid frame: [k, 2] = (d_row, d_col), each in {-1, 0, 1}
    step_dirs = cmap.bfs_first_step_to_nearest_uncovered(stuck_ids)
    d_row = step_dirs[:, 0].float()
    d_col = step_dirs[:, 1].float()

    # Grid frame: +row is +y, +col is +x  (matches CoverageMap)
    # World heading required to step toward the next cell:
    target_yaw = torch.atan2(d_row, d_col)  # [k]

    # Where no target was found (step==0), keep the policy's action — fall through
    no_target = (d_row == 0) & (d_col == 0)

    current_yaw = cmap.robot_yaw[stuck_ids]
    err = _wrap_to_pi(target_yaw - current_yaw)  # [k]

    # Build escape actions
    w_norm = torch.clamp(w_gain * err / math.pi, -1.0, 1.0)  # divide by pi so max err -> 1
    # Slow down when we need to turn sharply
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
