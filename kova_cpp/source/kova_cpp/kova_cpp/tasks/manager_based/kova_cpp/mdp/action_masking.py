# Copyright (c) 2026, KOVA Project.
# SPDX-License-Identifier: BSD-3-Clause

"""Layer-2 deployment-time escape.

`apply_dijkstra_escape` rescues a deterministic policy that has fallen into an
endgame local optimum (a state-loop where it revisits the same observation and
stops covering new cells). It does nothing until an env has gone `stuck_steps`
outer steps without a new cell, then plans an obstacle-aware path to the nearest
reachable uncovered cell and steers the robot there, handing control straight
back to the policy as soon as a new cell is covered.

IMPORTANT - terminations: this escape can ONLY work if the deployment env has the
`no_progress` and `stuck_in_place` terminations DISABLED. Those are training-only
episode-shorteners; in deployment `stuck_in_place` (30 steps without translating)
fires while the escape is turning in place toward a frontier, killing the episode
before it can drive. play_eval.py disables them; do the same in any deploy script.

Frame convention (what broke the previous version):
    The coverage map stores cells as row = +y, col = +x, with
        world_x = origin_x + (col + 0.5) * cell_size
        world_y = origin_y + (row + 0.5) * cell_size
    Everything here is computed in that world frame and compared against the
    *world* robot yaw (`robot_yaw`) using atan2(dy, dx). No egocentric / row-sign
    conversion, so the commanded heading always matches reality.

Division of labour: the escape handles GLOBAL relocation (getting the robot from
a dead end to the vicinity of an uncovered region over a collision-safe path);
the policy's reactive control handles LOCAL coverage, including the cells right
up against the walls, once it has been relocated there.

DEPLOYMENT / play only. Do not call it inside the training loop.
"""

from __future__ import annotations

import math
from collections import deque

import numpy as np
import torch
import torch.nn.functional as F
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _wrap_to_pi(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def _dilate(mask_2d: torch.Tensor, radius: int) -> torch.Tensor:
    """Binary dilation of a [H, W] bool mask by a square kernel of the given radius."""
    if radius <= 0:
        return mask_2d
    k = 2 * radius + 1
    m = mask_2d.float().view(1, 1, *mask_2d.shape)
    out = F.max_pool2d(m, kernel_size=k, stride=1, padding=radius) > 0.5
    return out.view(*mask_2d.shape)


def _bfs_path(traversable: np.ndarray, goals: np.ndarray, start: tuple[int, int]):
    """8-connected shortest path from `start` to the nearest `goals` cell.

    Only cells in `traversable` may be entered (these have clearance from every
    obstacle/wall). `start` itself need NOT be traversable. Diagonal moves are
    blocked from cutting an obstacle corner. Returns [start, ..., goal] or None.
    """
    H, W = traversable.shape
    sr, sc = start
    if goals[sr, sc]:
        return [(sr, sc)]

    seen = np.zeros((H, W), dtype=bool)
    parent = np.full((H, W, 2), -1, dtype=np.int32)
    seen[sr, sc] = True
    dq = deque([(sr, sc)])
    nbrs = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]

    goal = None
    while dq:
        r, c = dq.popleft()
        if goals[r, c]:
            goal = (r, c)
            break
        for dr, dc in nbrs:
            nr, nc = r + dr, c + dc
            if not (0 <= nr < H and 0 <= nc < W):
                continue
            if seen[nr, nc] or not traversable[nr, nc]:
                continue
            if dr != 0 and dc != 0:
                if not (traversable[r + dr, c] and traversable[r, c + dc]):
                    continue
            seen[nr, nc] = True
            parent[nr, nc] = (r, c)
            dq.append((nr, nc))

    if goal is None:
        return None

    path = [goal]
    cur = goal
    while cur != (sr, sc):
        pr, pc = parent[cur[0], cur[1]]
        cur = (int(pr), int(pc))
        path.append(cur)
    path.reverse()
    return path


def apply_dijkstra_escape(
    env: "ManagerBasedRLEnv",
    actions: torch.Tensor,
    stuck_steps: int = 20,
    lookahead: int = 2,
    v_cmd: float = 0.6,
    w_gain: float = 1.2,
    align_tol: float = 0.5,
    clearance_cells: int | None = None,
) -> torch.Tensor:
    """Override actions for envs trapped in an endgame local optimum.

    Args:
        actions:          policy actions [N, 2] == (v_norm, w_norm), each in [-1, 1].
        stuck_steps:      outer steps with no new cell before the escape engages.
                          (Requires no_progress / stuck_in_place terminations OFF.)
        lookahead:        cells ahead along the path to aim at (small = tight corners).
        v_cmd:            forward speed (normalised) once aligned with the waypoint.
        w_gain:           proportional turn gain on heading error.
        align_tol:        rad; above this the robot turns in place instead of driving
                          (no arcing -> safe near walls).
        clearance_cells:  obstacle dilation in cells. Default = ceil(radius/cell) (=2,
                          ~0.3 m clear). Raise to 3 if the escape ever clips a wall;
                          lower toward 1 only if 95% proves unreachable.
    """
    cmap = env.coverage_map
    stuck = cmap.steps_since_new_cell >= int(stuck_steps)
    if not bool(stuck.any()):
        return actions

    if clearance_cells is None:
        clearance_cells = int(math.ceil(cmap.robot_radius / cmap.cell_size))
    sweep_cells = int(math.ceil(cmap.robot_radius / cmap.cell_size))

    out = actions.clone()
    stuck_ids = torch.nonzero(stuck, as_tuple=False).squeeze(-1).tolist()

    for e in stuck_ids:
        inflated = _dilate(cmap.obstacles[e], clearance_cells)
        traversable = cmap.free_mask[e] & ~inflated          # robot centre may stand here
        uncovered = cmap.free_mask[e] & ~cmap.visited[e]     # cells still needing coverage
        goals = _dilate(uncovered, sweep_cells) & traversable  # stand here -> sweep hits uncovered

        if not bool(goals.any()):
            continue  # nothing safely reachable left -> leave the policy action untouched

        trav_np = traversable.detach().cpu().numpy()
        goals_np = goals.detach().cpu().numpy()
        sr = int(cmap.robot_row[e].item())
        sc = int(cmap.robot_col[e].item())

        path = _bfs_path(trav_np, goals_np, (sr, sc))

        rx = float(cmap.robot_xy_world[e, 0].item())
        ry = float(cmap.robot_xy_world[e, 1].item())
        yaw = float(cmap.robot_yaw[e].item())

        # No reachable target, or already on a covering cell but stuck: rotate gently
        # in place so the observation changes (safe now that stuck_in_place is off).
        if path is None or len(path) < 2:
            out[e, 0] = 0.0
            out[e, 1] = 0.5
            continue

        wp_r, wp_c = path[min(lookahead, len(path) - 1)]
        wp_x = cmap.origin_x + (wp_c + 0.5) * cmap.cell_size
        wp_y = cmap.origin_y + (wp_r + 0.5) * cmap.cell_size

        err = _wrap_to_pi(math.atan2(wp_y - ry, wp_x - rx) - yaw)

        if abs(err) > align_tol:
            v_norm = 0.0
            w_norm = math.copysign(max(0.4, w_gain * abs(err) / math.pi), err)
        else:
            v_norm = v_cmd
            w_norm = w_gain * err / math.pi

        out[e, 0] = float(max(-1.0, min(1.0, v_norm)))
        out[e, 1] = float(max(-1.0, min(1.0, w_norm)))

    return out