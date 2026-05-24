# Copyright (c) 2025, KOVA Project.
# SPDX-License-Identifier: BSD-3-Clause

"""Domain randomisation events for KOVA.

Each ``reset`` event reshapes the scene per environment, sampling:
    * Room size (width × height in metres), per curriculum level.
    * Obstacle positions inside the room.
    * Robot start pose inside the room.

Walls and obstacle cubes in the scene are kinematic ``RigidObject`` instances —
we move them with ``write_root_pose_to_sim``. Walls are positioned just outside
the interior on all four sides; obstacles are sampled with constraints.

After updating the physical scene, ``CoverageMap.reset_room`` is invoked to
keep the internal obstacle/free masks consistent with what the robot will see.

Curriculum
----------
The room-size sampling range is selected by ``env.cfg.curriculum_level`` (an
attribute we attach in the env cfg's ``__post_init__``). Levels follow the spec.
"""

from __future__ import annotations

import math
import torch
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


# ----------------------------------------------------------------------------
# Curriculum table — (room_min_m, room_max_m, max_obstacles)
# Episode length is set in env_cfg.__post_init__ based on the same level.
# ----------------------------------------------------------------------------
CURRICULUM_TABLE = {
    1: (3.0,  4.0,  0),
    2: (4.0,  6.0,  1),
    3: (5.0,  8.0,  1),
    4: (6.0, 12.0,  2),
    5: (8.0, 16.0,  2),
    6: (10.0, 20.0, 2),
}

OBSTACLE_HALF_RANGE = (0.2, 0.5)   # half-extents in metres
MAX_OBSTACLES = 2                  # absolute upper bound on obstacle slots
WALL_THICKNESS = 0.2               # m
WALL_HEIGHT = 0.5                  # m


def _curriculum_for(env: "ManagerBasedRLEnv") -> tuple[float, float, int]:
    level = int(getattr(env.cfg, "curriculum_level", 1))
    return CURRICULUM_TABLE.get(level, CURRICULUM_TABLE[1])


def _u(low: float, high: float, n: int, device) -> torch.Tensor:
    return torch.rand(n, device=device) * (high - low) + low


# ----------------------------------------------------------------------------
# Main reset event: rebuilds the entire room and tells CoverageMap about it.
# ----------------------------------------------------------------------------


def randomize_room(env: "ManagerBasedRLEnv", env_ids: torch.Tensor) -> None:
    """Sample room size, re-position walls, place obstacles, set robot pose,
    then sync the CoverageMap free/obstacle masks.

    Note: this is a single fused event so that the CoverageMap is updated
    atomically with the physical scene change. We expose smaller helper events
    for special-case overrides if needed.
    """
    if env_ids.numel() == 0:
        return
    device = env.device
    n = env_ids.numel()

    room_min, room_max, max_obs = _curriculum_for(env)

    # 1) Sample room size per env (independent W/H)
    room_w = _u(room_min, room_max, n, device)
    room_h = _u(room_min, room_max, n, device)

    # 2) Move the four walls. Walls are env-local: each env has its own scene
    #    cloned by Isaac Lab's InteractiveScene at {ENV_REGEX_NS}/wall_*.
    _move_walls(env, env_ids, room_w, room_h)

    # 3) Sample obstacle positions. Constraints:
    #    - inside the room with ≥ 0.5 m clearance from any wall
    #    - ≥ 0.5 m between obstacles
    #    - ≥ 0.8 m from the chosen robot start position
    #
    #    We first sample a tentative robot start, then sample obstacles around
    #    it with rejection (rejection per env, but in batched form: a small
    #    rejection loop with max attempts).
    robot_xy = _sample_robot_start(room_w, room_h, device)  # [n, 2]

    obs_xy, obs_half = _sample_obstacles(
        room_w=room_w, room_h=room_h, robot_xy=robot_xy,
        n_obs=min(MAX_OBSTACLES, max_obs), device=device,
    )

    # 4) Set obstacle poses in the physical scene
    _move_obstacles(env, env_ids, obs_xy, obs_half)

    # 5) Set robot pose
    yaw = _u(-math.pi, math.pi, n, device)
    _set_robot_pose(env, env_ids, robot_xy, yaw)

    # 6) Sync CoverageMap (interior + obstacle masks)
    #    Build obstacles_world: [n, MAX_OBSTACLES, 4] = (cx, cy, hx, hy); NaN = inactive.
    obstacles_world = torch.full(
        (n, MAX_OBSTACLES, 4), float("nan"), device=device, dtype=torch.float32
    )
    if obs_xy is not None:
        k = obs_xy.shape[1]
        obstacles_world[:, :k, 0] = obs_xy[..., 0]
        obstacles_world[:, :k, 1] = obs_xy[..., 1]
        obstacles_world[:, :k, 2] = obs_half[..., 0]
        obstacles_world[:, :k, 3] = obs_half[..., 1]

    room_size = torch.stack([room_w, room_h], dim=-1)  # [n, 2]
    env.coverage_map.reset_room(env_ids, room_size=room_size, obstacles_world=obstacles_world)


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------


def _move_walls(env, env_ids, room_w, room_h):
    """Position the four walls just outside the interior on each side."""
    # Each wall is a long thin cuboid. Walls are kinematic RigidObjects in the scene.
    # Pos schema: walls cover (room_w + 2*WALL_THICKNESS) along the long axis.
    half_w = 0.5 * room_w
    half_h = 0.5 * room_h
    half_t = 0.5 * WALL_THICKNESS

    # North wall (+y side)
    pos_n = torch.zeros(env_ids.numel(), 3, device=env.device)
    pos_n[:, 1] = half_h + half_t
    pos_n[:, 2] = 0.5 * WALL_HEIGHT
    _write_pose(env, "wall_north", env_ids, pos_n)

    pos_s = torch.zeros_like(pos_n)
    pos_s[:, 1] = -(half_h + half_t)
    pos_s[:, 2] = 0.5 * WALL_HEIGHT
    _write_pose(env, "wall_south", env_ids, pos_s)

    pos_e = torch.zeros_like(pos_n)
    pos_e[:, 0] = half_w + half_t
    pos_e[:, 2] = 0.5 * WALL_HEIGHT
    _write_pose(env, "wall_east", env_ids, pos_e)

    pos_w_ = torch.zeros_like(pos_n)
    pos_w_[:, 0] = -(half_w + half_t)
    pos_w_[:, 2] = 0.5 * WALL_HEIGHT
    _write_pose(env, "wall_west", env_ids, pos_w_)


def _move_obstacles(env, env_ids, obs_xy, obs_half):
    """Move (or hide) obstacle cubes. ``obs_xy``: [n, k, 2], ``obs_half``: [n, k, 2].
    Slots with NaN in obs_xy are hidden by parking them under the floor.
    """
    n = env_ids.numel()
    if obs_xy is None:
        # Hide all obstacle slots
        for k in range(MAX_OBSTACLES):
            pos = torch.zeros(n, 3, device=env.device)
            pos[:, 2] = -10.0  # well underground
            _write_pose(env, f"obstacle_{k}", env_ids, pos)
        return

    k_max = obs_xy.shape[1]
    for k in range(MAX_OBSTACLES):
        if k >= k_max:
            # No data for this slot — hide it
            pos = torch.zeros(n, 3, device=env.device)
            pos[:, 2] = -10.0
            _write_pose(env, f"obstacle_{k}", env_ids, pos)
            continue
        x = obs_xy[:, k, 0]
        y = obs_xy[:, k, 1]
        active = ~torch.isnan(x)
        # Z = WALL_HEIGHT/2 for active, -10 for inactive (hidden)
        z = torch.where(active, torch.full_like(x, 0.5 * WALL_HEIGHT), torch.full_like(x, -10.0))
        pos = torch.stack([
            torch.where(active, x, torch.zeros_like(x)),
            torch.where(active, y, torch.zeros_like(y)),
            z,
        ], dim=-1)
        _write_pose(env, f"obstacle_{k}", env_ids, pos)


def _write_pose(env, asset_name: str, env_ids: torch.Tensor, pos: torch.Tensor):
    """Write env-local pose (XYZ + identity quat) for an asset to sim.

    Isaac Lab stores poses in world frame; ``InteractiveScene.env_origins``
    gives each env's origin. We add the origin so the pose is correct for each
    cloned env.
    """
    asset = env.scene[asset_name]
    origins = env.scene.env_origins[env_ids]  # [n, 3]
    world_pos = pos + origins
    quat = torch.zeros(env_ids.numel(), 4, device=env.device)
    quat[:, 0] = 1.0  # identity (w, x, y, z)
    pose = torch.cat([world_pos, quat], dim=-1)  # [n, 7]
    asset.write_root_pose_to_sim(pose, env_ids=env_ids)
    # Also zero velocities for kinematic objects (they're kinematic so this is a no-op
    # but doing it once is harmless and ensures clean state)
    vel = torch.zeros(env_ids.numel(), 6, device=env.device)
    asset.write_root_velocity_to_sim(vel, env_ids=env_ids)


def _set_robot_pose(env, env_ids, robot_xy, yaw):
    """Set robot root pose using env-local XY + sampled yaw."""
    asset = env.scene["robot"]
    origins = env.scene.env_origins[env_ids]
    n = env_ids.numel()
    pos = torch.zeros(n, 3, device=env.device)
    pos[:, 0] = robot_xy[:, 0]
    pos[:, 1] = robot_xy[:, 1]
    pos[:, 2] = 0.05  # match asset init height
    world_pos = pos + origins
    half_yaw = 0.5 * yaw
    quat = torch.stack(
        [torch.cos(half_yaw), torch.zeros_like(yaw), torch.zeros_like(yaw), torch.sin(half_yaw)],
        dim=-1,
    )  # (w, x, y, z) for pure yaw rotation
    pose = torch.cat([world_pos, quat], dim=-1)
    asset.write_root_pose_to_sim(pose, env_ids=env_ids)
    vel = torch.zeros(n, 6, device=env.device)
    asset.write_root_velocity_to_sim(vel, env_ids=env_ids)


def _sample_robot_start(room_w: torch.Tensor, room_h: torch.Tensor, device) -> torch.Tensor:
    """Sample robot start inside the room with ≥ 0.5 m wall clearance."""
    n = room_w.shape[0]
    margin = 0.5
    x_lo = -0.5 * room_w + margin
    x_hi = 0.5 * room_w - margin
    y_lo = -0.5 * room_h + margin
    y_hi = 0.5 * room_h - margin
    rx = torch.rand(n, device=device) * (x_hi - x_lo) + x_lo
    ry = torch.rand(n, device=device) * (y_hi - y_lo) + y_lo
    return torch.stack([rx, ry], dim=-1)


def _sample_obstacles(
    room_w: torch.Tensor,
    room_h: torch.Tensor,
    robot_xy: torch.Tensor,
    n_obs: int,
    device,
    max_tries: int = 32,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    """Sample axis-aligned cube obstacles with constraints, with rejection sampling.

    Returns ``(centres, half_extents)`` each shaped ``[n_envs, n_obs, 2]``,
    or ``(None, None)`` when ``n_obs == 0``. Slots that fail validity after
    ``max_tries`` are NaN-padded.
    """
    if n_obs <= 0:
        return None, None
    n_envs = room_w.shape[0]

    centres = torch.full((n_envs, n_obs, 2), float("nan"), device=device)
    halves  = torch.full((n_envs, n_obs, 2), float("nan"), device=device)

    margin = 0.5
    for k in range(n_obs):
        valid = torch.zeros(n_envs, dtype=torch.bool, device=device)
        for _ in range(max_tries):
            # Sample half-extents
            hx = _u(*OBSTACLE_HALF_RANGE, n_envs, device)
            hy = _u(*OBSTACLE_HALF_RANGE, n_envs, device)
            # Sample centres inside room with margin from walls (incl. obstacle size)
            x_lo = -0.5 * room_w + margin + hx
            x_hi =  0.5 * room_w - margin - hx
            y_lo = -0.5 * room_h + margin + hy
            y_hi =  0.5 * room_h - margin - hy
            cx = torch.rand(n_envs, device=device) * (x_hi - x_lo) + x_lo
            cy = torch.rand(n_envs, device=device) * (y_hi - y_lo) + y_lo

            # Constraint: ≥ 0.8 m from robot start
            d_robot = torch.linalg.norm(
                torch.stack([cx - robot_xy[:, 0], cy - robot_xy[:, 1]], dim=-1), dim=-1
            )
            ok = d_robot >= 0.8

            # Constraint: ≥ 0.5 m from previously placed obstacles in this env
            for j in range(k):
                prev_cx = centres[:, j, 0]
                prev_cy = centres[:, j, 1]
                # If previous slot is NaN (skipped), no constraint
                prev_valid = ~torch.isnan(prev_cx)
                d_prev = torch.linalg.norm(
                    torch.stack([cx - prev_cx, cy - prev_cy], dim=-1), dim=-1
                )
                ok = ok & (~prev_valid | (d_prev >= (0.5 + hx + halves[:, j, 0])))

            # Also valid range (room large enough to host this cube + margin)
            ok = ok & (x_hi >= x_lo) & (y_hi >= y_lo)

            new = ok & ~valid
            centres[:, k, 0] = torch.where(new, cx, centres[:, k, 0])
            centres[:, k, 1] = torch.where(new, cy, centres[:, k, 1])
            halves[:, k, 0]  = torch.where(new, hx, halves[:, k, 0])
            halves[:, k, 1]  = torch.where(new, hy, halves[:, k, 1])
            valid = valid | new
            if valid.all():
                break
        # leftover invalid envs stay NaN — obstacle is "inactive" for those

    return centres, halves
