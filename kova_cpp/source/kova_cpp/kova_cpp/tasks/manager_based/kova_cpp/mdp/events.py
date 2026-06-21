# Copyright (c) 2026, KOVA Project.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math
import torch
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


CURRICULUM_LEVEL = 3

LEVEL_GEOMETRY = {
    1: dict(room_w=4.0,  room_h=4.0,  robot_box=(1.0, 1.0), obstacles=[]),
    2: dict(room_w=6.0,  room_h=6.0,  robot_box=(0.8, 0.8),
            obstacles=[(1.2, 1.2, 0.3, 0.3)]),
    3: dict(room_w=8.0,  room_h=8.0,  robot_box=(2.0, 2.0),
            obstacles=[(2.0, 1.5, 0.4, 0.4)]),
    4: dict(room_w=12.0, room_h=12.0, robot_box=(3.0, 3.0),
            obstacles=[(3.0, 2.0, 0.5, 0.5), (-2.5, -3.0, 0.4, 0.4)]),
    5: dict(room_w=16.0, room_h=16.0, robot_box=(4.0, 4.0),
            obstacles=[(4.0, 3.0, 0.5, 0.5), (-3.5, -4.0, 0.5, 0.5)]),
    6: dict(room_w=20.0, room_h=20.0, robot_box=(5.0, 5.0),
            obstacles=[(5.0, 4.0, 0.5, 0.5), (-4.5, -5.0, 0.5, 0.5)]),
}

MAX_OBSTACLES = 2

WALL_THICKNESS = 0.2  
WALL_HEIGHT = 0.5     


# Convenience accessors
def active_geometry() -> dict:
    """Return the geometry dict for the currently selected curriculum level."""
    return LEVEL_GEOMETRY.get(CURRICULUM_LEVEL, LEVEL_GEOMETRY[1])


def room_size_m() -> tuple[float, float]:
    g = active_geometry()
    return float(g["room_w"]), float(g["room_h"])


def obstacle_list() -> list[tuple[float, float, float, float]]:
    """Obstacles for the active level: list of (cx, cy, half_x, half_y)."""
    return list(active_geometry()["obstacles"])


# Reset event
def reset_level(env: "ManagerBasedRLEnv", env_ids: torch.Tensor) -> None:
    """Reset robot pose + sync coverage map for the fixed-geometry level."""
    if env_ids.numel() == 0:
        return
    device = env.device
    n = env_ids.numel()

    room_w, room_h = room_size_m()
    obstacles = obstacle_list()

    # 1) Reset robot pose (random within the level's robot_box, random yaw)
    g = active_geometry()
    bx, by = g["robot_box"]
    rx = (torch.rand(n, device=device) * 2.0 - 1.0) * bx
    ry = (torch.rand(n, device=device) * 2.0 - 1.0) * by
    yaw = (torch.rand(n, device=device) * 2.0 - 1.0) * math.pi
    _set_robot_pose(env, env_ids, rx, ry, yaw)

    env.coverage_map.robot_xy_world[env_ids] = torch.stack([rx, ry], dim=-1)

    # 2) Sync coverage map with FIXED room size + FIXED obstacles
    room_size = torch.tensor([room_w, room_h], device=device).expand(n, 2).contiguous()

    obstacles_world = torch.full(
        (n, MAX_OBSTACLES, 4), float("nan"), device=device, dtype=torch.float32
    )
    for k, (cx, cy, hx, hy) in enumerate(obstacles[:MAX_OBSTACLES]):
        obstacles_world[:, k, 0] = cx
        obstacles_world[:, k, 1] = cy
        obstacles_world[:, k, 2] = hx
        obstacles_world[:, k, 3] = hy

    env.coverage_map.reset_room(
        env_ids, room_size=room_size, obstacles_world=obstacles_world
    )


# Helpers
def _set_robot_pose(env, env_ids, rx, ry, yaw):
    """Set robot root pose: env-local XY (+ origin) + yaw rotation.

    Resetting the robot articulation at runtime is fully supported by Isaac Lab
    (unlike repositioning kinematic walls), so this is safe.
    """
    asset = env.scene["robot"]
    origins = env.scene.env_origins[env_ids]
    n = env_ids.numel()
    pos = torch.zeros(n, 3, device=env.device)
    pos[:, 0] = rx
    pos[:, 1] = ry
    pos[:, 2] = 0.05 
    world_pos = pos + origins
    half_yaw = 0.5 * yaw
    quat = torch.stack(
        [torch.cos(half_yaw), torch.zeros_like(yaw), torch.zeros_like(yaw), torch.sin(half_yaw)],
        dim=-1,
    )  
    pose = torch.cat([world_pos, quat], dim=-1)
    asset.write_root_pose_to_sim(pose, env_ids=env_ids)
    vel = torch.zeros(n, 6, device=env.device)
    asset.write_root_velocity_to_sim(vel, env_ids=env_ids)