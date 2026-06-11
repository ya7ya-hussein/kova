# Copyright (c) 2026, KOVA Project.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg
from isaaclab.managers.manager_base import ManagerTermBase
from isaaclab.managers.manager_term_cfg import ObservationTermCfg
from ..coverage_map import CoverageMap

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class CoverageMapObs(ManagerTermBase):
    """Owns CoverageMap lifecycle; returns multi-scale egocentric map (flattened)."""

    def __init__(self, cfg: ObservationTermCfg, env: "ManagerBasedRLEnv"):
        super().__init__(cfg, env)

        # Cfg parameters
        params = cfg.params or {}
        self._cell_size: float = params.get("cell_size", 0.1)
        self._max_world_size: float = params.get("max_world_size", 24.0)
        self._robot_radius: float = params.get("robot_radius", 0.18)
        self._n_scales: int = params.get("n_scales", 4)
        self._scale_factor: int = params.get("scale_factor", 4)
        self._finest_pixel_size: float = params.get("finest_pixel_size", 0.0375)
        self._patch_size: int = params.get("patch_size", 32)
        self._action_history_len: int = params.get("action_history_len", 10)

        # Lazily create the coverage map and attach to env 
        if not hasattr(env, "coverage_map"):
            env.coverage_map = CoverageMap(
                num_envs=env.num_envs,
                device=env.device,
                cell_size=self._cell_size,
                max_world_size=self._max_world_size,
                robot_radius=self._robot_radius,
                n_scales=self._n_scales,
                scale_factor=self._scale_factor,
                finest_pixel_size=self._finest_pixel_size,
                obs_patch_size=self._patch_size,
            )

        # Action history buffer 
        if not hasattr(env, "action_history"):
            env.action_history = torch.zeros(
                env.num_envs, self._action_history_len, 2, device=env.device, dtype=torch.float32
            )

        if not hasattr(env, "kova_prev_yaw"):
            env.kova_prev_yaw = torch.zeros(env.num_envs, device=env.device)

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        cmap: CoverageMap = self._env.coverage_map  
        cmap.reset(env_ids)
        if env_ids is None:
            self._env.action_history.zero_()
        else:
            self._env.action_history[env_ids] = 0.0

    def __call__(
        self,
        env: "ManagerBasedRLEnv",
        cell_size: float = 0.1,
        max_world_size: float = 20.0,
        robot_radius: float = 0.18,
        n_scales: int = 4,
        scale_factor: int = 4,
        finest_pixel_size: float = 0.0375,
        patch_size: int = 32,
        action_history_len: int = 10,
    ) -> torch.Tensor:
        robot = env.scene["robot"]
        robot_xy = robot.data.root_pos_w[:, :2] - env.scene.env_origins[:, :2]
        quat = robot.data.root_quat_w 
        w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
        yaw = torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))

        robot_xy = torch.nan_to_num(robot_xy, nan=0.0, posinf=0.0, neginf=0.0)
        yaw = torch.nan_to_num(yaw, nan=0.0, posinf=0.0, neginf=0.0)
        # --------------------------------------------------------------------

        env.coverage_map.update(robot_xy, yaw)

        last_act = env.action_manager.action  
        # Roll buffer
        env.action_history[:, :-1] = env.action_history[:, 1:].clone()
        env.action_history[:, -1] = last_act[:, :2]

        # Return multi-scale egocentric obs
        obs = env.coverage_map.get_multiscale_obs()
        return torch.nan_to_num(obs, nan=0.0, posinf=1.0, neginf=0.0)


# LiDAR
def lidar_obs(
    env: "ManagerBasedRLEnv",
    sensor_cfg: SceneEntityCfg,
    num_rays_out: int = 60,
    max_range: float = 5.0,
) -> torch.Tensor:

    sensor = env.scene[sensor_cfg.name]
    hits = sensor.data.ray_hits_w  
    pos = sensor.data.pos_w.unsqueeze(1)  
    dist = torch.linalg.norm(hits - pos, dim=-1)  
    # Sensor may emit inf for misses
    dist = torch.nan_to_num(dist, nan=max_range, posinf=max_range, neginf=max_range)
    dist = dist.clamp(0.0, max_range)

    # Downsample to num_rays_out
    R = dist.shape[1]
    if R != num_rays_out:
        # Pick evenly-spaced ray indices
        idxs = torch.linspace(0, R - 1, num_rays_out, device=dist.device).long()
        dist = dist[:, idxs]

    return (dist / max_range).clamp(0.0, 1.0)


# Action history
def action_history_obs(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """Flattened last-T actions: [N, T * 2]. CoverageMapObs maintains the buffer."""
    return env.action_history.view(env.num_envs, -1)


# Distance to nearest uncovered cell
def nearest_uncovered_distance(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """Single normalised scalar in [0, 1], shape [N, 1]."""
    return env.coverage_map.get_nearest_uncovered_distance()