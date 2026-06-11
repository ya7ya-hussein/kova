# Copyright (c) 2026, KOVA Project.
# SPDX-License-Identifier: BSD-3-Clause


from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg
from isaaclab.managers.manager_base import ManagerTermBase
from isaaclab.managers.manager_term_cfg import RewardTermCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


# Stateless rewards

def new_cell_reward(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """+1 per new cell covered this step (unweighted: caller supplies +1.0 weight)."""
    return env.coverage_map.cells_visited_this_step.float()


def step_penalty(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """Constant 1.0 per step (weight should be -1.0)."""
    return torch.ones(env.num_envs, device=env.device)


def distance_guidance_reward(env: "ManagerBasedRLEnv") -> torch.Tensor:
    
    d = env.coverage_map.get_nearest_uncovered_distance().squeeze(-1)  # [N]
    return 1.0 / (d + 0.5)


def collision_penalty(
    env: "ManagerBasedRLEnv",
    force_threshold: float = 0.5,
    startup_grace_steps: int = 30,
    sensor_cfg: SceneEntityCfg | None = None,
) -> torch.Tensor:
    
    sensor = env.scene[sensor_cfg.name]
    forces = sensor.data.net_forces_w
    if forces is None:
        return torch.zeros(env.num_envs, device=env.device)
    horizontal = torch.linalg.norm(forces[..., :2], dim=-1)
    any_hit = (horizontal > force_threshold).any(dim=-1)
    grace = env.episode_length_buf < startup_grace_steps
    return (any_hit & ~grace).float()


def completion_bonus(
    env: "ManagerBasedRLEnv",
    coverage_threshold: float = 0.95,
) -> torch.Tensor:

    cmap = env.coverage_map
    pct = cmap.coverage_pct()
    crossed_now = (pct >= coverage_threshold) & ~cmap.completion_bonus_given
    # Latch the flag so this fires only once per episode
    cmap.completion_bonus_given = cmap.completion_bonus_given | crossed_now
    return crossed_now.float()


# Stateful rewards
class total_variation_reward(ManagerTermBase):

    def __init__(self, cfg: RewardTermCfg, env: "ManagerBasedRLEnv"):
        super().__init__(cfg, env)
        params = cfg.params or {}
        v_max = params.get("v_max", 0.6)
        dt = params.get("dt", env.step_dt)  # outer (post-decimation) step duration
        self._norm = max(1e-6, 2.0 * v_max * dt)
        self._initialised = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        if env_ids is None:
            self._initialised.zero_()
        else:
            self._initialised[env_ids] = False
        # CoverageMap owns prev_tv and resets it on its own reset()

    def __call__(
        self,
        env: "ManagerBasedRLEnv",
        v_max: float = 0.6,
        dt: float | None = None,
    ) -> torch.Tensor:
        cmap = env.coverage_map
        tv_now = cmap.compute_tv()
        delta = tv_now - cmap.prev_tv
        # On the first step after reset there is no previous TV — return 0
        reward = torch.where(self._initialised, -delta / self._norm, torch.zeros_like(delta))
        # Update state
        cmap.prev_tv = tv_now.detach()
        self._initialised[:] = True
        return reward


class direction_change_penalty(ManagerTermBase):

    def __init__(self, cfg: RewardTermCfg, env: "ManagerBasedRLEnv"):
        super().__init__(cfg, env)
        self._prev_sign = torch.zeros(env.num_envs, device=env.device)
        self._initialised = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        if env_ids is None:
            self._prev_sign.zero_()
            self._initialised.zero_()
        else:
            self._prev_sign[env_ids] = 0.0
            self._initialised[env_ids] = False

    def __call__(self, env: "ManagerBasedRLEnv") -> torch.Tensor:
        # Use raw policy action's ω component 
        action = env.action_manager.action  # [N, 2]
        omega = action[:, 1]
        sign_now = torch.sign(omega)
        flipped = (
            self._initialised
            & (sign_now != 0)
            & (self._prev_sign != 0)
            & (sign_now != self._prev_sign)
        )
        # Update state
        self._prev_sign = torch.where(sign_now != 0, sign_now, self._prev_sign)
        self._initialised[:] = True
        return flipped.float()


class blocking_penalty(ManagerTermBase):
    _DIRS = torch.tensor(
        [[0.0, 1.0], [1.0, 0.0], [0.0, -1.0], [-1.0, 0.0]],
        dtype=torch.float32,
    )

    def __init__(self, cfg: RewardTermCfg, env: "ManagerBasedRLEnv"):
        super().__init__(cfg, env)
        self._dirs = self._DIRS.to(env.device) 

    def __call__(self, env: "ManagerBasedRLEnv") -> torch.Tensor:
        cmap = env.coverage_map
        valid_dirs = cmap.get_valid_directions()
        has_valid_option = valid_dirs.any(dim=-1) 

        # Compute the world frame heading direction unit vector from yaw
        yaw = cmap.robot_yaw
        heading = torch.stack([torch.cos(yaw), torch.sin(yaw)], dim=-1)

        # Project heading onto each cardinal, pick the one with max dot product
        dots = heading @ self._dirs.T
        chosen = dots.argmax(dim=-1)

        # The chosen cardinal is "invalid" if valid_dirs[chosen] == False
        idx = torch.arange(env.num_envs, device=env.device)
        chose_invalid = ~valid_dirs[idx, chosen]
        # Only penalise when valid options actually exist
        return (chose_invalid & has_valid_option).float()
