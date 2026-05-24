# Copyright (c) 2025, KOVA Project.
# SPDX-License-Identifier: BSD-3-Clause

"""Reward functions for KOVA Coverage Path Planning.

All rewards are batched, GPU-resident, and return ``torch.Tensor`` of shape
``[num_envs]``. Weights live in the env cfg's ``RewardsCfg``; this module
provides the raw signals.

Convention: a function returns the POSITIVE quantity to be combined with its
weight. So e.g. ``step_penalty`` returns 1.0 always, and the weight is -1.0.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg
from isaaclab.managers.manager_base import ManagerTermBase
from isaaclab.managers.manager_term_cfg import RewardTermCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


# ----------------------------------------------------------------------------
# Stateless rewards (read state from env.coverage_map / action manager)
# ----------------------------------------------------------------------------


def new_cell_reward(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """+1 per new cell covered this step (unweighted: caller supplies +1.0 weight)."""
    return env.coverage_map.cells_visited_this_step.float()


def step_penalty(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """Constant 1.0 per step (weight should be -1.0)."""
    return torch.ones(env.num_envs, device=env.device)


def distance_guidance_reward(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """1 / (d_norm + 0.5). Always-on pull toward uncovered space.

    Weight should be +0.25 (per spec). ``d_norm`` is in [0, 1].
    """
    d = env.coverage_map.get_nearest_uncovered_distance().squeeze(-1)  # [N]
    return 1.0 / (d + 0.5)


def collision_penalty(
    env: "ManagerBasedRLEnv",
    force_threshold: float = 0.5,
    startup_grace_steps: int = 30,
    sensor_cfg: SceneEntityCfg | None = None,
) -> torch.Tensor:
    """1.0 on collision, 0.0 otherwise (weight should be -10.0).

    Startup grace prevents spurious collisions from the asset settling at reset.
    """
    sensor = env.scene[sensor_cfg.name]
    forces = sensor.data.net_forces_w  # [N, B, 3]
    if forces is None:
        return torch.zeros(env.num_envs, device=env.device)
    horizontal = torch.linalg.norm(forces[..., :2], dim=-1)  # [N, B]
    any_hit = (horizontal > force_threshold).any(dim=-1)
    grace = env.episode_length_buf < startup_grace_steps
    return (any_hit & ~grace).float()


def completion_bonus(
    env: "ManagerBasedRLEnv",
    coverage_threshold: float = 0.95,
) -> torch.Tensor:
    """One-time +1 when coverage first crosses the threshold (weight = +200)."""
    cmap = env.coverage_map
    pct = cmap.coverage_pct()
    crossed_now = (pct >= coverage_threshold) & ~cmap.completion_bonus_given
    # Latch the flag so this fires only once per episode
    cmap.completion_bonus_given = cmap.completion_bonus_given | crossed_now
    return crossed_now.float()


# ----------------------------------------------------------------------------
# Stateful rewards
# ----------------------------------------------------------------------------


class total_variation_reward(ManagerTermBase):
    """-ΔTV(C) / (2 · v_max · dt).

    The instantaneous reward is ``- (TV_t - TV_{t-1}) / norm`` (positive when
    TV decreases, i.e. when the coverage boundary becomes smoother / smaller).
    Weight should be +0.2 per spec — so the FINAL sign is ``+0.2 * -(ΔTV/norm)``
    which penalises increases in TV. We return the bare ``- ΔTV / norm`` term;
    the cfg weight is +0.2.
    """

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
    """1.0 when the sign of angular velocity flips between consecutive steps
    (weight should be -0.25)."""

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
        # Use raw policy action's ω component (index 1)
        action = env.action_manager.action  # [N, 2]
        omega = action[:, 1]
        sign_now = torch.sign(omega)
        # Treat exact zero as "no change" — only count true sign flips
        flipped = (
            self._initialised
            & (sign_now != 0)
            & (self._prev_sign != 0)
            & (sign_now != self._prev_sign)
        )
        # Update state — only overwrite prev_sign with nonzero signs
        self._prev_sign = torch.where(sign_now != 0, sign_now, self._prev_sign)
        self._initialised[:] = True
        return flipped.float()


class blocking_penalty(ManagerTermBase):
    """Layer-1 action masking implemented as reward shaping.

    Penalise steering toward already-covered cells when uncovered options exist
    in the 4 cardinal directions. Returns 1.0 when the dominant heading vector
    points toward an INVALID direction (covered or blocked) but valid options
    exist; 0.0 otherwise.

    Weight should be small negative (e.g. -0.5) — exact tuning is in the cfg.
    """

    # Direction vectors (matching CoverageMap.get_valid_directions order)
    # N(+y), E(+x), S(-y), W(-x). In world frame the policy doesn't think in
    # cardinal directions, so we map the robot's commanded velocity vector
    # (rotated by yaw) to the closest cardinal direction.
    _DIRS = torch.tensor(
        [[0.0, 1.0], [1.0, 0.0], [0.0, -1.0], [-1.0, 0.0]],
        dtype=torch.float32,
    )

    def __init__(self, cfg: RewardTermCfg, env: "ManagerBasedRLEnv"):
        super().__init__(cfg, env)
        self._dirs = self._DIRS.to(env.device)  # [4, 2]

    def __call__(self, env: "ManagerBasedRLEnv") -> torch.Tensor:
        cmap = env.coverage_map
        valid_dirs = cmap.get_valid_directions()  # [N, 4] bool
        has_valid_option = valid_dirs.any(dim=-1)  # [N]

        # Compute the world-frame heading direction unit vector from yaw
        yaw = cmap.robot_yaw  # [N]
        heading = torch.stack([torch.cos(yaw), torch.sin(yaw)], dim=-1)  # [N, 2]

        # Project heading onto each cardinal; pick the one with max dot product
        dots = heading @ self._dirs.T  # [N, 4]
        chosen = dots.argmax(dim=-1)   # [N]

        # The chosen cardinal is "invalid" if valid_dirs[chosen] == False
        idx = torch.arange(env.num_envs, device=env.device)
        chose_invalid = ~valid_dirs[idx, chosen]
        # Only penalise when valid options actually exist
        return (chose_invalid & has_valid_option).float()