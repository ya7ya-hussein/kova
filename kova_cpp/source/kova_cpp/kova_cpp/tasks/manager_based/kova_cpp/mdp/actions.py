# Copyright (c) 2025, KOVA Project.
# SPDX-License-Identifier: BSD-3-Clause

"""Custom differential-drive action term for KOVA.

Maps a 2-D normalised policy output (v, ω) ∈ [-1, 1]^2 to per-wheel velocity
targets via differential-drive kinematics, matching the spec:

    ω_left  = (v - ω · L/2) / r_wheel
    ω_right = (v + ω · L/2) / r_wheel

where L is the wheel base and r_wheel is the wheel radius.
"""

from __future__ import annotations

import torch
from dataclasses import MISSING

from isaaclab.assets import Articulation
from isaaclab.managers.action_manager import ActionTerm, ActionTermCfg
from isaaclab.utils import configclass


class DifferentialDriveAction(ActionTerm):
    """Differential-drive action: 2-D continuous (v, ω) → two wheel velocities."""

    cfg: "DifferentialDriveActionCfg"
    _asset: Articulation

    def __init__(self, cfg: "DifferentialDriveActionCfg", env):
        super().__init__(cfg, env)

        # Resolve wheel joint indices on the articulation
        left_ids, _ = self._asset.find_joints(cfg.left_wheel_joint_name)
        right_ids, _ = self._asset.find_joints(cfg.right_wheel_joint_name)
        if len(left_ids) != 1 or len(right_ids) != 1:
            raise ValueError(
                f"DifferentialDriveAction expected exactly one match each for "
                f"'{cfg.left_wheel_joint_name}' and '{cfg.right_wheel_joint_name}'."
            )
        self._left_joint_id = left_ids[0]
        self._right_joint_id = right_ids[0]
        self._wheel_joint_ids = torch.tensor(
            [self._left_joint_id, self._right_joint_id],
            device=self.device,
            dtype=torch.long,
        )

        # Buffers
        self._raw_actions = torch.zeros(self.num_envs, 2, device=self.device)
        self._processed_actions = torch.zeros(self.num_envs, 2, device=self.device)  # [v_mps, w_radps]
        self._wheel_targets = torch.zeros(self.num_envs, 2, device=self.device)       # [w_L, w_R]

        # Stash kinematic params
        self._r = float(cfg.wheel_radius)
        self._L = float(cfg.wheel_base)
        self._v_max = float(cfg.max_linear_speed)
        self._w_max = float(cfg.max_angular_speed)

    # ----- ActionTerm API
    @property
    def action_dim(self) -> int:
        return 2

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    def process_actions(self, actions: torch.Tensor) -> None:
        # Policy is expected to output values roughly in [-1, 1]; clamp defensively.
        self._raw_actions.copy_(actions)
        a = actions.clamp(-1.0, 1.0)
        v = a[:, 0] * self._v_max     # m/s
        w = a[:, 1] * self._w_max     # rad/s
        self._processed_actions[:, 0] = v
        self._processed_actions[:, 1] = w

        # Diff-drive inverse kinematics
        half_L = 0.5 * self._L
        wl = (v - w * half_L) / self._r
        wr = (v + w * half_L) / self._r
        self._wheel_targets[:, 0] = wl
        self._wheel_targets[:, 1] = wr

    def apply_actions(self) -> None:
        # Set velocity targets on both wheel joints in one call
        self._asset.set_joint_velocity_target(
            self._wheel_targets, joint_ids=self._wheel_joint_ids.tolist()
        )

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        if env_ids is None:
            self._raw_actions.zero_()
            self._processed_actions.zero_()
            self._wheel_targets.zero_()
        else:
            self._raw_actions[env_ids] = 0.0
            self._processed_actions[env_ids] = 0.0
            self._wheel_targets[env_ids] = 0.0


@configclass
class DifferentialDriveActionCfg(ActionTermCfg):
    """Cfg for ``DifferentialDriveAction``."""

    class_type: type = DifferentialDriveAction

    left_wheel_joint_name: str = MISSING
    right_wheel_joint_name: str = MISSING
    wheel_radius: float = 0.03575
    wheel_base: float = 0.233
    max_linear_speed: float = 0.6
    max_angular_speed: float = 0.6
