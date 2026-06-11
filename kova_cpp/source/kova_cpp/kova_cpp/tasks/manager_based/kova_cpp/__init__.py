# Copyright (c) 2026, KOVA Project.
# SPDX-License-Identifier: BSD-3-Clause


import os
import gymnasium as gym

from . import agents
from .kova_cpp_env_cfg import KovaCppEnvCfg, KovaCppEnvCfg_PLAY


_AGENTS_DIR = os.path.join(os.path.dirname(__file__), "agents")
_SKRL_CFG = os.path.join(_AGENTS_DIR, "skrl_ppo_cfg.yaml")


gym.register(
    id="Isaac-Coverage-KOVA-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": KovaCppEnvCfg,
        "skrl_cfg_entry_point": _SKRL_CFG,
    },
)

gym.register(
    id="Isaac-Coverage-KOVA-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": KovaCppEnvCfg_PLAY,
        "skrl_cfg_entry_point": _SKRL_CFG,
    },
)
