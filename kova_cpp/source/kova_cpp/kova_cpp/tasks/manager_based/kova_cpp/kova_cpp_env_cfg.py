# Copyright (c) 2025, KOVA Project.
# SPDX-License-Identifier: BSD-3-Clause

"""Manager-based RL environment for KOVA Coverage Path Planning.

Follows the same pattern as YHBot's ``yhbot_navigation_env_cfg.py``.

CURRICULUM LEARNING (fixed-per-level geometry)
----------------------------------------------
Walls and obstacles are spawned ONCE at fixed positions for the current
curriculum level and are NEVER moved at runtime (runtime repositioning of
kinematic bodies is unstable in Isaac Lab). To change level, edit
``CURRICULUM_LEVEL`` in ``mdp/events.py`` — every dimension below derives
from ``LEVEL_GEOMETRY`` automatically.
"""

from __future__ import annotations

import math

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg, ArticulationCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, MultiMeshRayCasterCfg
from isaaclab.sensors.ray_caster import patterns
from isaaclab.utils import configclass

from . import mdp
from .assets import (
    KOVA_CFG,
    KOVA_BODY_RADIUS,
    KOVA_MAX_ANG,
    KOVA_MAX_LIN,
    KOVA_WHEEL_BASE,
    KOVA_WHEEL_RADIUS,
)
from .mdp.events import (
    CURRICULUM_LEVEL,
    MAX_OBSTACLES,
    WALL_HEIGHT,
    WALL_THICKNESS,
    active_geometry,
    obstacle_list,
    room_size_m,
)


# ============================================================================
# Level-derived scene geometry (module-level — NOT inside the @configclass
# body, because Isaac Lab's InteractiveScene treats every class attribute as
# an asset cfg).
#
# CURRICULUM: all of these derive from the active level in mdp/events.py.
# To change difficulty, edit CURRICULUM_LEVEL there — not here.
# ============================================================================

_ROOM_W, _ROOM_H = room_size_m()           # interior room size for this level
_WALL_HEIGHT = WALL_HEIGHT
_WALL_THICK = WALL_THICKNESS

# Wall lengths span the full room side plus the corner overlap so corners close.
_WALL_LEN_X = _ROOM_W + 2.0 * _WALL_THICK   # north/south walls run along X
_WALL_LEN_Y = _ROOM_H + 2.0 * _WALL_THICK   # east/west walls run along Y

# Wall centre offsets: each wall sits just outside the interior on its side.
_HALF_W = 0.5 * _ROOM_W
_HALF_H = 0.5 * _ROOM_H
_HALF_T = 0.5 * _WALL_THICK
_WALL_Z = 0.5 * _WALL_HEIGHT

# Obstacle layout for this level: list of (cx, cy, half_x, half_y).
_OBSTACLES = obstacle_list()


def _obstacle_init_pos(slot: int) -> tuple[float, float, float]:
    """Spawn position for obstacle ``slot``. Active obstacles sit in the room;
    unused slots are parked underground (and never moved again)."""
    if slot < len(_OBSTACLES):
        cx, cy, _, _ = _OBSTACLES[slot]
        return (float(cx), float(cy), _WALL_Z)
    return (0.0, 0.0, -10.0)  # hidden underground


def _obstacle_size(slot: int) -> tuple[float, float, float]:
    if slot < len(_OBSTACLES):
        _, _, hx, hy = _OBSTACLES[slot]
        return (2.0 * float(hx), 2.0 * float(hy), _WALL_HEIGHT)
    return (0.6, 0.6, _WALL_HEIGHT)  # default size for parked slot


# ============================================================================
# Scene
# ============================================================================


@configclass
class KovaCppSceneCfg(InteractiveSceneCfg):
    """Scene: ground + robot + 4 walls + 2 obstacle slots + LiDAR + contact sensor."""

    # ---- Ground (shared, but spawned per env via env_origins) ----
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(size=(500.0, 500.0)),
    )

    # ---- Lighting ----
    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(color=(0.9, 0.9, 0.9), intensity=2500.0),
    )

    # ---- Robot ----
    robot: ArticulationCfg = KOVA_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    # ---- Walls (4 kinematic cubes, FIXED at this level's room geometry) ----
    # CURRICULUM: size & position derive from the active level (mdp/events.py).
    # These are spawned once and NEVER moved at runtime — that keeps the sim
    # stable (runtime kinematic repositioning is what made walls shake before).
    wall_north: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/wall_north",
        spawn=sim_utils.CuboidCfg(
            size=(_WALL_LEN_X, _WALL_THICK, _WALL_HEIGHT),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.75, 0.75, 0.78)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, _HALF_H + _HALF_T, _WALL_Z)),
    )
    wall_south: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/wall_south",
        spawn=sim_utils.CuboidCfg(
            size=(_WALL_LEN_X, _WALL_THICK, _WALL_HEIGHT),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.75, 0.75, 0.78)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, -(_HALF_H + _HALF_T), _WALL_Z)),
    )
    wall_east: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/wall_east",
        spawn=sim_utils.CuboidCfg(
            size=(_WALL_THICK, _WALL_LEN_Y, _WALL_HEIGHT),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.75, 0.75, 0.78)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(_HALF_W + _HALF_T, 0.0, _WALL_Z)),
    )
    wall_west: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/wall_west",
        spawn=sim_utils.CuboidCfg(
            size=(_WALL_THICK, _WALL_LEN_Y, _WALL_HEIGHT),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.75, 0.75, 0.78)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(-(_HALF_W + _HALF_T), 0.0, _WALL_Z)),
    )

    # ---- Obstacle cubes (FIXED per level; static, never moved at runtime) ----
    # CURRICULUM: active obstacles spawn at this level's positions; unused slots
    # spawn parked underground and stay there. Edit obstacle layout in
    # LEVEL_GEOMETRY (mdp/events.py), not here.
    obstacle_0: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/obstacle_0",
        spawn=sim_utils.CuboidCfg(
            size=_obstacle_size(0),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.35, 0.35, 0.38)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=_obstacle_init_pos(0)),
    )
    obstacle_1: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/obstacle_1",
        spawn=sim_utils.CuboidCfg(
            size=_obstacle_size(1),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.35, 0.35, 0.38)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=_obstacle_init_pos(1)),
    )

    # ---- 2-D LiDAR (Multi-mesh RayCaster on a single horizontal ring) ----
    # 360° FOV @ 6° resolution = 60 rays. Mounted 0.1 m above robot base.
    lidar = MultiMeshRayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        update_period=1.0 / 30.0,
        offset=MultiMeshRayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 0.1)),
        ray_alignment="yaw",
        pattern_cfg=patterns.LidarPatternCfg(
            channels=1,
            vertical_fov_range=(0.0, 0.0),
            horizontal_fov_range=(0.0, 360.0),
            horizontal_res=6.0,
        ),
        max_distance=5.0,
        mesh_prim_paths=[
            "/World/ground",
            "{ENV_REGEX_NS}/wall_north",
            "{ENV_REGEX_NS}/wall_south",
            "{ENV_REGEX_NS}/wall_east",
            "{ENV_REGEX_NS}/wall_west",
            "{ENV_REGEX_NS}/obstacle_0",
            "{ENV_REGEX_NS}/obstacle_1",
        ],
        debug_vis=False,
    )

    # ---- Contact sensor on the whole robot body ----
    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*",
        update_period=0.0,
        history_length=3,
        debug_vis=False,
    )


# ============================================================================
# Actions
# ============================================================================


@configclass
class ActionsCfg:
    """Differential-drive action: 2-D (v, ω) in [-1, 1]."""
    diff_drive = mdp.DifferentialDriveActionCfg(
        asset_name="robot",
        left_wheel_joint_name="left_wheel_joint",
        right_wheel_joint_name="right_wheel_joint",
        wheel_radius=KOVA_WHEEL_RADIUS,
        wheel_base=KOVA_WHEEL_BASE,
        max_linear_speed=KOVA_MAX_LIN,
        max_angular_speed=KOVA_MAX_ANG,
    )


# ============================================================================
# Observations
# ============================================================================


@configclass
class ObservationsCfg:
    """Observations are concatenated in order. The coverage_map term MUST come
    first — it owns env.coverage_map and env.action_history."""

    @configclass
    class PolicyCfg(ObsGroup):
        # Coverage map (multi-scale ego map). 3 * 4 * 32 * 32 = 12288.
        # This term ALSO updates env.coverage_map and env.action_history each step.
        coverage_map = ObsTerm(
            func=mdp.CoverageMapObs,
            params={
                "cell_size": 0.1,
                "max_world_size": 24.0,
                "robot_radius": KOVA_BODY_RADIUS,
                "n_scales": 4,
                "scale_factor": 4,
                "finest_pixel_size": 0.0375,
                "patch_size": 32,
                "action_history_len": 10,
            },
        )

        # LiDAR (60 rays normalised)
        lidar = ObsTerm(
            func=mdp.lidar_obs,
            params={"sensor_cfg": SceneEntityCfg("lidar"), "num_rays_out": 60, "max_range": 5.0},
        )

        # Action history (last 10 (v, ω) = 20)
        action_history = ObsTerm(func=mdp.action_history_obs)

        # Distance to nearest uncovered cell (normalised, 1 scalar)
        nearest_uncovered = ObsTerm(func=mdp.nearest_uncovered_distance)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


# ============================================================================
# Events
# ============================================================================


@configclass
class EventsCfg:
    """Reset event: resets the robot pose and syncs the coverage map to this
    level's FIXED room + obstacle layout. Walls/obstacles are static (set once
    at spawn). See ``mdp/events.py`` and CURRICULUM_LEVEL there.
    """
    reset_scene = EventTerm(func=mdp.reset_level, mode="reset")


# ============================================================================
# Rewards
# ============================================================================


@configclass
class RewardsCfg:
    """All weights from the spec."""

    new_cell = RewTerm(func=mdp.new_cell_reward, weight=1.0)
    step = RewTerm(func=mdp.step_penalty, weight=-1.0)
    tv = RewTerm(
        func=mdp.total_variation_reward,
        weight=0.2,
        params={"v_max": KOVA_MAX_LIN, "dt": None},  # filled by env at runtime
    )
    distance_guidance = RewTerm(func=mdp.distance_guidance_reward, weight=0.25)
    direction_change = RewTerm(func=mdp.direction_change_penalty, weight=-0.25)
    completion = RewTerm(
        func=mdp.completion_bonus,
        weight=200.0,
        params={"coverage_threshold": 0.95},
    )
    collision = RewTerm(
        func=mdp.collision_penalty,
        weight=-10.0,
        params={
            "force_threshold": 0.5,
            "startup_grace_steps": 30,
            "sensor_cfg": SceneEntityCfg("contact_forces"),
        },
    )
    # Layer-1 action masking via reward shaping (small negative weight).
    blocking = RewTerm(func=mdp.blocking_penalty, weight=-0.5)


# ============================================================================
# Terminations
# ============================================================================


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    no_progress = DoneTerm(
        func=mdp.no_progress,
        params={"max_steps_without_new_cell": 500},
        time_out=False,
    )
    collision = DoneTerm(
        func=mdp.collision_termination,
        params={
            "force_threshold": 0.5,
            "startup_grace_steps": 30,
            "sensor_cfg": SceneEntityCfg("contact_forces"),
        },
        time_out=False,
    )
    # Physics-instability guard: reset any env whose robot flies away / goes NaN
    # before the bad state corrupts the coverage map and policy input.
    out_of_bounds = DoneTerm(
        func=mdp.robot_out_of_bounds,
        params={"max_height": 1.5, "max_xy_from_origin": 30.0},
        time_out=False,
    )


# ============================================================================
# Environment cfg
# ============================================================================


@configclass
class KovaCppEnvCfg(ManagerBasedRLEnvCfg):
    """KOVA CPP environment configuration."""

    # CURRICULUM: the active level is set by CURRICULUM_LEVEL in mdp/events.py.
    # This mirrors it here only for episode-length / world-size derivation.
    curriculum_level: int = CURRICULUM_LEVEL

    # Scene
    scene: KovaCppSceneCfg = KovaCppSceneCfg(num_envs=4096, env_spacing=30.0)

    # MDP managers
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventsCfg = EventsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    def __post_init__(self) -> None:
        # General
        self.decimation = 4
        self.sim.dt = 1.0 / 60.0
        self.sim.render_interval = self.decimation

        # Episode length scales with level (bigger rooms need more time to cover).
        # CURRICULUM: adjust these if a level needs more/less time to plateau.
        level_episode_s = {1: 60.0, 2: 80.0, 3: 100.0, 4: 120.0, 5: 150.0, 6: 200.0}
        self.episode_length_s = level_episode_s.get(CURRICULUM_LEVEL, 120.0)

        # Coverage-map world size: cover the room plus margin. Grid scales with
        # level so lower levels use less VRAM.
        room_w, room_h = room_size_m()
        max_world = max(6.0, max(room_w, room_h) + 4.0)
        self.observations.policy.coverage_map.params["max_world_size"] = max_world

        # Inject post-decimation step dt into the TV reward (for normalisation).
        step_dt = self.decimation * self.sim.dt
        self.rewards.tv.params["dt"] = step_dt

        # Env spacing must exceed the room so per-env walls never overlap a neighbour.
        self.scene.env_spacing = max(self.scene.env_spacing, max(room_w, room_h) + 6.0)

        # Viewer
        self.viewer.eye = (8.0, 8.0, 8.0)
        self.viewer.lookat = (0.0, 0.0, 0.0)


# ============================================================================
# Play variant (single env, no randomisation across multiple envs)
# ============================================================================


@configclass
class KovaCppEnvCfg_PLAY(KovaCppEnvCfg):
    """Single-env config for visualisation / play."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.scene.env_spacing = 30.0
        self.observations.policy.enable_corruption = False
