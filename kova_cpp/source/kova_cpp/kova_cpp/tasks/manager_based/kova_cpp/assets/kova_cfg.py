# Copyright (c) 2025, KOVA Project.
# SPDX-License-Identifier: BSD-3-Clause

"""iRobot Create 3 articulation config for KOVA."""

import os

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

# Default USD path. Override this in your env_cfg if your Create 3 USD lives elsewhere.
# Common locations:
#   - {ISAAC_NUCLEUS_DIR}/Robots/iRobot/Create_3/create_3.usd  (if shipped)
#   - A locally converted USD from create3_sim's create3.urdf
_DEFAULT_USD = os.environ.get(
    "KOVA_CREATE3_USD",
    os.path.join(os.path.dirname(__file__), "Create3/create_3.usd"),
)

# iRobot Create 3 specs (per the task brief)
KOVA_WHEEL_RADIUS = 0.03575   # m
KOVA_WHEEL_BASE   = 0.233     # m (distance between left & right wheels)
KOVA_MAX_LIN      = 0.6       # m/s
KOVA_MAX_ANG      = 0.6       # rad/s
KOVA_BODY_RADIUS  = 0.18      # m (chassis disc — informational; used by coverage map)

KOVA_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=_DEFAULT_USD,
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            rigid_body_enabled=True,
            max_linear_velocity=2.0,
            max_angular_velocity=10.0,
            max_depenetration_velocity=1.0,
            enable_gyroscopic_forces=True,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=4,
            sleep_threshold=0.005,
            stabilization_threshold=0.001,
        ),
        collision_props=sim_utils.CollisionPropertiesCfg(
            contact_offset=0.02,
            rest_offset=0.0,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        # Start slightly above ground so chassis settles cleanly.
        pos=(0.0, 0.0, 0.05),
        rot=(1.0, 0.0, 0.0, 0.0),
        joint_pos={
            "left_wheel_joint": 0.0,
            "right_wheel_joint": 0.0,
        },
        joint_vel={
            "left_wheel_joint": 0.0,
            "right_wheel_joint": 0.0,
        },
    ),
    actuators={
        # Velocity-controlled wheel actuators. Velocity limit is well above the
        # wheel speed implied by max linear/angular cmds:
        #   ω_wheel_max = (KOVA_MAX_LIN + KOVA_MAX_ANG * KOVA_WHEEL_BASE/2) / KOVA_WHEEL_RADIUS
        #              ≈ 18.7 rad/s
        "wheels": ImplicitActuatorCfg(
            joint_names_expr=["(left|right)_wheel_joint"],
            stiffness=0.0,
            damping=100.0,
            effort_limit_sim=20.0,
            velocity_limit_sim=25.0,
            armature=0.001,
            friction=0.0,
        ),
    },
)
