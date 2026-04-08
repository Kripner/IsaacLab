# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import warp as wp

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import combine_frame_transforms

if TYPE_CHECKING:
    from isaaclab.assets import RigidObject
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.sensors import FrameTransformer


def _object_held_mask(
    env: ManagerBasedRLEnv,
    minimal_height: float,
    grip_distance: float,
    object_cfg: SceneEntityCfg,
    ee_frame_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Boolean mask: object is above ``minimal_height`` AND end-effector is
    within ``grip_distance`` of the object.

    This prevents reward hacking by throwing the object: a thrown cube is
    "lifted" but not "held", so the bonus stops the moment it leaves the
    gripper.
    """
    object: RigidObject = env.scene[object_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    cube_pos_w = wp.to_torch(object.data.root_pos_w)
    ee_w = wp.to_torch(ee_frame.data.target_pos_w)[..., 0, :]
    is_lifted = cube_pos_w[:, 2] > minimal_height
    is_held = torch.linalg.norm(cube_pos_w - ee_w, dim=1) < grip_distance
    return is_lifted & is_held


def object_is_lifted(
    env: ManagerBasedRLEnv,
    minimal_height: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    grip_distance: float = 0.06,
) -> torch.Tensor:
    """Reward the agent for lifting the object above the minimal height while
    holding it with the gripper.

    The bonus is only given when the end-effector is within ``grip_distance``
    of the object. Without this gating the agent learns to throw the cube
    instead of grip-and-lift, since a thrown cube farms the lift bonus while
    airborne.
    """
    return _object_held_mask(
        env, minimal_height, grip_distance, object_cfg, ee_frame_cfg
    ).float()


def object_ee_distance(
    env: ManagerBasedRLEnv,
    std: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Reward the agent for reaching the object using tanh-kernel."""
    # extract the used quantities (to enable type-hinting)
    object: RigidObject = env.scene[object_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    # Target object position: (num_envs, 3)
    cube_pos_w = wp.to_torch(object.data.root_pos_w)
    # End-effector position: (num_envs, 3)
    ee_w = wp.to_torch(ee_frame.data.target_pos_w)[..., 0, :]
    # Distance of the end-effector to the object: (num_envs,)
    object_ee_distance = torch.linalg.norm(cube_pos_w - ee_w, dim=1)

    return 1 - torch.tanh(object_ee_distance / std)


def object_goal_distance(
    env: ManagerBasedRLEnv,
    std: float,
    minimal_height: float,
    command_name: str,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    grip_distance: float = 0.06,
) -> torch.Tensor:
    """Reward the agent for tracking the goal pose using tanh-kernel.

    Gated on the gripper actually holding the object (cube above
    ``minimal_height`` AND end-effector within ``grip_distance``). Without
    this gating, a thrown cube passing through the goal region farms partial
    credit, encouraging throwing instead of placement.
    """
    # extract the used quantities (to enable type-hinting)
    robot: RigidObject = env.scene[robot_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    command = env.command_manager.get_command(command_name)
    # compute the desired position in the world frame
    des_pos_b = command[:, :3]
    des_pos_w, _ = combine_frame_transforms(
        wp.to_torch(robot.data.root_pos_w), wp.to_torch(robot.data.root_quat_w), des_pos_b
    )
    # distance of the end-effector to the object: (num_envs,)
    object_pos_w = wp.to_torch(object.data.root_pos_w)
    distance = torch.linalg.norm(des_pos_w - object_pos_w, dim=1)
    # rewarded only when the gripper actually holds the object
    held = _object_held_mask(env, minimal_height, grip_distance, object_cfg, ee_frame_cfg)
    return held.float() * (1 - torch.tanh(distance / std))
