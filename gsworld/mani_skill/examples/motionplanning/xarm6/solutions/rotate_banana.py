import argparse
import gymnasium as gym
import numpy as np
import sapien
from transforms3d.euler import euler2quat
import torch
import random

from mani_skill.utils import common

from gsworld.mani_skill.envs.tasks.tabletop import BananaRotationXArmEnv
from gsworld.mani_skill.examples.motionplanning.xarm6.motionplanner import \
    XArmMotionPlanningSolver
from mani_skill.examples.motionplanning.panda.utils import (
    compute_grasp_info_by_obb, get_actor_obb)
from gsworld.utils.io_utils import read_hdf5_to_dict_recursively
from gsworld.mani_skill.utils.wrappers import GSWorldWrapper
from mani_skill.utils.geometry.rotation_conversions import quaternion_to_axis_angle, quaternion_multiply, axis_angle_to_quaternion

def solve(env: BananaRotationXArmEnv | GSWorldWrapper, seed=None, debug=False, vis=False, state_recover_file: str = None):
    env.reset(seed=seed)
    
    planner = XArmMotionPlanningSolver(
        env,
        debug=debug,
        vis=vis,
        base_pose=env.unwrapped.agent.robot.pose,
        visualize_target_grasp_pose=vis,
        print_env_info=False,
    )

    FINGER_LENGTH = 0.025
    env = env.unwrapped

    obb = get_actor_obb(env.banana)

    approaching = np.array([0, 0, -1])
    
    target_closing = env.agent.tcp.pose.to_transformation_matrix()[0, :3, 1].cpu().numpy()
    
    
    grasp_info = compute_grasp_info_by_obb(
        obb,
        approaching=approaching,
        target_closing=target_closing,
        depth=FINGER_LENGTH,
    )
    closing, center = grasp_info["closing"], grasp_info["center"]
    grasp_pose = env.agent.build_grasp_pose(approaching, closing, center)
    
    res = planner.open_gripper()

    # -------------------------------------------------------------------------- #
    # Reach
    # -------------------------------------------------------------------------- #
    reach_pose = grasp_pose * sapien.Pose([0, 0, -0.1])
    planner.move_to_pose_with_screw(reach_pose)

    # -------------------------------------------------------------------------- #
    # Grasp
    # -------------------------------------------------------------------------- #
    grasp_pose_final = grasp_pose
    planner.move_to_pose_with_screw(grasp_pose_final)
    planner.close_gripper()
    # -------------------------------------------------------------------------- #
    # Lift up above red can
    # -------------------------------------------------------------------------- #
    lift_pose = grasp_pose * sapien.Pose([0, 0, -0.1])

    res = planner.move_to_pose_with_screw(lift_pose)
    if res == -1:
        planner.close()
        return res

    # -------------------------------------------------------------------------- #
    # Rotate
    # -------------------------------------------------------------------------- #
    rotate_pose = lift_pose * sapien.Pose(q=axis_angle_to_quaternion(axis_angle=torch.tensor([0.0, 0.0, -torch.pi / 3])))
    res = planner.move_to_pose_with_screw(rotate_pose)
    if res == -1:
        planner.close()
        return res
    
    # -------------------------------------------------------------------------- #
    # Drop the object
    # -------------------------------------------------------------------------- #
    res = planner.open_gripper()
    if res == -1:
        planner.close()
        return res

    planner.close()
    return res
