import argparse
import gymnasium as gym
import numpy as np
import sapien
from transforms3d.euler import euler2quat
import torch
import random

from gsworld.mani_skill.envs.tasks import PourMustardFr3Env
from gsworld.mani_skill.examples.motionplanning.franka.motionplanner import \
    FR3UmiMotionPlanningSolver
from mani_skill.examples.motionplanning.panda.utils import (
    compute_grasp_info_by_obb, get_actor_obb)
from mani_skill.utils.wrappers.record import RecordEpisode
from mani_skill.utils.structs.pose import Pose
from mani_skill.utils import common
from gsworld.utils.io_utils import read_hdf5_to_dict_recursively
from mani_skill.utils.geometry.rotation_conversions import axis_angle_to_quaternion

def solve(env: PourMustardFr3Env, seed=None, debug=False, vis=False, state_recover_file: str = None):
    
    env.reset(seed=seed)
    assert env.unwrapped.control_mode in [
        "pd_joint_pos",
        "pd_joint_pos_vel",
    ], env.unwrapped.control_mode
    
    planner = FR3UmiMotionPlanningSolver(
        env,
        debug=debug,
        vis=vis,
        base_pose=env.unwrapped.agent.robot.pose,
        visualize_target_grasp_pose=vis,
        print_env_info=False,
    )
    
    FINGER_LENGTH = 0.1
    obb = get_actor_obb(env.unwrapped.obj)

    # Compute grasp pose
    approaching = np.array([0, 0, -1])  # Approach from top
    target_closing = common.to_numpy(env.unwrapped.agent.tcp.pose.to_transformation_matrix()[0, :3, 1])
    grasp_info = compute_grasp_info_by_obb(
        obb,
        approaching=approaching,
        target_closing=target_closing,
        depth=FINGER_LENGTH,
    )
    closing, center = grasp_info["closing"], grasp_info["center"]
    grasp_pose = env.unwrapped.agent.build_grasp_pose(approaching, closing, center)

    # Search for a valid grasp pose by trying different angles
    angles = np.arange(0, np.pi * 2 / 3, np.pi / 2)
    angles = np.repeat(angles, 2)
    angles[1::2] *= -1
    for angle in angles:
        delta_pose = sapien.Pose(q=euler2quat(0, 0, angle))
        grasp_pose2 = grasp_pose * delta_pose
        res = planner.move_to_pose_with_screw(grasp_pose2, dry_run=True)
        if res == -1:
            continue
        grasp_pose = grasp_pose2
        break
    else:
        print("Fail to find a valid grasp pose")
        return -1

    # If not already grasped, plan and execute the grasping sequence
    if not env.unwrapped.get_info()['is_grasped']:
        
        reach_pose = grasp_pose * sapien.Pose([0, 0, -0.2])
        planner.move_to_pose_with_screw(reach_pose)

        # Grasp
        grasp_pose_umi = grasp_pose * Pose.create_from_pq(p=[-0.03, 0.0, -0.05], q=axis_angle_to_quaternion(torch.tensor([0.0, torch.pi/6, 0.0]))).sp
        planner.move_to_pose_with_screw(grasp_pose_umi)
        planner.close_gripper()

    # Check if grasping was successful
    if not env.unwrapped.get_info()['is_grasped']:
        print("Failed to grasp the mustard bottle")
        return -1

    # Lift bottle
    lift_pose = sapien.Pose([0, 0, 0.2]) * grasp_pose * Pose.create_from_pq(q=axis_angle_to_quaternion(torch.tensor([0.0, torch.pi/6, 0.0]))).sp
    planner.move_to_pose_with_screw(lift_pose)

    # Move above the box
    goal_pose = env.unwrapped.goal_site.pose * sapien.Pose([0, 0, 0.22])
    current_pos = env.unwrapped.obj.pose.p
    
    # Calculate position above the box
    offset = common.to_numpy((goal_pose.p - current_pos))[0]
    align_pose = sapien.Pose(lift_pose.p + offset, lift_pose.q)
    planner.move_to_pose_with_screw(align_pose)

    # Execute pour motion
    pour_pose = align_pose * Pose.create_from_pq(p=[-0.08, 0.0, 0.0], q=axis_angle_to_quaternion(torch.tensor([0.0, -torch.pi/3, 0.0]))).sp
    res = planner.move_to_pose_with_screw(pour_pose)
    
    # Close planner
    planner.close()
    
    # Check if pouring was successful
    has_poured = env.unwrapped.get_info()['has_poured']
    success = env.unwrapped.get_info()['success']
    
    return res