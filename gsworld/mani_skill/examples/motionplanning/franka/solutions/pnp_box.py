import argparse
import gymnasium as gym
import numpy as np
import sapien
from transforms3d.euler import euler2quat
import torch
import random

from gsworld.mani_skill.envs.tasks import PnpBoxFr3Env
from gsworld.mani_skill.examples.motionplanning.franka.motionplanner import \
    FR3UmiMotionPlanningSolver
from mani_skill.examples.motionplanning.panda.utils import (
    compute_grasp_info_by_obb, get_actor_obb)
from mani_skill.utils.wrappers.record import RecordEpisode
from mani_skill.utils.structs.pose import Pose
from mani_skill.utils import common
from gsworld.utils.io_utils import read_hdf5_to_dict_recursively

def solve(env: PnpBoxFr3Env, seed=None, debug=False, vis=False, state_recover_file: str = None):
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
    env = env.unwrapped
    obb = get_actor_obb(env.obj)

    approaching = np.array([0, 0, -1])
    target_closing = common.to_numpy(env.agent.tcp.pose.to_transformation_matrix()[0, :3, 1])
    grasp_info = compute_grasp_info_by_obb(
        obb,
        approaching=approaching,
        target_closing=target_closing,
        depth=FINGER_LENGTH,
    )
    closing, center = grasp_info["closing"], grasp_info["center"]
    grasp_pose = env.agent.build_grasp_pose(approaching, closing, center)

    # Search a valid pose
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

    # add randomization to motion planning
    if seed is not None:
        torch.manual_seed(seed)
    if not env.unwrapped.get_info()['is_grasped']:
        # -------------------------------------------------------------------------- #
        # Reach
        # -------------------------------------------------------------------------- #
        reach_pose = grasp_pose * sapien.Pose([0, 0, -0.2]) * Pose.create_from_pq(p=torch.randn(3) * 0.01).sp
        planner.move_to_pose_with_screw(reach_pose)

        # -------------------------------------------------------------------------- #
        # Grasp
        # -------------------------------------------------------------------------- #
        grasp_pose_umi = grasp_pose * sapien.Pose([0, 0, -0.1])
        planner.move_to_pose_with_screw(grasp_pose_umi)
        planner.close_gripper()

    # -------------------------------------------------------------------------- #
    # Lift
    # -------------------------------------------------------------------------- #
    lift_pose = Pose.create_from_pq(p=torch.randn(3) * 0.01).sp * sapien.Pose([0, 0, 0.2]) * grasp_pose
    if env.unwrapped.get_info()['is_grasped']:
        planner.move_to_pose_with_screw(lift_pose)

    # -------------------------------------------------------------------------- #
    # Place
    # -------------------------------------------------------------------------- #
    goal_pose = env.goal_site.pose * sapien.Pose([0, 0, 0.18]) * Pose.create_from_pq(p=torch.randn(3) * 0.01).sp
    offset = common.to_numpy((goal_pose.p - env.obj.pose.p))[0]
    align_pose = sapien.Pose(lift_pose.p + offset, lift_pose.q)
    planner.move_to_pose_with_screw(align_pose)

    planner.open_gripper()

    # lift eef
    reach_pose2 = align_pose * sapien.Pose([0, 0, -0.1])
    res = planner.move_to_pose_with_screw(reach_pose2)

    planner.close()
    return res
