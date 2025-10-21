import copy
from typing import Any, Dict, List, Union

from gsworld.mani_skill.utils.building.actors.dtc import get_dtc_builder
import numpy as np
import sapien
import torch
from mani_skill.agents.robots.fetch.fetch import Fetch
from mani_skill.agents.robots.panda.panda import Panda
from mani_skill.agents.robots.panda.panda_wristcam import PandaWristCam
from mani_skill.envs.utils.randomization.pose import random_quaternions
from mani_skill.utils import common
from mani_skill.utils.building import actors
from mani_skill.utils.registration import register_env
from mani_skill.utils.structs.actor import Actor
from mani_skill.utils.structs.pose import Pose

from gsworld.mani_skill.envs.tasks.real_xarm_env import RealXArm6
from gsworld.mani_skill.utils.scene_builder.table import TableSceneBuilderOffset
from mani_skill.utils.geometry.rotation_conversions import axis_angle_to_quaternion, quaternion_multiply
from mani_skill.examples.motionplanning.panda.utils import (
    compute_grasp_info_by_obb, get_actor_obb)

WARNED_ONCE = False


@register_env(
    "AlignXArmEnv-v1", max_episode_steps=100, asset_download_ids=["ycb"]
)
class AlignXArmEnv(RealXArm6):

    SUPPORTED_ROBOTS = ["xarm6_uf_gripper_wrist435", "xarm6_uf_gripper"]
    agent: Union[Panda, PandaWristCam, Fetch]
    goal_thresh = 0.025

    def __init__(
        self,
        *args,
        robot_uids="xarm6_uf_gripper",
        robot_init_qpos_noise=0.02,
        num_envs=1,
        reconfiguration_freq=None,
        **kwargs,
    ):
        self.robot_init_qpos_noise = robot_init_qpos_noise
        reconfiguration_freq = 0
        self.x_offset = 0.615
        super().__init__(
            *args,
            robot_uids=robot_uids,
            reconfiguration_freq=reconfiguration_freq,
            num_envs=num_envs,
            **kwargs,
        )

    def _load_scene(self, options: dict):
        self.table_scene = TableSceneBuilderOffset(
            env=self, robot_init_qpos_noise=self.robot_init_qpos_noise
        )
        self.table_scene.build()

        # Add robot base at origin
        self.robot_base = actors.build_box(
            self.scene,
            half_sizes=[0.05, 0.05, 0.015],  # 10cm x 10cm x 1cm square base
            color=[0.3, 0.3, 0.3, 1],  # Dark gray color
            name="robot_base",
            initial_pose=sapien.Pose(
                [0, 0, 0.005]
            ),  # Position so top is at z=0.01 (1cm height)
            body_type="static",  # Make it static so it doesn't move
        )

        # Create tomato soup can first (following original logic)
        model_ids = ["005_tomato_soup_can"]
        distance = [0.2, -0.2]

        self._objs: List[Actor] = []
        for i, model_id in enumerate(model_ids):
            builder = actors.get_actor_builder(
                self.scene,
                id=f"ycb:{model_id}",
            )
            builder.initial_pose = sapien.Pose(p=[self.x_offset, distance[i], 0])
            builder.set_scene_idxs(list(range(self.num_envs)))
            if i == 0:
                self._objs.append(builder.build(name=f"{model_id}"))
            elif i == 1:
                self.tomato_can_site = builder.build(name=f"{model_id}-{i}")
            else:
                raise ValueError(
                    "Unexpected value. Only support 1 goal and 1 objects currently."
                )

        model_id = "Container_Toy_GratedParmesanCheese_C87B6E98"
        builder = get_dtc_builder(self.scene, id=model_id)
        builder.initial_pose = sapien.Pose(p=[self.x_offset, 0.2, 0])
        builder.set_scene_idxs(list(range(self.num_envs)))
        self.green = builder.build(name=f"dtc_green_can")

        self.tomato_can_site = self._objs[0]
        self._objs = [self.green]

        self.goal_height = 0.051
        self.obj_height = 0.05

        # Create a small red visual-only marker for goal position (no collision)
        self.goal_marker = actors.build_sphere(
            self.scene,
            radius=0.01,
            color=[1, 0, 0, 1],
            name="goal_marker",
            initial_pose=sapien.Pose([self.x_offset, 0, self.goal_height]),
            scene_idxs=list(range(self.num_envs)),
            body_type="kinematic",
            add_collision=False,
        )

    def _after_reconfigure(self, options: dict):
        self.object_zs = []
        for obj in self._objs:
            collision_mesh = obj.get_first_collision_mesh()
            self.object_zs.append(-collision_mesh.bounding_box.bounds[0, 2])
        self.object_zs = common.to_tensor(self.object_zs, device=self.device)

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):

            can_rotate_fix = quaternion_multiply(axis_angle_to_quaternion(axis_angle=torch.tensor([0.0, 0.0, torch.pi / 4])), axis_angle_to_quaternion(axis_angle=torch.tensor([torch.pi / 2, 0.0, 0.0])))

            b = len(env_idx)
            self.table_scene.initialize(env_idx)

            # Randomize green can (manipulated object) position
            obj_xyz_0 = torch.zeros((b, 3))
            _ = random_quaternions(b, lock_x=True, lock_y=True)
            obj_xyz_0[:, 0] = -0.125 + torch.rand(b) * 0.125 + self.x_offset + torch.rand(b) * 0.05 - 0.025
            obj_xyz_0[:, 1] = 0.1 + abs(torch.rand((b))) * 0.1 + 0.15 + torch.rand(b) * 0.05 - 0.025

            # Randomize tomato soup can (goal site) position
            goal_xyz_0 = torch.zeros((b, 3))
            goal_xyz_0[:, 0] = (
                torch.rand((b)) * 0.2
                - 0.25
                + self.x_offset
                + torch.rand(b) * 0.04
                - 0.02
                + 0.05
            )
            goal_xyz_0[:, 1] = (
                obj_xyz_0[:, 1]
                - 0.25
                - abs(torch.rand((b))) * 0.1
                + torch.rand(b) * 0.04
                - 0.02
            )
            goal_xyz_0[:, 2] = self.goal_height

            # Swap positions to put green can on right, tomato can on left
            obj_xyz_0[:, :2], goal_xyz_0[:, :2] = (
                goal_xyz_0[:, :2].clone(),
                obj_xyz_0[:, :2].clone(),
            )

            # Set object poses
            self._objs[0].set_pose(Pose.create_from_pq(p=obj_xyz_0, q=can_rotate_fix))

            # Sync visual goal marker to goal position (visual-only, no collision)
            self.goal_marker.set_pose(Pose.create_from_pq(p=goal_xyz_0))

            self.tomato_can_site.set_pose(Pose.create_from_pq(p=goal_xyz_0))
            self.goal_pos = goal_xyz_0
            self.goal_pos[:, 0] -= 0.1


            # Initialize robot arm
            if self.robot_uids in ["xarm6_uf_gripper_wrist435", "xarm6_uf_gripper"]:
                from gsworld.constants import robot_task_init_qpos

                self.agent.reset(robot_task_init_qpos[self.robot_uids])
                self.agent.robot.set_root_pose(
                    sapien.Pose([self.x_offset - 0.615, 0, 0.03])
                )
            else:
                raise NotImplementedError(self.robot_uids)
            
            self.init_tcp_q = copy.deepcopy(self.agent.tcp.pose.q)

    def evaluate(self):
        obj_pos = self._objs[0].pose.p
        goal_pos = self.goal_marker.pose.p

        # Check if object is within the goal region
        xy_offset = obj_pos[..., :2] - goal_pos[..., :2]
        half_xy_goal = (
            np.abs(
                self.tomato_can_site.get_first_collision_mesh().bounds[1, :]
                - self.tomato_can_site.get_first_collision_mesh().bounds[0, :]
            )[:2].max()
            * 0.5
        )
        is_xy_inside = torch.linalg.norm(xy_offset, axis=1) <= half_xy_goal

        # Check if robot is grasping the object
        is_grasped = self.agent.is_grasping(self._objs[0])

        # Check if robot is static (not moving)
        is_robot_static = self.agent.is_static(0.2)

        # Check if objects are static (not moving)
        is_obj_static = torch.logical_and(
            torch.linalg.norm(self._objs[0].linear_velocity, axis=1) < 0.05,
            torch.linalg.norm(self._objs[0].angular_velocity, axis=1) < 0.5,
        )
        is_goal_site_static = torch.logical_and(
            torch.linalg.norm(self.tomato_can_site.linear_velocity, axis=1) < 0.05,
            torch.linalg.norm(self.tomato_can_site.angular_velocity, axis=1) < 0.5,
        )
        is_all_static = torch.logical_and(is_obj_static, is_goal_site_static)

        # Success: object in goal region, not grasped, everything static
        success = torch.logical_and(is_xy_inside, ~is_grasped)
        success = torch.logical_and(success, is_all_static)

        return {
            "is_grasped_0": is_grasped,
            "is_obj_in_box": is_xy_inside,
            "is_robot_static": is_robot_static,
            "is_obj_static": is_all_static,
            "is_goal_site_static": is_goal_site_static,
            "success": success,
        }

    def _get_obs_extra(self, info: Dict):
        obs = dict(
            tcp_pose=self.agent.tcp.pose.raw_pose,
            goal_pos=self.goal_marker.pose.p,
            is_grasped=info["is_grasped_0"],
        )
        obs.update(
            tcp_to_goal_pos=self.goal_marker.pose.p - self.agent.tcp.pose.p,
            obj_pose=self._objs[0].pose.raw_pose,
            tcp_to_obj_pos=self._objs[0].pose.p - self.agent.tcp.pose.p,
            obj_to_goal_pos=self.goal_marker.pose.p - self._objs[0].pose.p,
        )
        return obs

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        is_grasped = info["is_grasped_0"]
        is_obj_in_box = info["is_obj_in_box"]

        # Reaching reward - encourage getting close to the object
        tcp_to_obj_dist = torch.linalg.norm(
            self._objs[0].pose.p - self.agent.tcp.pose.p, axis=1
        )
        reaching_reward = 1 - torch.tanh(5 * tcp_to_obj_dist)
        reward = copy.deepcopy(reaching_reward)

        # Grasping reward - bonus for successfully grasping the object
        grasp_reward = 1.0
        reward += grasp_reward * is_grasped
        
        # Lift reward - bonus for lifting the object
        lift_reward = 1.0 * (is_grasped & (self._objs[0].pose.p[..., 2] > 0.05))
        reward += lift_reward

        # Transport reward - if object is grasped, reward moving it toward the goal
        obj_to_goal_dist = torch.linalg.norm(
            self.goal_marker.pose.p - self._objs[0].pose.p, axis=1
        )
        transport_reward = 1 - torch.tanh(obj_to_goal_dist)
        reward += 3 * transport_reward * (is_grasped & ~is_obj_in_box)

        # Placement reward - bonus for getting object in goal region
        placement_reward = 1.0
        reward += 2 * placement_reward * is_obj_in_box

        # Release reward - encourage releasing the object when it's in the goal region
        release_reward = 4.0 * (is_obj_in_box & ~is_grasped)
        reward += release_reward

        # Stability reward - encourage stable final state
        is_obj_static = info["is_obj_static"]
        is_robot_static = info["is_robot_static"]
        stability_reward = 1.0 * (is_obj_in_box & is_obj_static & is_robot_static)
        reward += stability_reward

        # Success reward - large bonus for complete success
        success = info["success"]
        reward[success] += 20.0

        return reward

    def compute_normalized_dense_reward(
        self, obs: Any, action: torch.Tensor, info: Dict
    ):
        return self.compute_dense_reward(obs=obs, action=action, info=info) / 10.0
