from typing import Any, Dict, List, Union

import numpy as np
import sapien
import torch

from mani_skill import ASSET_DIR
from mani_skill.agents.robots.fetch.fetch import Fetch
from mani_skill.agents.robots.panda.panda import Panda
from mani_skill.agents.robots.panda.panda_wristcam import PandaWristCam
from mani_skill.utils import common
from mani_skill.utils.building import actors
from mani_skill.utils.registration import register_env
from gsworld.mani_skill.utils.scene_builder.table import TableSceneBuilderOffset
from mani_skill.utils.structs.actor import Actor
from mani_skill.utils.structs.pose import Pose
from gsworld.mani_skill.envs.tasks.real_xarm_env import RealXArm6
from mani_skill.utils.geometry.rotation_conversions import quaternion_to_axis_angle, quaternion_multiply, axis_angle_to_quaternion
from mani_skill.examples.motionplanning.panda.utils import (
    compute_grasp_info_by_obb, get_actor_obb)
from gsworld.mani_skill.utils.building.actors.ycb import get_ycb_builder

WARNED_ONCE = False

@register_env("BananaRotationXArmEnv-v1", max_episode_steps=100, asset_download_ids=["ycb"])
class BananaRotationXArmEnv(RealXArm6):

    SUPPORTED_ROBOTS = ["xarm6_uf_gripper", "xarm6_uf_gripper_wrist435"]
    agent: Union[Panda, PandaWristCam, Fetch]
    rotation_thresh = 30

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
            initial_pose=sapien.Pose([0, 0, 0.005]),  # Position so top is at z=0.01 (1cm height)
            body_type="static"  # Make it static so it doesn't move
        )

        model_id = "011_banana"
        builder = get_ycb_builder(
            self.scene,
            id=f"{model_id}",
        )
        self.banana_init_q = quaternion_multiply(axis_angle_to_quaternion(axis_angle=torch.tensor([0.0, 0.0, 0.0])), axis_angle_to_quaternion(torch.tensor([0.0, 0.0, torch.pi / 2])))
        builder.initial_pose = sapien.Pose(p=[self.x_offset, 0.2, 0], q=self.banana_init_q) # gq: make sure doesnt intersect
        builder.set_scene_idxs(list(range(self.num_envs)))
        self.banana = builder.build(name=f"{model_id}")
        self.banana.set_mass(0.01)

        # Set up the objects list
        self._objs = [self.banana]  # Objects list contains the banana

        self.obj_height = 0.019  # Height to place banana (accounting for its half-length)

    def _after_reconfigure(self, options: dict):
        self.object_zs = []
        for obj in self._objs:
            collision_mesh = obj.get_first_collision_mesh()
            self.object_zs.append(-collision_mesh.bounding_box.bounds[0, 2])
        self.object_zs = common.to_tensor(self.object_zs, device=self.device)

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):            
            b = len(env_idx)
            self.table_scene.initialize(env_idx)
            
            # Randomize banana position
            banana_xyz = torch.zeros((b, 3), device=self.device)
            banana_xyz[:, 0] = self.x_offset + torch.rand(b, device=self.device) * 0.2 - 0.3
            banana_xyz[:, 1] = torch.rand(b, device=self.device) * 0.2 - 0.1
            banana_xyz[:, 2] = self.obj_height

            target_q = quaternion_multiply(
                self.banana_init_q, 
                axis_angle_to_quaternion(torch.tensor([0.0, 0.0, -torch.pi / 3], device=self.device))
            )
            if target_q.ndim == 1:
                target_q_batch = target_q.unsqueeze(0).repeat(b, 1)
            else:
                target_q_batch = target_q

            # --- Initialize global buffer (only once)
            if not hasattr(self, "target_pose_p"):
                self.target_pose_p = torch.zeros((self.num_envs, 3), device=self.device)
                self.target_pose_q = torch.zeros((self.num_envs, 4), device=self.device)

            # Update reset envs slice, not overwrite the whole batch
            self.target_pose_p[env_idx] = banana_xyz
            self.target_pose_q[env_idx] = target_q_batch

            # If still need Pose object, rebuild once (full batch)
            self.target_pose = Pose.create_from_pq(
                p=self.target_pose_p, q=self.target_pose_q
            )

            # Set banana pose
            self.banana.set_pose(Pose.create_from_pq(p=banana_xyz, q=self.banana_init_q))
            
            if self.robot_uids in ["xarm6_uf_gripper_wrist435", "xarm6_uf_gripper"]:
                from gsworld.constants import robot_task_init_qpos
                self.agent.reset(robot_task_init_qpos[self.robot_uids])
                self.agent.robot.set_root_pose(sapien.Pose([self.x_offset-0.615, 0, 0.03]))
            else:
                raise NotImplementedError(self.robot_uids)
            
            # build grasping pose
            self.FINGER_LENGTH = 0.025
            self.obb = get_actor_obb(self.banana)
            self.approaching = np.array([0, 0, -1])
            self.target_closing = self.agent.tcp.pose.to_transformation_matrix()[0, :3, 1].cpu().numpy()
            # we can build a simple grasp pose using this information for Panda
            self.grasp_info = compute_grasp_info_by_obb(
                self.obb,
                approaching=self.approaching,
                target_closing=self.target_closing,
                depth=self.FINGER_LENGTH,
            )
            self.closing, self.center = self.grasp_info["closing"], self.grasp_info["center"]
            self.grasp_pose = self.agent.build_grasp_pose(self.approaching, self.closing, self.center)
            self.lift_pose = self.grasp_pose * sapien.Pose([0, 0, -0.1])
            self.rotate_pose = self.lift_pose * sapien.Pose(q=common.to_numpy(axis_angle_to_quaternion(axis_angle=torch.tensor([0.0, 0.0, -torch.pi / 3]))))

    def _get_rotation_difference(self, current_quat, target_quat):
        """Calculate the angular difference between two quaternions in degrees"""
        # Normalize quaternions
        current_quat = current_quat / torch.linalg.norm(current_quat, dim=-1, keepdim=True)
        target_quat = target_quat / torch.linalg.norm(target_quat, dim=-1, keepdim=True)
        
        # Calculate relative rotation
        current_quat_inv = current_quat.clone()
        current_quat_inv[..., :3] *= -1  # Conjugate for inverse
        
        # Quaternion multiplication: q_rel = target * current_inv
        w1, x1, y1, z1 = target_quat[..., 3], target_quat[..., 0], target_quat[..., 1], target_quat[..., 2]
        w2, x2, y2, z2 = current_quat_inv[..., 3], current_quat_inv[..., 0], current_quat_inv[..., 1], current_quat_inv[..., 2]
        
        rel_w = w1*w2 - x1*x2 - y1*y2 - z1*z2
        
        # Convert to angle (in radians then degrees)
        angle_rad = 2 * torch.acos(torch.clamp(torch.abs(rel_w), 0, 1))
        angle_deg = angle_rad * 180.0 / torch.pi
        
        return angle_deg

    def evaluate(self):
        banana_pose = self._objs[0].pose
        target_pose = self.target_pose
        
        # Check rotational alignment
        rotation_diff = self._get_rotation_difference(banana_pose.q, self.banana_init_q)
        is_rotation_correct = rotation_diff > self.rotation_thresh
        
        # Check if banana is at table height (not dropped or floating)
        is_at_table_height = torch.logical_and(
            torch.abs(banana_pose.p[..., 2] - self.obj_height) <= 0.05, 
            torch.abs(banana_pose.p[..., 2] - self.obj_height) >= 0.02
        )

        # Check if robot is grasping the banana
        is_grasped = self.agent.is_grasping(self._objs[0], max_angle=180)

        # Check if robot is static (not moving)
        is_robot_static = self.agent.is_static(0.2)
        
        # Check if banana is static (not moving)
        is_banana_static = torch.logical_and(
            torch.linalg.norm(self._objs[0].linear_velocity, axis=1) < 0.05,
            torch.linalg.norm(self._objs[0].angular_velocity, axis=1) < 0.5
        )

        # Success: correct rotation, on table, not grasped, everything static
        success = torch.logical_and(is_rotation_correct, is_at_table_height)
        success = torch.logical_and(success, ~is_grasped)
        # success = torch.logical_and(success, is_banana_static)
        success = torch.logical_and(success, is_robot_static)
        
        return {
            "is_grasped": is_grasped,
            "is_rotation_correct": is_rotation_correct,
            "is_at_table_height": is_at_table_height,
            "is_robot_static": is_robot_static,
            "is_banana_static": is_banana_static,
            "rotation_diff_degrees": rotation_diff,
            "success": success,
        }

    def _get_obs_extra(self, info: Dict):
        obs = dict(
            tcp_pose=self.agent.tcp.pose.raw_pose,
            target_rotation=quaternion_to_axis_angle(self.target_pose.q),
            current_rotation=quaternion_to_axis_angle(self._objs[0].pose.q),
            is_grasped=info["is_grasped"],
            rotation_diff=info["rotation_diff_degrees"],
            goal_pos=self.target_pose.p,
        )
        obs.update(
            tcp_to_goal_pos=common.to_tensor(self.grasp_pose.p, device=self.device) - self.agent.tcp.pose.p,
            obj_pose=self.banana.pose.raw_pose,
            tcp_to_obj_pos=self.banana.pose.p - self.agent.tcp.pose.p,
            obj_to_goal_pos=common.to_tensor(self.rotate_pose.p, device=self.device) - self.banana.pose.p,
        )
        return obs

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        """
        Dense reward function for banana rotation task.
        The task involves: pick up banana -> rotate it 60 degrees around z-axis -> place it down
        """
        
        tcp_to_banana_dist = torch.linalg.norm(
            self.banana.pose.p - self.agent.tcp.pose.p, axis=1
        )
        reaching_reward = torch.exp(-50 * tcp_to_banana_dist)
        reward = reaching_reward
        
        is_grasped = info["is_grasped"]
        reward += is_grasped
        
        rot_to_target = torch.linalg.norm(
            self.banana.pose.raw_pose - self.target_pose.raw_pose, axis=1
        )
        rotation_reward = torch.exp(-10 * rot_to_target)
        reward += rotation_reward * is_grasped
        
        reward[info["success"]] += 1
        
        return reward

    def compute_normalized_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        # The normalization factor should be the maximum possible reward.
        # This is a good approximation based on the sum of bonuses.
        return self.compute_dense_reward(obs=obs, action=action, info=info) / 5.0