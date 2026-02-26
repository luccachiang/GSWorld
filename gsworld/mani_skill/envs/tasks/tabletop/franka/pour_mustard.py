from typing import Any, Dict, List, Union

import numpy as np
import sapien
import torch

from mani_skill import ASSET_DIR
from mani_skill.agents.robots.fetch.fetch import Fetch
from mani_skill.agents.robots.panda.panda import Panda
from gsworld.mani_skill.agents.robots.panda.fr3_umi import Fr3Umi
from mani_skill.agents.robots.panda.panda_wristcam import PandaWristCam
# from mani_skill.agents.robots.xmate3.xmate3 import Xmate3Robotiq
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.envs.utils.randomization.pose import random_quaternions
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import common, sapien_utils
from mani_skill.utils.building import actors
from mani_skill.utils.io_utils import load_json
from mani_skill.utils.registration import register_env
from gsworld.mani_skill.utils.scene_builder.table import TableSceneBuilderOffset
from mani_skill.utils.structs.actor import Actor
from mani_skill.utils.structs.pose import Pose
from mani_skill.utils.structs.types import GPUMemoryConfig, SimConfig
from gsworld.mani_skill.envs.tasks.real_fr3_env import RealFr3
from gsworld.constants import fr3_umi_task_init_qpos
from transforms3d.euler import quat2euler
from mani_skill.utils.geometry.rotation_conversions import matrix_to_euler_angles, quaternion_to_matrix

@register_env("PourMustardFr3Env-v1", max_episode_steps=50, asset_download_ids=["ycb"])
class PourMustardFr3Env(RealFr3):
    SUPPORTED_ROBOTS = ["fr3_umi", "fr3_umi_wrist435"]
    agent: Union[Panda, PandaWristCam, Fetch, Fr3Umi]
    pour_angle_thresh = torch.pi / 10
    pour_position_thresh = 0.15

    def __init__(
        self,
        *args,
        robot_uids="fr3_umi",
        robot_init_qpos_noise=0.02,
        num_envs=1,
        reconfiguration_freq=None,
        **kwargs,
    ):
        self.robot_init_qpos_noise = robot_init_qpos_noise
        self.model_id = None
        self.all_model_ids = np.array(
            list(
                load_json(ASSET_DIR / "assets/mani_skill2_ycb/info_pick_v0.json").keys()
            )
        )
        if reconfiguration_freq is None:
            if num_envs == 1:
                reconfiguration_freq = 1
            else:
                reconfiguration_freq = 0
        self.x_offset = 0.615
        self.has_poured = torch.zeros(num_envs, dtype=torch.bool)
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

        # Only use mustard bottle
        model_ids = ["006_mustard_bottle"]

        # Create the mustard bottle
        self._objs: List[Actor] = []
        for i, model_id in enumerate(model_ids):
            builder = actors.get_actor_builder(
                self.scene,
                id=f"ycb:{model_id}",
            )
            builder.initial_pose = sapien.Pose(p=[self.x_offset, 0.2, 0])
            builder.set_scene_idxs([i])
            self._objs.append(builder.build(name=f"{model_id}-{i}"))
            self.remove_from_state_dict_registry(self._objs[-1])
        self.obj = Actor.merge(self._objs, name=f"{model_id}")
        self.add_to_state_dict_registry(self.obj)
        
        # Create the box as the goal site
        self.goal_site = actors.build_box(
            self.scene,
            half_sizes=[0.14 * 0.5, 0.115 * 0.5, 0.015 * 0.5],
            color=[0, 1, 0, 1],
            name="bread_slice",
            initial_pose=sapien.Pose([self.x_offset, -0.2, 0]),
            # initial_pose=sapien.Pose(),
        )
        
        self.pouring_state = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self.bottle_height = 0.098
        self.goal_height = 0.010

    def _after_reconfigure(self, options: dict):
        self.object_zs = []
        for obj in self._objs:
            collision_mesh = obj.get_first_collision_mesh()
            self.object_zs.append(-collision_mesh.bounding_box.bounds[0, 2])
        self.object_zs = common.to_tensor(self.object_zs, device=self.device)
        self.has_poured = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)
            
            # Reset pouring state
            self.has_poured[env_idx] = False
            self.pouring_state[env_idx] = 0.0
            
            # Randomize bottle position
            bottle_xyz = torch.zeros((b, 3))
            bottle_xyz[:, 0] = torch.rand((b)) * 0.2 - 0.3 + self.x_offset
            bottle_xyz[:, 1] = torch.rand((b)) * 0.1 + 0.1
            bottle_xyz[:, 2] = self.bottle_height
            bottle_qs = random_quaternions(b, lock_x=True, lock_y=True, bounds=(0, np.pi * 0.5))
            # bottle_qs = None

            # Randomize box position
            box_xyz = torch.zeros((b, 3))
            box_xyz[:, 0] = torch.rand((b)) * 0.2 - 0.3 + self.x_offset
            box_xyz[:, 1] = torch.rand((b)) * 0.1 - 0.2
            box_xyz[:, 2] = self.goal_height / 2  # Half height above table since we're using half-sizes
            
            # Make sure bottle and box don't overlap
            min_distance = 0.15
            while torch.any(torch.linalg.norm(bottle_xyz[:, :2] - box_xyz[:, :2], axis=1) < min_distance):
                idx = torch.linalg.norm(bottle_xyz[:, :2] - box_xyz[:, :2], axis=1) < min_distance
                box_xyz[idx, 0] = torch.rand(idx.sum()) * 0.2 - 0.3 + self.x_offset
                box_xyz[idx, 1] = torch.rand(idx.sum()) * 0.1 - 0.2

            # Swap positions randomly
            if torch.rand(1) > 0.5:
                bottle_xyz[:, :2], box_xyz[:, :2] = box_xyz[:, :2].clone(), bottle_xyz[:, :2].clone()

            print(f"Bottle pose: {bottle_xyz} {bottle_qs}")
            print(f"Box pose: {box_xyz}")
            
            # Set positions
            self.obj.set_pose(Pose.create_from_pq(p=bottle_xyz, q=bottle_qs))
            self.goal_site.set_pose(Pose.create_from_pq(box_xyz))
            
            # Initialize robot arm
            if self.robot_uids in ["fr3_umi", "fr3_umi_wrist435", "fr3_umi_wrist435_cam_mount"]:
                self.agent.reset(fr3_umi_task_init_qpos)
                self.agent.robot.set_root_pose(sapien.Pose([self.x_offset-0.615, 0, 0]))
            else:
                raise NotImplementedError(self.robot_uids)

    def evaluate(self):
        # Get positions and orientations
        obj_pos = self.obj.pose.p
        obj_rot = self.obj.pose.q
        box_pos = self.goal_site.pose.p
        
        # Convert quaternion to euler angles to check bottle tilt
        bottle_euler = matrix_to_euler_angles(quaternion_to_matrix(obj_rot), convention="XYZ")
        
        # Calculate tilt angle (how much the bottle is tilted)
        # Assuming the bottle's main axis is aligned with the z-axis when upright
        bottle_tilt = torch.abs(bottle_euler[:, 0])  # x-axis rotation (pitch)
        
        # Check if bottle is above box
        bottle_to_box_xy = torch.linalg.norm(obj_pos[:, :2] - box_pos[:, :2], axis=1)
        is_above_box = bottle_to_box_xy < self.pour_position_thresh
        
        # Check if bottle is tilted enough for pouring
        is_tilted_enough = bottle_tilt > self.pour_angle_thresh
        
        # Update has_poured status (once poured, it stays true)
        currently_pouring = torch.logical_and(is_above_box, is_tilted_enough)
        self.has_poured = torch.logical_or(self.has_poured, currently_pouring)
        
        # Update the pouring state (visual feedback for the amount poured)
        self.pouring_state = torch.where(
            currently_pouring,
            torch.minimum(self.pouring_state + 0.1, torch.tensor(1.0)),  # Increment pouring state
            self.pouring_state
        )
        
        # Check if bottle is grasped
        is_grasped = self.agent.is_grasping(self.obj)
        
        # Check if objects are static
        is_robot_static = self.agent.is_static(0.2)
        is_obj_static = torch.logical_and(
            torch.linalg.norm(self.obj.linear_velocity, axis=1) < 0.05,
            torch.linalg.norm(self.obj.angular_velocity, axis=1) < 0.5
        )
        
        # Success: pouring occurred and everything is stable
        # Note: Not requiring release of the bottle
        # success = torch.logical_and(self.has_poured, is_obj_static)
        success = torch.logical_and(is_grasped, is_above_box)
        
        return {
            "is_grasped": is_grasped,
            "is_above_box": is_above_box,
            "is_tilted_enough": is_tilted_enough,
            "has_poured": self.has_poured,
            "is_robot_static": is_robot_static,
            "is_obj_static": is_obj_static,
            "pouring_state": self.pouring_state,
            "success": success,
        }

    def _get_obs_extra(self, info: Dict):
        obs = dict(
            tcp_pose=self.agent.tcp.pose.raw_pose,
            goal_pos=self.goal_site.pose.p,
            is_grasped=info["is_grasped"],
            has_poured=info["has_poured"],
            pouring_state=info["pouring_state"],
        )
        if "state" in self.obs_mode:
            obs.update(
                tcp_to_goal_pos=self.goal_site.pose.p - self.agent.tcp.pose.p,
                obj_pose=self.obj.pose.raw_pose,
                tcp_to_obj_pos=self.obj.pose.p - self.agent.tcp.pose.p,
                obj_to_goal_pos=self.goal_site.pose.p - self.obj.pose.p,
            )
        return obs

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        # Reaching reward - encourage getting close to the bottle
        tcp_to_obj_dist = torch.linalg.norm(
            self.obj.pose.p - self.agent.tcp.pose.p, axis=1
        )
        reaching_reward = 1 - torch.tanh(5 * tcp_to_obj_dist)
        reward = reaching_reward  # Base reward

        # Grasping reward - bonus for successfully grasping the bottle
        is_grasped = info["is_grasped"]
        grasp_reward = 1.0  # Fixed bonus for grasping
        reward += grasp_reward * is_grasped

        # If bottle is grasped, reward moving it above the box
        if torch.any(is_grasped):
            # Distance from bottle to box (xy only)
            obj_to_box_dist = torch.linalg.norm(
                self.goal_site.pose.p[:, :2] - self.obj.pose.p[:, :2], axis=1
            )
            positioning_reward = 1 - torch.tanh(5 * obj_to_box_dist)
            reward += positioning_reward * is_grasped * (~info["has_poured"])

        # Pouring reward - encourage tilting the bottle when it's above the box
        is_above_box = info["is_above_box"]
        is_tilted_enough = info["is_tilted_enough"]
        pouring_progress = info["pouring_state"]
        
        # Reward for being above box (prerequisite for pouring)
        above_box_reward = 0.5 * is_above_box * is_grasped * (~info["has_poured"])
        reward += above_box_reward
        
        # Reward for tilting the bottle when above box
        tilt_reward = 1.0 * is_above_box * is_tilted_enough * is_grasped
        reward += tilt_reward
        
        # Reward for successful pouring (one-time reward)
        pour_completion_reward = 2.0 * (info["has_poured"] & ~info["has_poured"].clone().detach())
        reward += pour_completion_reward
        
        # Stability reward after completing the task
        stability_reward = 1.0 * (info["has_poured"] & info["is_obj_static"] & info["is_robot_static"])
        reward += stability_reward
        
        # Success reward - large bonus for complete success
        success = info["success"]
        reward[success] = 6.0  # Maximum reward for complete success
        
        return reward

    def compute_normalized_dense_reward(
        self, obs: Any, action: torch.Tensor, info: Dict
    ):
        return self.compute_dense_reward(obs=obs, action=action, info=info) / 6.0