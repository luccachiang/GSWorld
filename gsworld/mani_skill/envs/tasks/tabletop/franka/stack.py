from typing import Any, Dict, List, Union

import numpy as np
import sapien
import torch

from mani_skill import ASSET_DIR
from mani_skill.agents.robots.fetch.fetch import Fetch
from mani_skill.agents.robots.panda.panda import Panda
from gsworld.mani_skill.agents.robots.panda.fr3_umi import Fr3Umi
from mani_skill.agents.robots.panda.panda_wristcam import PandaWristCam
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.envs.utils.randomization.pose import random_quaternions
from mani_skill.utils.geometry.rotation_conversions import axis_angle_to_quaternion, quaternion_multiply
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import common, sapien_utils
from mani_skill.utils.building import actors
from gsworld.mani_skill.utils.building.actors.dtc import get_dtc_builder
from mani_skill.utils.io_utils import load_json
from mani_skill.utils.registration import register_env
from gsworld.mani_skill.utils.scene_builder.table import TableSceneBuilderOffset
from mani_skill.utils.structs.actor import Actor
from mani_skill.utils.structs.pose import Pose
from mani_skill.utils.structs.types import GPUMemoryConfig, SimConfig
from gsworld.mani_skill.envs.tasks.real_fr3_env import RealFr3
from gsworld.constants import fr3_umi_task_init_qpos

WARNED_ONCE = False

@register_env("StackFr3Env-v1", max_episode_steps=100, asset_download_ids=["ycb"])
class StackFr3Env(RealFr3):
    SUPPORTED_ROBOTS = ["fr3_umi", "fr3_umi_wrist435"]
    agent: Union[Panda, PandaWristCam, Fetch, Fr3Umi]
    goal_thresh = 0.025

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
        global WARNED_ONCE
        self.table_scene = TableSceneBuilderOffset(
            env=self, robot_init_qpos_noise=self.robot_init_qpos_noise
        )
        self.table_scene.build()

        model_ids = ["005_tomato_soup_can"]
        if (
            self.num_envs > 1
            and self.num_envs < len(self.all_model_ids)
            and self.reconfiguration_freq <= 0
            and not WARNED_ONCE
        ):
            WARNED_ONCE = True
            print(
                """There are less parallel environments than total available models to sample.
                Not all models will be used during interaction even after resets unless you call env.reset(options=dict(reconfigure=True))
                or set reconfiguration_freq to be >= 1."""
            )

        self._objs: List[Actor] = []
        self.obj_heights = []
        
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
            self.goal_site = builder.build(name=f"{model_id}")

        model_id = "Can_Toy_TomatoSauce_D4D3CCC0"
        builder = get_dtc_builder(self.scene, id=model_id)
        builder.initial_pose = sapien.Pose(p=[self.x_offset, -0.2, 0])
        builder.set_scene_idxs(list(range(self.num_envs)))
        self._objs.append(builder.build(name=f"dtc_red_tomato_can_fr3"))

        self.goal_height = 0.051  # red can height
        self.obj_height = 0.05   # green can height

    def _after_reconfigure(self, options: dict):
        self.object_zs = []
        for obj in self._objs:
            collision_mesh = obj.get_first_collision_mesh()
            self.object_zs.append(-collision_mesh.bounding_box.bounds[0, 2])
        self.object_zs = common.to_tensor(self.object_zs, device=self.device)

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            from transforms3d.euler import euler2quat
            import math
            
            red_can_rotate_fix = quaternion_multiply(
                axis_angle_to_quaternion(axis_angle=torch.tensor([0.0, 0.0, torch.pi / 4])), 
                axis_angle_to_quaternion(axis_angle=torch.tensor([torch.pi / 2, 0.0, 0.0]))
            )
            
            b = len(env_idx)
            self.table_scene.initialize(env_idx)
            # add noise
            obj_xyz_0 = torch.zeros((b, 3))
            # torch rand is [0, 1)
            obj_xyz_0[:, 0] = -0.125 + torch.rand(b) * 0.125 + self.x_offset # 0 to -0.25
            obj_xyz_0[:, 1] = 0.1 + abs(torch.rand((b))) * 0.1    #-0.2 to 0
            obj_xyz_0[:, 2] = self.obj_height # self.object_zs[env_idx] TODO
            obj_qs_0 = random_quaternions(b, lock_x=True, lock_y=True)
 

            goal_xyz_0 = torch.zeros((b, 3))
            goal_xyz_0[:, 0] = torch.rand((b)) * 0.2 - 0.25 + self.x_offset # 
            goal_xyz_0[:, 1] =  obj_xyz_0[:, 1] - 0.15 -  abs(torch.rand((b))) * 0.1  # seperate by 0.15 to 0.25
            goal_xyz_0[:, 2] = self.goal_height # self.goal_height # * 0.5 # put it on the table, hard code from human render mode


            print(obj_xyz_0[:, 1] - goal_xyz_0[:, 1], "\n\n\n")
            goal_qs = random_quaternions(b, lock_x=True, lock_y=True)
            obj_xyz_0[:, :2], goal_xyz_0[:, :2] = goal_xyz_0[:, :2].clone(), obj_xyz_0[:, :2].clone()

            def convert_to_sapien_pose(p, q):
                if p.shape[0] != 1:
                    raise NotImplementedError("This function needs to be implemented.")
                p = p[0]
                q = q[0]
                return sapien.Pose(p=[p[0], p[1], p[2]], q=[q[0], q[1], q[2], q[3]])
            
            print(f"obj pose {obj_xyz_0} {obj_qs_0}, goal pose {goal_xyz_0}")
            
            self._objs[0].set_pose(Pose.create_from_pq(p=obj_xyz_0, q=red_can_rotate_fix))
            self.goal_site.set_pose(Pose.create_from_pq(p=goal_xyz_0))  # Red can (cylinder)

            if self.robot_uids in ["panda", "panda_wristcam"]:
                qpos = np.array([0.0, 0, 0, -np.pi * 2 / 3, 0, np.pi * 2 / 3, np.pi / 4, 0.04, 0.04])
                qpos[:-2] += self._episode_rng.normal(
                    0, self.robot_init_qpos_noise, len(qpos) - 2
                )
                self.agent.reset(qpos)
                self.agent.robot.set_root_pose(sapien.Pose([self.x_offset-0.615, 0, 0]))
            elif self.robot_uids in ["fr3_umi", "fr3_umi_wrist435", "fr3_umi_wrist435_cam_mount"]:
                self.agent.reset(fr3_umi_task_init_qpos)
                self.agent.robot.set_root_pose(sapien.Pose([self.x_offset-0.615, 0, 0]))
            else:
                raise NotImplementedError(self.robot_uids)

    def evaluate(self):
        # Get positions
        obj_pos_0 = self._objs[0].pose.p


        goal_box_pos = self.goal_site.pose.p
        
        # Check if object is inside the box
        # The box has dimensions defined by half_size (x,y) and height (z)
        # Check if object is within the x-y bounds of the box
        xy_offset_0 = obj_pos_0[..., :2] - goal_box_pos[..., :2]
        half_xy_goal = np.abs(self.goal_site.get_first_collision_mesh().bounds[1, :] - self.goal_site.get_first_collision_mesh().bounds[0, :])[:2].max() * 0.5
        is_xy_inside = torch.linalg.norm(xy_offset_0, axis=1) <= half_xy_goal - 0.02  # Allow some margin
        
        # Object is inside box if both xy and z conditions are met
        # is_obj_in_box = torch.logical_and(is_xy_inside, is_z_inside)
        is_obj_in_box = is_xy_inside # we dont check z axis
        
        # Check if robot is grasping the object
        is_grasped_0 = self.agent.is_grasping(self._objs[0])

        # Check if robot is static (not moving)
        is_robot_static = self.agent.is_static(0.2)
        
        # Check if object is static (not moving)
        is_obj_0_static = torch.logical_and(
            torch.linalg.norm(self._objs[0].linear_velocity, axis=1) < 0.05,
            torch.linalg.norm(self._objs[0].angular_velocity, axis=1) < 0.5
        )
        is_goal_site_static = torch.logical_and(
            torch.linalg.norm(self.goal_site.linear_velocity, axis=1) < 0.05,
            torch.linalg.norm(self.goal_site.angular_velocity, axis=1) < 0.5
        )
        is_obj_static = torch.logical_and(is_obj_0_static, is_goal_site_static)


        success = torch.logical_and(is_obj_in_box, ~is_grasped_0)
        success = torch.logical_and(success, is_obj_static)
        
        return {
            "is_grasped_0": is_grasped_0,
            "is_obj_in_box": is_obj_in_box,
            "is_robot_static": is_robot_static,
            "is_obj_static": is_obj_static,
            "is_goal_site_static": is_goal_site_static,
            "success": success,
        }

    def _get_obs_extra(self, info: Dict):
        obs = dict(
            tcp_pose=self.agent.tcp.pose.raw_pose,
            goal_pos=self.goal_site.pose.p,
            is_grasped_0=info["is_grasped_0"],
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
        # Reaching reward - encourage getting close to the object
        tcp_to_obj_dist = torch.linalg.norm(
            self.obj.pose.p - self.agent.tcp.pose.p, axis=1
        )
        reaching_reward = 1 - torch.tanh(5 * tcp_to_obj_dist)
        reward = reaching_reward  # Base reward

        # Grasping reward - bonus for successfully grasping the object
        is_grasped = info["is_grasped_0"]
        grasp_reward = 1.0  # Fixed bonus for grasping
        reward += grasp_reward * is_grasped

        # If object is grasped, reward moving it toward the box
        if torch.any(is_grasped):
            obj_to_box_dist = torch.linalg.norm(
                self.goal_site.pose.p - self.obj.pose.p, axis=1
            )
            transport_reward = 1 - torch.tanh(5 * obj_to_box_dist)
            reward += transport_reward * is_grasped

        # Placement reward - bonus for getting object inside the box
        is_obj_in_box = info["is_obj_in_box"]
        placement_reward = 1.0  # Fixed bonus for successful placement
        reward += placement_reward * is_obj_in_box
        
        # Release reward - encourage releasing the object when it's in the box
        release_reward = 1.0 * (is_obj_in_box & ~is_grasped)
        reward += release_reward
        
        # Stability reward - encourage object and robot to be stable when object is in box
        is_obj_static = info["is_obj_static"]
        is_robot_static = info["is_robot_static"]
        stability_reward = 1.0 * (is_obj_in_box & is_obj_static & is_robot_static)
        reward += stability_reward
        
        # Success reward - large bonus for complete success
        success = info["success"]
        reward[success] = 6.0  # Maximum reward for complete success
        
        return reward

    def compute_normalized_dense_reward(
        self, obs: Any, action: torch.Tensor, info: Dict
    ):
        return self.compute_dense_reward(obs=obs, action=action, info=info) / 6.0
