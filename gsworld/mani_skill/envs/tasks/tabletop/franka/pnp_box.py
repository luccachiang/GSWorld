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
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import common, sapien_utils
from mani_skill.utils.building import actors
from mani_skill.utils.io_utils import load_json
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from gsworld.mani_skill.utils.scene_builder.table import TableSceneBuilderOffset
from mani_skill.utils.structs.actor import Actor
from mani_skill.utils.structs.pose import Pose
from mani_skill.utils.structs.types import GPUMemoryConfig, SimConfig
from gsworld.mani_skill.envs.tasks.real_fr3_env import RealFr3
from gsworld.constants import fr3_umi_task_init_qpos

WARNED_ONCE = False


@register_env("PnpBoxFr3Env-v1", max_episode_steps=50, asset_download_ids=["ycb"])
class PnpBoxFr3Env(RealFr3):
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

        model_ids = ["006_mustard_bottle"]
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
        for i, model_id in enumerate(model_ids):
            builder = actors.get_actor_builder(
                self.scene,
                id=f"ycb:{model_id}",
            )
            builder.initial_pose = sapien.Pose(p=[self.x_offset, 0.2, 0])
            builder.set_scene_idxs([i])
            self._objs.append( builder.build(name=f"{model_id}-{i}"))
            self.remove_from_state_dict_registry(self._objs[-1])
        self.obj = Actor.merge(self._objs, name=f"{model_id}")
        self.add_to_state_dict_registry(self.obj)
        
        self.goal_site = actors.build_box(
                    self.scene,
                    half_sizes=[0.33 * 0.5, 0.195 * 0.5, 0.065 * 0.5],
                    color=[0, 1, 0, 1],
                    name="snack_box",
                    initial_pose=sapien.Pose([self.x_offset, -0.2, 0]), # set to origin to export mesh to align
                )
        self.goal_height = 0.033
        self.obj_height = 0.098

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
            xyz = torch.zeros((b, 3))
            xyz[:, 0] = torch.rand((b)) * 0.2 - 0.25 + self.x_offset
            xyz[:, 1] = torch.rand((b)) * 0.1 + 0.1
            xyz[:, 2] = self.obj_height
            qs = random_quaternions(b, lock_x=True, lock_y=True)

            goal_xyz = torch.zeros((b, 3))
            goal_xyz[:, 0] = torch.rand((b)) * 0.2 - 0.25 + self.x_offset # 0.05, -0.15 / + self.x_offset
            goal_xyz[:, 1] = torch.rand((b)) * 0.1 - 0.2 # -0.2, -0.1
            goal_xyz[:, 2] = self.goal_height
            if torch.rand(1) > 0.5:
                xyz[:, :2], goal_xyz[:, :2] = goal_xyz[:, :2].clone(), xyz[:, :2].clone()
            print(f"obj pose {xyz} {qs}, goal pose {goal_xyz}")
            self.obj.set_pose(Pose.create_from_pq(p=xyz, q=qs))
            self.goal_site.set_pose(Pose.create_from_pq(goal_xyz))

            if self.robot_uids in ["panda", "panda_wristcam"]:
                # fmt: off
                qpos = np.array(
                    [0.0, 0, 0, -np.pi * 2 / 3, 0, np.pi * 2 / 3, np.pi / 4, 0.04, 0.04]
                )
                # fmt: on
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
        obj_pos = self.obj.pose.p
        goal_box_pos = self.goal_site.pose.p
        
        # Check if object is inside the box
        # The box has dimensions defined by half_size (x,y) and height (z)
        # Check if object is within the x-y bounds of the box
        xy_offset = obj_pos[..., :2] - goal_box_pos[..., :2]
        half_xy_goal = np.abs(self.goal_site.get_first_collision_mesh().bounds[1, :] - self.goal_site.get_first_collision_mesh().bounds[0, :])[:2].max() * 0.5
        is_xy_inside = torch.linalg.norm(xy_offset, axis=1) <= half_xy_goal - 0.02  # Allow some margin
        
        # Object is inside box if both xy and z conditions are met
        # is_obj_in_box = torch.logical_and(is_xy_inside, is_z_inside)
        is_obj_in_box = is_xy_inside # we dont check z axis
        
        # Check if robot is grasping the object
        is_grasped = self.agent.is_grasping(self.obj)
        
        # Check if robot is static (not moving)
        is_robot_static = self.agent.is_static(0.2)
        
        # Check if object is static (not moving)
        is_obj_static = torch.logical_and(
            torch.linalg.norm(self.obj.linear_velocity, axis=1) < 0.05,
            torch.linalg.norm(self.obj.angular_velocity, axis=1) < 0.5
        )
        
        success = torch.logical_and(is_obj_in_box, ~is_grasped)
        success = torch.logical_and(success, is_obj_static)
        
        return {
            "is_grasped": is_grasped,
            "is_obj_in_box": is_obj_in_box,
            "is_robot_static": is_robot_static,
            "is_obj_static": is_obj_static,
            "success": success,
        }

    def _get_obs_extra(self, info: Dict):
        obs = dict(
            tcp_pose=self.agent.tcp.pose.raw_pose,
            goal_pos=self.goal_site.pose.p,
            is_grasped=info["is_grasped"],
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
        is_grasped = info["is_grasped"]
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
