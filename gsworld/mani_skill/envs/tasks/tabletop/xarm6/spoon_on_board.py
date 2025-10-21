from typing import Any, Dict, List, Union

from gsworld.mani_skill.utils.building.actors.dtc import get_dtc_builder
import numpy as np
import sapien
import torch
import warnings

from mani_skill import ASSET_DIR
from mani_skill.agents.robots.fetch.fetch import Fetch
from mani_skill.agents.robots.panda.panda import Panda
from mani_skill.agents.robots.panda.panda_wristcam import PandaWristCam
from mani_skill.envs.utils.randomization.pose import random_quaternions
from mani_skill.utils import common
from mani_skill.utils.building import actors
from mani_skill.utils.io_utils import load_json
from mani_skill.utils.registration import register_env
from gsworld.mani_skill.utils.scene_builder.table import TableSceneBuilderOffset
from mani_skill.utils.structs.actor import Actor
from mani_skill.utils.structs.pose import Pose
from gsworld.mani_skill.envs.tasks.real_xarm_env import RealXArm6
from mani_skill.utils.geometry.rotation_conversions import axis_angle_to_quaternion, quaternion_multiply

WARNED_ONCE = False

@register_env("SpoonOnBoardXArmEnv-v1", max_episode_steps=100, asset_download_ids=["ycb"])
class SpoonOnBoardXArmEnv(RealXArm6):

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
        
        self.robot_base = actors.build_box(
            self.scene,
            half_sizes=[0.05, 0.05, 0.015],  # 10cm x 10cm x 1cm square base
            color=[0.3, 0.3, 0.3, 1],  # Dark gray color
            name="robot_base",
            initial_pose=sapien.Pose([0, 0, 0.005]),  # Position so top is at z=0.01 (1cm height)
            body_type="static"  # Make it static so it doesn't move
        )

        # Create invisible support blocks for the spoon
        self._create_support_blocks()

        model_ids = ["Kitchen_Spoon_B008H2JLP8_LargeWooden"]
        if (
            self.num_envs > 1
            and self.num_envs < len(model_ids)
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
            builder = get_dtc_builder(self.scene, id=model_id)
            builder.initial_pose = sapien.Pose(p=[self.x_offset, 0.2, 0])
            builder.set_scene_idxs(list(range(self.num_envs)))
            self._objs.append( builder.build(name=f"dtc:{model_id}-{i}"))
            self.remove_from_state_dict_registry(self._objs[-1])
        self.spoon = Actor.merge(self._objs, name=f"dtc:{model_id}")
        self.add_to_state_dict_registry(self.spoon)
        
        goal_model_id = ["Cutting_Board_B005CZ90HM_LimeGreen"]
        self._goals: List[Actor] = []
        for i, model_id in enumerate(goal_model_id):
            builder = get_dtc_builder(self.scene, id=model_id)
            builder.initial_pose = sapien.Pose(p=[self.x_offset, 0.2, 0])
            builder.set_scene_idxs(list(range(self.num_envs)))
            self._goals.append( builder.build(name=f"dtc:{model_id}-{i}"))
            self.remove_from_state_dict_registry(self._goals[-1])
        self.goal_site = Actor.merge(self._goals, name=f"dtc:{model_id}")
        self.add_to_state_dict_registry(self.goal_site)

        self.goal_height = 0.012
        self.obj_height = 0.0

    def _after_reconfigure(self, options: dict):
        self.object_zs = []
        for obj in self._objs:
            collision_mesh = obj.get_first_collision_mesh()
            self.object_zs.append(-collision_mesh.bounding_box.bounds[0, 2])
        self.object_zs = common.to_tensor(self.object_zs, device=self.device)

    def _create_support_blocks(self):
        """Create two invisible support blocks for the spoon."""
        half_block_size = [0.02, 0.02, 0.01]
        
        self.support_blocks = []
        
        for i in range(2): 
            block = actors.build_box(
                self.scene,
                half_sizes=half_block_size,
                color=[0.5, 0.5, 0.5, 0.0],
                name=f"support_block_{i}",
                initial_pose=sapien.Pose([self.x_offset + 1., 0., i]),
            )
            self.support_blocks.append(block)

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)
            
            spoon_xyz = torch.zeros((b, 3), device=self.device)
            spoon_xyz[:, 0] = self.x_offset - 0.3 + torch.rand(b, device=self.device) * 0.05
            spoon_xyz[:, 1] = -0.05 + torch.rand(b, device=self.device) * 0.05
            spoon_xyz[:, 2] = self.obj_height + 0.01
            
            spoon_qs = quaternion_multiply(axis_angle_to_quaternion(axis_angle=torch.tensor([0.0, 0.0, torch.pi / 2], device=self.device)), axis_angle_to_quaternion(axis_angle=torch.tensor([torch.pi / 2, 0.0, 0.0], device=self.device)))
            
            spoon_length = 0.20
            block_offset = spoon_length / 3 
            
            # Block positions relative to spoon center
            block_positions = torch.zeros((b, 2, 3), device=self.device)  # 2 blocks, 3D positions
            
            # Head block (front of spoon)
            block_positions[:, 0, 0] = spoon_xyz[:, 0] - block_offset
            block_positions[:, 0, 1] = spoon_xyz[:, 1]
            block_positions[:, 0, 2] = 0.005
            
            # Tail block (handle of spoon)
            block_positions[:, 1, 0] = spoon_xyz[:, 0] + block_offset
            block_positions[:, 1, 1] = spoon_xyz[:, 1]
            block_positions[:, 1, 2] = 0.005
            
            for i, block in enumerate(self.support_blocks):
                if b > 0:
                    block_pos = block_positions[0, i].cpu().numpy()
                    block.set_pose(sapien.Pose(block_pos))
            
            board_xyz = torch.zeros((b, 3), device=self.device)
            board_xyz[:, 0] = self.x_offset - 0.3 + torch.rand(b, device=self.device) * 0.1
            board_xyz[:, 1] = 0.15 + torch.rand(b, device=self.device) * 0.1
            board_xyz[:, 2] = self.goal_height
            board_qs = axis_angle_to_quaternion(axis_angle=torch.tensor([-torch.pi / 2, 0.0, 0.0], device=self.device))

            if not hasattr(self, "goal_pos"):
                self.goal_pos = torch.zeros((self.num_envs, 3), device=self.device)

            goal_pos_batch = board_xyz.clone()
            goal_pos_batch[:, 2] = self.goal_height + 0.01

            self.goal_pos[env_idx] = goal_pos_batch

            self.spoon.set_pose(Pose.create_from_pq(p=spoon_xyz, q=spoon_qs))
            self.goal_site.set_pose(Pose.create_from_pq(p=board_xyz, q=board_qs))

            if self.robot_uids in ["xarm6_uf_gripper_wrist435", "xarm6_uf_gripper"]:
                from gsworld.constants import robot_task_init_qpos
                self.agent.reset(robot_task_init_qpos[self.robot_uids])
                self.agent.robot.set_root_pose(sapien.Pose([self.x_offset-0.615, 0, 0.03]))
            else:
                raise NotImplementedError(self.robot_uids)

    def evaluate(self):
        spoon_pos = self._objs[0].pose.p
        goal_pos = self.goal_pos
        
        xy_offset = spoon_pos[..., :2] - goal_pos[..., :2]
        board_half_x, board_half_y = 0.15, 0.1
        is_xy_inside = torch.logical_and(
            torch.abs(xy_offset[..., 0]) <= board_half_x - 0.02,
            torch.abs(xy_offset[..., 1]) <= board_half_y - 0.02
        )
        
        z_offset = torch.abs(spoon_pos[..., 2] - goal_pos[..., 2])
        is_z_correct = z_offset <= 0.05
        
        is_grasped = self.agent.is_grasping(self._objs[0], max_angle=180)

        is_robot_static = self.agent.is_static(0.2)
        
        is_spoon_static = torch.logical_and(
            torch.linalg.norm(self._objs[0].linear_velocity, axis=1) < 0.05,
            torch.linalg.norm(self._objs[0].angular_velocity, axis=1) < 0.5
        )

        success = torch.logical_and(is_xy_inside, is_z_correct)
        success = torch.logical_and(success, ~is_grasped)
        success = torch.logical_and(success, is_robot_static)
        
        return {
            "is_grasped": is_grasped,
            "is_spoon_on_board": torch.logical_and(is_xy_inside, is_z_correct),
            "is_robot_static": is_robot_static,
            "is_spoon_static": is_spoon_static,
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
                spoon_pose=self.spoon.pose.raw_pose,
                tcp_to_spoon_pos=self.spoon.pose.p - self.agent.tcp.pose.p,
                spoon_to_board_pos=self.goal_site.pose.p - self.spoon.pose.p,
            )
        return obs

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        # Reaching reward - encourage getting close to the spoon
        tcp_to_spoon_dist = torch.linalg.norm(
            self.spoon.pose.p - self.agent.tcp.pose.p, axis=1
        )
        reaching_reward = 1 - torch.tanh(5 * tcp_to_spoon_dist)
        reward = reaching_reward

        # Grasping reward - bonus for successfully grasping the spoon
        is_grasped = info["is_grasped"]
        grasp_reward = 2.0
        reward += grasp_reward * is_grasped

        # Transport reward - if spoon is grasped, reward moving it toward the cutting board
        spoon_to_board_dist = torch.linalg.norm(
            self.goal_site.pose.p - self.spoon.pose.p, axis=1
        )
        transport_reward = 1 - torch.tanh(5 * spoon_to_board_dist)
        reward += transport_reward * is_grasped

        # Placement reward - bonus for getting spoon on cutting board
        is_spoon_on_board = info["is_spoon_on_board"]
        placement_reward = 1.0
        reward += placement_reward * is_spoon_on_board
        
        # Release reward - encourage releasing the spoon when it's on the cutting board
        release_reward = 1.0 * (is_spoon_on_board & ~is_grasped)
        reward += release_reward
        
        # Stability reward - encourage stable final state
        is_spoon_static = info["is_spoon_static"]
        is_robot_static = info["is_robot_static"]
        stability_reward = 1.0 * (is_spoon_on_board & is_spoon_static & is_robot_static)
        reward += stability_reward
        
        # Success reward - large bonus for complete success
        success = info["success"]
        reward[success] = 8.0
        
        return reward

    def compute_normalized_dense_reward(
        self, obs: Any, action: torch.Tensor, info: Dict
    ):
        return self.compute_dense_reward(obs=obs, action=action, info=info) / 8.0