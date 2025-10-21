import os.path as osp
from pathlib import Path
from typing import List

import numpy as np
import sapien
import sapien.render
import torch
from transforms3d.euler import euler2quat

from mani_skill.agents.multi_agent import MultiAgent
from mani_skill.agents.robots.fetch import FETCH_WHEELS_COLLISION_BIT
from mani_skill.utils.building.ground import build_ground
from mani_skill.utils.scene_builder import SceneBuilder

from gsworld.constants import fr3_umi_task_init_qpos

x_offset = 0.615
table_height_offset = 0.0 # real-world table is 5cm below base, TODO what about gaussian -> 0312 we have lift the table

# TODO (stao): make the build and initialize api consistent with other scenes
class TableSceneBuilderOffset(SceneBuilder):
    def build(self):
        builder = self.scene.create_actor_builder()
        model_dir = Path(osp.dirname(__file__)) / "assets"
        table_model_file = str(model_dir / "table.glb")
        scale = 1.75

        table_pose = sapien.Pose(p=[x_offset, 0, 0], q=euler2quat(0, 0, np.pi / 2))
        # builder.add_nonconvex_collision_from_file(
        #     filename=table_model_file,
        #     scale=[scale] * 3,
        #     pose=table_pose,
        # )
        builder.add_box_collision(
            pose=sapien.Pose(p=[x_offset, 0, 0.9196429 / 2]),
            half_size=(2.418 / 2, 1.209 / 2, 0.9196429 / 2),
        )
        builder.add_visual_from_file(
            filename=table_model_file, scale=[scale] * 3, pose=table_pose
        )
        builder.initial_pose = sapien.Pose(
            p=[x_offset-0.12, 0, -0.9196429], q=euler2quat(0, 0, np.pi / 2)
        )
        table = builder.build_kinematic(name="table-workspace")
        aabb = (
            table._objs[0]
            .find_component_by_type(sapien.render.RenderBodyComponent)
            .compute_global_aabb_tight()
        )
        self.table_length = aabb[1, 0] - aabb[0, 0]
        self.table_width = aabb[1, 1] - aabb[0, 1]
        self.table_height = aabb[1, 2] - aabb[0, 2]
        floor_width = 100
        if self.scene.parallel_in_single_scene:
            floor_width = 500
        self.ground = build_ground(
            self.scene, floor_width=floor_width, altitude=-self.table_height
        )
        self.table = table
        self.scene_objects: List[sapien.Entity] = [self.table, self.ground]

    def initialize(self, env_idx: torch.Tensor):
        # table_height = 0.9196429
        b = len(env_idx)
        self.table.set_pose(
            sapien.Pose(p=[x_offset-0.12, 0, -0.9196429 + table_height_offset], q=euler2quat(0, 0, np.pi / 2))
        )
        if self.env.robot_uids in ["panda"]: # "fr3_umi", gq: we set qpos for table scene in the env
            qpos = np.array(
                [
                    0.0,
                    np.pi / 8,
                    0,
                    -np.pi * 5 / 8,
                    0,
                    np.pi * 3 / 4,
                    np.pi / 4,
                    0.04,
                    0.04,
                ]
            )
            if self.env._enhanced_determinism:
                qpos = (
                    self.env._batched_episode_rng[env_idx].normal(
                        0, self.robot_init_qpos_noise, len(qpos)
                    )
                    + qpos
                )
            else:
                qpos = (
                    self.env._episode_rng.normal(
                        0, self.robot_init_qpos_noise, (b, len(qpos))
                    )
                    + qpos
                )
            qpos[:, -2:] = 0.04
            self.env.agent.reset(qpos)
            self.env.agent.robot.set_pose(sapien.Pose([x_offset-0.615, 0, 0]))
        elif self.env.robot_uids in ["fr3_umi"]: # TODO what is the influence on motion planning?
            print("Setting fr3 init qpos.")
            qpos = fr3_umi_task_init_qpos # TODO
            if self.env._enhanced_determinism: # gq: TODO what does this mean?
                qpos = (
                    self.env._batched_episode_rng[env_idx].normal(
                        0, self.robot_init_qpos_noise, len(qpos)
                    )
                    + qpos
                )
            else:
                qpos = (
                    self.env._episode_rng.normal(
                        0, self.robot_init_qpos_noise, (b, len(qpos))
                    )
                    + qpos
                )
            qpos[:, -2:] = 0.04 # gq: this is important since qpos will be changed above
            self.env.agent.reset(qpos)
            # self.env.agent.robot.set_pose(sapien.Pose([-0.615, 0, 0]))
