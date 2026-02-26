from typing import Dict

import numpy as np
import sapien
import torch

from mani_skill.agents.robots.fetch.fetch import Fetch
from mani_skill.agents.robots.panda.panda import Panda
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from mani_skill.utils.building.ground import build_ground
from mani_skill.utils.registration import register_env
from mani_skill.utils.structs.types import GPUMemoryConfig, SimConfig
from gsworld.mani_skill.utils.gsworld_sapien_utils import calib_mat2sapien_trans_mat
from mani_skill.utils.structs.pose import Pose
from mani_skill.utils.geometry.rotation_conversions import matrix_to_quaternion
from mani_skill.utils import common
from gsworld.constants import rs_d435i_rgb_k, wrist2eef, right2base


@register_env("RealFr3-v1", max_episode_steps=200000)
class RealFr3(BaseEnv):
    SUPPORTED_REWARD_MODES = ["none", "dense", "sparse"]
    """
    This is just a dummy environment for showcasing robots in a empty scene
    """

    def __init__(self, *args, robot_uids="panda", **kwargs):
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    @property
    def _default_sensor_configs(self):
        # TODO refactor later
        # mind the conversion
        wrist2eef_sp = calib_mat2sapien_trans_mat(wrist2eef)
        wrist_p = wrist2eef_sp[:3, 3]
        wrist_q = matrix_to_quaternion(common.to_tensor(wrist2eef_sp[:3, :3][None, ...]))
        wrist_pose = Pose.create_from_pq(p=wrist_p, q=wrist_q)

        right2base_sp = calib_mat2sapien_trans_mat(right2base)
        # base matrix self.agent.robot.get_root_pose().to_transformation_matrix()
        right_p = right2base_sp[:3, 3] # xyz of the camera
        right_q = matrix_to_quaternion(common.to_tensor(right2base_sp[:3, :3][None, ...]))
        right_pose = Pose.create_from_pq(p=right_p, q=right_q)

        return [
            # CameraConfig("wrist_cam", wrist_pose, width=128, height=128, fov=np.pi / 3, near=0.01, far=100, mount=self.agent.robot.find_link_by_name("fr3_hand_tcp")),
            CameraConfig("wrist_cam", wrist_pose, width=640, height=480, intrinsic=rs_d435i_rgb_k, near=0.01, far=100, mount=self.agent.robot.find_link_by_name(self.agent.ee_link_name)),
            CameraConfig("right_cam", right_pose, width=640, height=480, intrinsic=rs_d435i_rgb_k, near=0.01, far=100, mount=self.agent.robot.find_link_by_name(self.agent.base_link_name)), # mourt base or not is the same
            # CameraConfig("base_camera", pose=sapien_utils.look_at([0.615 + 0.5, 0.5, 0.7], [0.615, 0.2, 0.35]), width=1024, height=1024, intrinsic=rs_d435i_rgb_k, near=0.01, far=100) # third-person view for rendering
            ]

    # @property
    # def _default_sensor_configs(self):
    #     pose = sapien_utils.look_at([0.7, 0.8, 0.7], [0.2, 0.1, 0.35])
    #     return CameraConfig(
    #         "base_camera", pose=pose, width=512, height=512, fov=1, near=0.01, far=100
    #     )

    @property
    def _default_human_render_camera_configs(self):
        # pose = sapien_utils.look_at([1.25, -1.25, 1.5], [0.0, 0.0, 0.2])
        # return CameraConfig("render_camera", pose, 2048, 2048, 1, 0.01, 100)
        pose = sapien_utils.look_at([1, 0.2, 0.5], [0.0, 0.0, 0.15])
        return CameraConfig(
            "render_camera", pose=pose, width=640, height=480, intrinsic=rs_d435i_rgb_k, near=0.01, far=100
        )

    def _load_agent(self, options: dict):
        # default Pose([0, 0, 0], [1, 0, 0, 0])
        super()._load_agent(options, sapien.Pose())
        # TODO to render some specified qpos
        # self.agent.robot.set_qpos(task_init_qpos)
        # mujoco_qpos = np.array([
        #         0.022818852216005325,
        #         0.019855299964547157,
        #         -0.31693196296691895,
        #         0.20567715167999268,
        #         -0.007576329633593559,
        #         -1.5873229503631592,
        #         0.8329577445983887,
        #         1.5010179281234741, 
        #         0.010787677019834518,
        #     ], dtype=np.float32)
        # self.agent.robot.set_qpos(mujoco_qpos) # mujoco
        # pyb_qpos = np.array([
        #         0.025479335337877274,
        #         0.017703557386994362,
        #         -0.077215276658535,
        #         -0.08843664824962616,
        #         -0.16147692501544952,
        #         -1.4793751239776611,
        #         -0.12675271928310394,
        #         1.8767969608306885, 
        #         -0.030816059559583664,
        #     ], dtype=np.float32)
        # self.agent.robot.set_qpos(pyb_qpos) # pybullet
        # sapien3_qpos = np.array([
        #         0.023324966430664062,
        #         0.023194747045636177,
        #         0.0005057987873442471,
        #         -0.43101203441619873,
        #         0.0002071257185889408,
        #         -1.3006396293640137,
        #         -0.0031649728771299124,
        #         1.1569830179214478, 
        #         0.3061021864414215,
        #     ], dtype=np.float32)
        # self.agent.robot.set_qpos(sapien3_qpos) # sapien3

    def _load_scene(self, options: dict):
        self.ground = build_ground(self.scene)
        self.ground.set_collision_group_bit(group=2, bit_idx=30, bit=1)

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        pass

    def evaluate(self):
        return {}

    def _get_obs_extra(self, info: Dict):
        return dict()