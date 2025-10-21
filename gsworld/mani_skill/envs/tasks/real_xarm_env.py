from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Union, Optional, Sequence

import numpy as np
import sapien
import torch

import dacite
from mani_skill.agents.robots.fetch.fetch import Fetch
from mani_skill.agents.robots.panda.panda import Panda
from mani_skill.envs.sapien_env import BaseEnv
import mani_skill.envs.utils.randomization as randomization
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from mani_skill.utils.building.ground import build_ground
from mani_skill.utils.registration import register_env
from mani_skill.utils import common
from mani_skill.utils.structs.pose import Pose
from mani_skill.utils.geometry.rotation_conversions import matrix_to_quaternion
from gsworld.mani_skill.utils.gsworld_sapien_utils import calib_mat2sapien_trans_mat
from gsworld.constants import rs_d435i_rgb_k, xarm_wrist2base, xarm_right2base

@dataclass
class SO100GraspCubeDomainRandomizationConfig:
    ### task agnostic domain randomizations, many of which you can copy over to your own tasks ###
    initial_qpos_noise_scale: float = 0.02
    robot_color: Optional[Union[str, Sequence[float]]] = None
    """Color of the robot in RGB format in scale of 0 to 1 mapping to 0 to 255.
    If you want to randomize it just set this value to "random". If left as None which is
    the default, it will set the robot parts to white and motors to black. For more fine-grained choices on robot colors you need to modify
    mani_skill/assets/robots/so100/so100.urdf in the ManiSkill package."""
    randomize_lighting: bool = True # as we use gsworld, this is only for baseline
    max_camera_offset: Sequence[float] = (0.025, 0.025, 0.025)
    """max camera offset from the base camera position in x, y, and z axes"""
    camera_target_noise: float = 1e-3
    """scale of noise added to the camera target position"""
    camera_view_rot_noise: float = 5e-3
    """scale of noise added to the camera view rotation"""
    camera_fov_noise: float = np.deg2rad(2)
    """scale of noise added to the camera fov"""

    ### task-specific related domain randomizations that occur during scene loading ###
    obj_scale_range: Sequence[float] = (0.95, 1.05)
    obj_friction_mean: float = 0.3
    obj_friction_std: float = 0.05
    obj_friction_bounds: Sequence[float] = (0.1, 0.5)
    randomize_obj_color: bool = True

    def dict(self):
        return {k: v for k, v in asdict(self).items()}


@register_env("RealXArm6-v1", max_episode_steps=200000)
class RealXArm6(BaseEnv):
    # SUPPORTED_REWARD_MODES = ["none", "dense", "sparse"]
    """
    This is just a dummy environment for showcasing robots in a empty scene
    """

    def __init__(self, *args, 
                robot_uids="panda", 
                domain_randomization_config: Union[
                    SO100GraspCubeDomainRandomizationConfig, dict
                ] = SO100GraspCubeDomainRandomizationConfig(),
                domain_randomization=False,
                **kwargs):
        self.domain_randomization = domain_randomization
        """whether randomization is turned on or off."""
        self.domain_randomization_config = SO100GraspCubeDomainRandomizationConfig()
        """domain randomization config"""
        merged_domain_randomization_config = self.domain_randomization_config.dict()
        if isinstance(domain_randomization_config, dict):
            common.dict_merge(
                merged_domain_randomization_config, domain_randomization_config
            )
            self.domain_randomization_config = dacite.from_dict(
                data_class=SO100GraspCubeDomainRandomizationConfig,
                data=domain_randomization_config,
                config=dacite.Config(strict=True),
            )
        
        
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    @property
    def _default_sensor_configs(self):
        if self.domain_randomization:
            camera_fov_noise = self.domain_randomization_config.camera_fov_noise * (
                2 * self._batched_episode_rng.rand() - 1
            )
        else:
            camera_fov_noise = 0
        
        # TODO refactor later
        # mind the conversion
        wrist2eef_sp = calib_mat2sapien_trans_mat(xarm_wrist2base)
        wrist_p = wrist2eef_sp[:3, 3]
        wrist_q = matrix_to_quaternion(
            common.to_tensor(wrist2eef_sp[:3, :3][None, ...])
        )
        wrist_pose = Pose.create_from_pq(p=wrist_p, q=wrist_q)

        right2base_sp = calib_mat2sapien_trans_mat(xarm_right2base)
        # base matrix self.agent.robot.get_root_pose().to_transformation_matrix()
        right_p = right2base_sp[:3, 3]  # xyz of the camera
        right_q = matrix_to_quaternion(
            common.to_tensor(right2base_sp[:3, :3][None, ...])
        )
        right_pose = Pose.create_from_pq(p=right_p, q=right_q)

        return [
            # CameraConfig("wrist_cam", wrist_pose, width=128, height=128, fov=np.pi / 3, near=0.01, far=100, mount=self.agent.robot.find_link_by_name("fr3_hand_tcp")),
            CameraConfig( # gap too big
                "wrist_cam",
                wrist_pose,
                width=640,
                height=480,
                intrinsic=rs_d435i_rgb_k,
                near=0.01,
                far=100,
                mount=self.agent.robot.find_link_by_name(self.agent.ee_link_name),
            ),
            CameraConfig(
                "right_cam",
                right_pose,
                width=640,
                height=480,
                intrinsic=rs_d435i_rgb_k,
                near=0.01,
                far=100,
                mount=self.agent.robot.find_link_by_name(self.agent.base_link_name),
            ),  # mount base or not is the same
            # CameraConfig("base_camera", pose=sapien_utils.look_at([0.615 + 0.5, 0.5, 0.7], [0.615, 0.2, 0.35]), width=640, height=480, intrinsic=rs_d435i_rgb_k, near=0.01, far=100) # third-person view for rendering
        ]

    # @property
    # def _default_sensor_configs(self):
    #     pose = sapien_utils.look_at([1.2, 1.0, 0.7], [0.2, 0.1, 0.])
    #     return CameraConfig(
    #         "base_camera", pose=pose, width=640, height=480, intrinsic=rs_d435i_rgb_k, near=0.01, far=100
    #     )

    @property
    def _default_human_render_camera_configs(self):
        # pose = sapien_utils.look_at([1.25, -1.25, 1.5], [0.0, 0.0, 0.2])
        # return CameraConfig("render_camera", pose, 2048, 2048, 1, 0.01, 100)
        pose = sapien_utils.look_at([1.7, 1.0, 0.7], [0.0, 0.0, 0.15])
        return CameraConfig(
            "render_camera",
            pose=pose,
            width=640,
            height=480,
            intrinsic=rs_d435i_rgb_k,
            near=0.01,
            far=100,
        )

    def _load_agent(self, options: dict):
        # default Pose([0, 0, 0], [1, 0, 0, 0])
        super()._load_agent(options, sapien.Pose())
        # self.agent.robot.set_qpos(task_init_qpos) # TODO set qpos here may cause bug

    def _load_scene(self, options: dict):
        self.ground = build_ground(self.scene)
        self.ground.set_collision_group_bit(group=2, bit_idx=30, bit=1)
        
        # randomize cube sizes, colors, and frictions
        if self.domain_randomization:
            # note that we use self._batched_episode_rng instead of torch.rand or np.random as it ensures even with a different number of parallel
            # environments the same seed leads to the same RNG, which is important for reproducibility as geometric changes here aren't saveable in environment state
            obj_scales = self._batched_episode_rng.uniform(
                low=self.domain_randomization_config.obj_scale_range[0],
                high=self.domain_randomization_config.obj_scale_range[1],
            )
            if self.domain_randomization_config.randomize_obj_color:
                colors = self._batched_episode_rng.uniform(low=0, high=1, size=(3,))
            frictions = self._batched_episode_rng.normal(
                self.domain_randomization_config.obj_friction_mean,
                self.domain_randomization_config.obj_friction_std,
            )
            frictions = frictions.clip(
                *self.domain_randomization_config.obj_friction_bounds
            )
            
    def sample_camera_poses(self, n: int):
        # a custom function to sample random camera poses
        # the way this works is we first sample "eyes", which are the camera positions
        # then we use the noised_look_at function to sample the full camera poses given the sampled eyes
        # and a target position the camera is pointing at
        if self.domain_randomization:
            # in case these haven't been moved to torch tensors on the environment device
            self.base_camera_settings["pos"] = common.to_tensor(
                self.base_camera_settings["pos"], device=self.device
            )
            self.base_camera_settings["target"] = common.to_tensor(
                self.base_camera_settings["target"], device=self.device
            )
            self.domain_randomization_config.max_camera_offset = common.to_tensor(
                self.domain_randomization_config.max_camera_offset, device=self.device
            )

            eyes = randomization.camera.make_camera_rectangular_prism(
                n,
                scale=self.domain_randomization_config.max_camera_offset,
                center=self.base_camera_settings["pos"],
                theta=0,
                device=self.device,
            )
            return randomization.camera.noised_look_at(
                eyes,
                target=self.base_camera_settings["target"],
                look_at_noise=self.domain_randomization_config.camera_target_noise,
                view_axis_rot_noise=self.domain_randomization_config.camera_view_rot_noise,
                device=self.device,
            )
        else:
            return sapien_utils.look_at(
                eye=self.base_camera_settings["pos"],
                target=self.base_camera_settings["target"],
            )

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        pass
    
    def _before_control_step(self):
        # update the camera poses before agent actions are executed
        if self.domain_randomization:
            self.camera_mount.set_pose(self.sample_camera_poses(n=self.num_envs))
            if self.gpu_sim_enabled:
                self.scene._gpu_apply_all()

    def evaluate(self):
        return {}

    def _get_obs_extra(self, info: Dict):
        return dict()
