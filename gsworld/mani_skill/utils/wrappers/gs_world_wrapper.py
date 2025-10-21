import copy
from typing import Union
import gymnasium as gym
import torch
import os

from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.utils.structs import Actor, Link
from mani_skill.utils.structs.pose import Pose

from scene.cameras import Camera
from gaussian_renderer import render
from arguments import PipelineParams

from gsworld.utils.pcd_utils import extract_rigid_transform
from gsworld.utils.gs_utils import transform_gaussians
from gsworld.constants import *
from gsworld.utils.gaussian_merger import GaussianModelMerger

from PIL import Image

try:
    from diff_gaussian_rasterization import SparseGaussianAdam
    SPARSE_ADAM_AVAILABLE = True
except:
    SPARSE_ADAM_AVAILABLE = False


class GSWorldWrapper(gym.Wrapper):
    """This wrapper wraps any maniskill env created via gym.make to add gaussian splattings.
    Also, this wrapper returns the gaussian observation. Code is adapted from original gymnasium wrapper.
    """

    def __init__(
        self,
        env: gym.Env,
        robot_pipe: PipelineParams,
        scene_gs_cfg_name: str = "franka_merger",
        device: Union[str, torch.device] = "cuda",
        log_state: bool = False,
        state_log_path: str = "./exp_log",
        cam_randomization: bool = False,
    ):
        super().__init__(env)
        self.num_envs = env.num_envs
        self.robot_pipe = robot_pipe
        self.device = device
        self.scene_gs_cfg_name = scene_gs_cfg_name
        self.log_state = log_state
        self.cam_randomization = cam_randomization
        ###########################################################
        if "xarm" in self.scene_gs_cfg_name:
            self.gs_semantics = xarm_gs_semantics
            self.sim2gs_arm_trans = sim2gs_xarm_trans
        elif "fr3" in self.scene_gs_cfg_name or "franka" in self.scene_gs_cfg_name:
            self.gs_semantics = fr3_gs_semantics
            self.sim2gs_arm_trans = sim2gs_arm_trans
        elif "r1" in self.scene_gs_cfg_name:
            self.gs_semantics = r1_gs_semantics
            self.sim2gs_arm_trans = sim2gs_r1_trans
        else:
            raise NotImplementedError
        self.sim2gs_arm_trans = torch.tensor(self.sim2gs_arm_trans, dtype=torch.float32, device=self.device)
        self.obj_gs_semantics = obj_gs_semantics
        ###########################################################
        
        # Extract rigid transformation
        self.rigid_sim2real, self.scale_sim2real, R_sim2real, t_sim2real = extract_rigid_transform(self.sim2gs_arm_trans)
        
        # Setup camera
        self.human_cam_config = dict(render_camera=self.base_env.scene.human_render_cameras['render_camera'].get_params())
        self.state_log_path_prefix = state_log_path

        ###########################################################################
        merger = GaussianModelMerger()
        path = os.path.join(CFG_DIR, f"{scene_gs_cfg_name}.json")
        indices = merger.load_models_from_config(path)
        merged_model = merger.merge_models()
        self.initial_merger_robot = merged_model
        ###########################################################################

        env4moving = gym.make(
            "Empty-v1",
            obs_mode="none",
            reward_mode="none",
            enable_shadow=False,
            robot_uids=self.base_env.agent.robot.name,
            render_mode=None,
            sim_config=dict(sim_freq=100, control_freq=20),
            sim_backend="auto",
        )
        env4moving.reset(seed=0)
        self.env4moving: BaseEnv = env4moving.unwrapped

        self.gs_initial_qpos = robot_scan_qpos[self.base_env.agent.robot.name]
        self.task_init_qpos = robot_task_init_qpos[self.base_env.agent.robot.name]
        # ########################################
        self.env4moving.agent.robot.set_qpos(self.gs_initial_qpos)
        self.gs_link_pose_mats = []
        for link in self.env4moving.agent.robot.get_links():
            link_mat = link.pose.to_transformation_matrix().to(self.device)
            self.gs_link_pose_mats.append(link_mat)
        env4moving.close()
        print("finished storing initial qpos and link pose mats")
        # ######################

        self.gs_movable_pts = dict()


    def transform_gs_perlink(self, splats, gs_init_qpos=None, target_mat=None, eef_pos=None):
        """Transform gaussian splatting model based on robot joint states"""
        transformed_splats = copy.deepcopy(splats)

        ######################
        for idx, link in enumerate(self.env.unwrapped.agent.robot.get_links()):
            link_mat = link.pose.to_transformation_matrix().to(self.device)
            if "xarm" in self.env.unwrapped.agent.uid:
                for i in range(3):
                    link_mat[:, i, 3] += object_offset["xarm_arm"][i]
            link_trans = self.sim2gs_arm_trans @ link_mat @ torch.linalg.inv(self.gs_link_pose_mats[idx]) @ torch.linalg.inv(self.sim2gs_arm_trans)
            target = torch.tensor(self.gs_semantics[link.name], device=transformed_splats._semantics.device)
            mask = torch.isin(transformed_splats._semantics.long().squeeze(-1), target.long())
            link_xyz, link_scaling, link_rotation, link_opacity = transform_gaussians(
                transformed_splats,
                selected_indices=torch.where(mask)[0],
                scale=None,
                rot_mat=link_trans[:, :3, :3],
                translation=link_trans[:, :3, 3],
                new_opacity=None,
                )
            self.gs_movable_pts[link.name] = (link_xyz, link_scaling, link_rotation, link_opacity)

        #######################

        if "actors" in self.env.base_env.get_state_dict():
            for actor_key in self.env.base_env.get_state_dict()['actors']:
                if actor_key not in sim2gs_object_transforms:
                    continue
                pose_tensor = self.env.base_env.get_state_dict()['actors'][actor_key][:, :7]
                mat = Pose.create(pose=pose_tensor).to_transformation_matrix().to(self.device)
                if actor_key in object_offset:
                    for i in range(3):
                        mat[:, i, 3] += object_offset[actor_key][i]

                # Get sim2gs transform for this object
                sim2gs_obj_trans = torch.tensor(sim2gs_object_transforms[actor_key], device=self.device, dtype=torch.float32)
                full_transform = self.sim2gs_arm_trans @ mat @ torch.linalg.inv(sim2gs_obj_trans)

                # Extract and apply rigid transform
                transform, scale, _, _ = extract_rigid_transform(full_transform)

                indices = torch.where(transformed_splats._semantics == self.obj_gs_semantics[actor_key])[0]
                obj_xyz, obj_scaling, obj_rotation, obj_opacity = transform_gaussians(
                    transformed_splats,
                    selected_indices=indices,
                    scale=scale * object_scale[actor_key],
                    rot_mat=transform[:, :3, :3],
                    translation=transform[:, :3, 3],
                    new_opacity=None,
                )
                self.gs_movable_pts[actor_key] = (obj_xyz, obj_scaling, obj_rotation, obj_opacity)
                self.gs_semantics[actor_key] = self.obj_gs_semantics[actor_key]

    @property
    def base_env(self) -> BaseEnv:
        return self.env.unwrapped
    
    def get_seg_id2obj_mapping(self):
        for obj_id, obj in sorted(self.base_env.segmentation_id_map.items()):
            if isinstance(obj, Actor):
                print(f"{obj_id}: Actor, name - {obj.name}")
            elif isinstance(obj, Link):
                print(f"{obj_id}: Link, name - {obj.name}")


    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        self.transform_gs_perlink(self.initial_merger_robot)
        #######################################################################
        
        gs_renders = self._render_gsworld()
        for cam_name, gs_render in gs_renders.items():
            obs['sensor_data'][cam_name]['rgb'] = gs_render
        
        return obs, reward, terminated, truncated, info

    def reset(self, *, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)

        self.transform_gs_perlink(self.initial_merger_robot)
        ##############################################
        
        gs_renders = self._render_gsworld()
        for cam_name, gs_render in gs_renders.items():
            obs['sensor_data'][cam_name]['rgb'] = gs_render
        
        return obs, info

    def render(self):
        ret = self.env.render()
        return ret
    
    def render_current_step(self):
        # code adapted from sapien base env
        info = self.env.unwrapped.get_info()
        obs = self.env.unwrapped.get_obs(info)
        reward = self.unwrapped.get_reward(obs=obs, action=None, info=info)
        if "success" in info:
            if "fail" in info:
                terminated = torch.logical_or(info["success"], info["fail"])
            else:
                terminated = info["success"].clone()
        else:
            if "fail" in info:
                terminated = info["fail"].clone()
            else:
                terminated = torch.zeros(self.num_envs, dtype=bool, device=self.device)

        gs_renders = self._render_gsworld()
        for cam_name, gs_render in gs_renders.items():
            obs['sensor_data'][cam_name]['rgb'] = gs_render
        
        return (
            obs,
            reward,
            terminated,
            torch.zeros(self.num_envs, dtype=bool, device=self.device),
            info,
        )
        
    def _render_gsworld(self, renderer='gaussian-splatting'):
        # get gs rendering
        bg_color = [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        gs_renders = {}
        self.gs_cam = self.cam_maniskill2gs(self.base_env.get_sensor_params(), self.device, cam_names=list(self.base_env.get_sensor_params().keys()))
        for cam_name, cam_param in self.gs_cam.items():
            if renderer == 'gaussian-splatting':
                cam_name_rendering = []
                for i in range(self.num_envs):
                    # prepare gaussian model
                    gs4render = copy.deepcopy(self.initial_merger_robot)
                    for key in self.gs_movable_pts:
                        if self.gs_movable_pts[key][0].shape[0] == self.num_envs:
                            gs4render._xyz[
                                torch.isin(gs4render._semantics.long().squeeze(-1),
                                torch.tensor(self.gs_semantics[key],
                                device=gs4render._semantics.device).long())] = self.gs_movable_pts[key][0][i]
                        if self.gs_movable_pts[key][1].shape[0] == self.num_envs:
                            gs4render._scaling[
                                torch.isin(gs4render._semantics.long().squeeze(-1), 
                                torch.tensor(self.gs_semantics[key], 
                                device=gs4render._semantics.device).long())] = self.gs_movable_pts[key][1][i]
                        if self.gs_movable_pts[key][2].shape[0] == self.num_envs:
                            gs4render._rotation[
                                torch.isin(gs4render._semantics.long().squeeze(-1), 
                                torch.tensor(self.gs_semantics[key], 
                                device=gs4render._semantics.device).long())] = self.gs_movable_pts[key][2][i]
                        if self.gs_movable_pts[key][3].shape[0] == self.num_envs:
                            gs4render._opacity[
                                torch.isin(gs4render._semantics.long().squeeze(-1), 
                                torch.tensor(self.gs_semantics[key], 
                                device=gs4render._semantics.device).long())] = self.gs_movable_pts[key][3][i]
                    rendering = render(cam_param, gs4render, self.robot_pipe, background, 
                                    use_trained_exp=False, separate_sh=SPARSE_ADAM_AVAILABLE)["render"]
                    gs_render = rendering.permute(1, 2, 0).unsqueeze(dim=0)
                    # Convert gs_render from [0,1] float32 to [0,255] uint8
                    cam_name_rendering.append((gs_render * 255).clamp(0, 255).to(torch.uint8))
            else:
                return ValueError(f"{renderer} is not supported. Official renderer also supports gsplat.")
            gs_renders.update({cam_name: torch.vstack(cam_name_rendering)}) # (num_envs, H, W, 3)
        
        return gs_renders
        
    def cam_maniskill2gs(self, config_maniskill: dict, device: Union[str, torch.device], cam_names: list) -> Camera:
        cam_params = {}
        for cam_name in cam_names:
            extrinsic_rt_cv = config_maniskill[cam_name]['extrinsic_cv'][0].to(device)
            intrinsic_k = config_maniskill[cam_name]['intrinsic_cv'][0].to(device)
            
            sim_world2cam = torch.vstack([extrinsic_rt_cv, torch.tensor([0, 0, 0, 1], dtype=extrinsic_rt_cv.dtype, device=extrinsic_rt_cv.device)])
            sim_cam2world = torch.linalg.inv(sim_world2cam)

            fx = intrinsic_k[0, 0]
            fy = intrinsic_k[1, 1]
            img_w = self.base_env.get_sensor_images()[cam_name]['rgb'].shape[2]
            img_h = self.base_env.get_sensor_images()[cam_name]['rgb'].shape[1]

            resolution = (img_w, img_h)

            fovx = 2 * torch.arctan(img_w / (2*fx))
            fovy = 2 * torch.arctan(img_h / (2*fy))

            # Transform the camera pose to real-world frame
            real_cam2world = sim_cam2world
            real_cam2world[:3, 3] = real_cam2world[:3, 3] * self.scale_sim2real
            real_world2cam = torch.linalg.inv(self.rigid_sim2real @ real_cam2world)
            R = real_world2cam[:3, :3].T
            T = real_world2cam[:3, 3]

            invdepthmap = None
            img = Image.new('RGB', (img_w, img_h), color='white')
            img_name = cam_name
            colmap_id = 0
            uid = 0

            cam_param = Camera(
                resolution=resolution,
                colmap_id=colmap_id,
                R=R.cpu().numpy(),
                T=T.cpu().numpy(),
                FoVx=float(fovx),
                FoVy=float(fovy),
                depth_params=None,
                image=img,
                invdepthmap=invdepthmap,
                image_name=img_name,
                uid=uid,
                data_device=device,
            )
            cam_params.update({cam_name: cam_param})

        return cam_params