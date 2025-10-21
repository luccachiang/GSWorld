#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

from argparse import ArgumentParser, Namespace
import sys
import os

import numpy as np
import torch
import torch.nn as nn
from mani_skill.utils.geometry.rotation_conversions import matrix_to_quaternion, quaternion_multiply


class GroupParams:
    pass

class ParamGroup:
    def __init__(self, parser: ArgumentParser, name : str, fill_none = False):
        group = parser.add_argument_group(name)
        for key, value in vars(self).items():
            shorthand = False
            if key.startswith("_"):
                shorthand = True
                key = key[1:]
            t = type(value)
            value = value if not fill_none else None 
            if shorthand:
                if t == bool:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, action="store_true")
                else:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, type=t)
            else:
                if t == bool:
                    group.add_argument("--" + key, default=value, action="store_true")
                else:
                    group.add_argument("--" + key, default=value, type=t)

    def extract(self, args):
        group = GroupParams()
        for arg in vars(args).items():
            if arg[0] in vars(self) or ("_" + arg[0]) in vars(self):
                setattr(group, arg[0], arg[1])
        return group

class ModelParams(ParamGroup): 
    def __init__(self, parser, sentinel=False):
        self.sh_degree = 3
        self._source_path = ""
        self._model_path = ""
        self._images = "images"
        self._depths = ""
        self._resolution = -1
        self._white_background = False
        self.train_test_exp = False
        self.data_device = "cuda"
        self.eval = False
        super().__init__(parser, "Loading Parameters", sentinel)

    def extract(self, args):
        g = super().extract(args)
        g.source_path = os.path.abspath(g.source_path)
        return g

class PipelineParams(ParamGroup):
    def __init__(self, parser):
        self.convert_SHs_python = False
        self.compute_cov3D_python = False
        self.debug = False
        self.antialiasing = False
        super().__init__(parser, "Pipeline Parameters")

class OptimizationParams(ParamGroup):
    def __init__(self, parser):
        self.iterations = 30_000
        self.position_lr_init = 0.00016
        self.position_lr_final = 0.0000016
        self.position_lr_delay_mult = 0.01
        self.position_lr_max_steps = 30_000
        self.feature_lr = 0.0025
        self.opacity_lr = 0.025
        self.scaling_lr = 0.005
        self.rotation_lr = 0.001
        self.exposure_lr_init = 0.01
        self.exposure_lr_final = 0.001
        self.exposure_lr_delay_steps = 0
        self.exposure_lr_delay_mult = 0.0
        self.percent_dense = 0.01
        self.lambda_dssim = 0.2
        self.densification_interval = 100
        self.opacity_reset_interval = 3000
        self.densify_from_iter = 500
        self.densify_until_iter = 15_000 # TODO turn off densification by setting to -1
        self.densify_grad_threshold = 0.0002
        self.depth_l1_weight_init = 1.0
        self.depth_l1_weight_final = 0.01
        self.random_background = False
        self.optimizer_type = "default"
        super().__init__(parser, "Optimization Parameters")

def get_combined_args(parser : ArgumentParser):
    cmdlne_string = sys.argv[1:]
    cfgfile_string = "Namespace()"
    args_cmdline = parser.parse_args(cmdlne_string)

    try:
        cfgfilepath = os.path.join(args_cmdline.model_path, "cfg_args")
        print("Looking for config file in", cfgfilepath)
        with open(cfgfilepath) as cfg_file:
            print("Config file found: {}".format(cfgfilepath))
            cfgfile_string = cfg_file.read()
    except TypeError:
        print("Config file not found at")
        pass
    args_cfgfile = eval(cfgfile_string)

    merged_dict = vars(args_cfgfile).copy()
    for k,v in vars(args_cmdline).items():
        if v != None:
            merged_dict[k] = v
    return Namespace(**merged_dict)

def convert_dict_to_model_gs(gs_model, param_dict):
    xyz = param_dict["means"]
    features_dc = param_dict["sh0"]
    features_extra = param_dict["shN"] # This is slightly different
    scales = param_dict["scales"]
    rots = param_dict["quats"]
    opacities = param_dict["opacities"][..., None]
    gs_model._xyz = nn.Parameter(torch.tensor(xyz.clone().detach(), dtype=torch.float, device="cuda").requires_grad_(True))
    gs_model._features_dc = nn.Parameter(torch.tensor(features_dc.clone().detach(), dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
    gs_model._features_rest = nn.Parameter(torch.tensor(features_extra.clone().detach(), dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
    gs_model._opacity = nn.Parameter(torch.tensor(opacities.clone().detach(), dtype=torch.float, device="cuda").requires_grad_(True))
    gs_model._scaling = nn.Parameter(torch.tensor(scales.clone().detach(), dtype=torch.float, device="cuda").requires_grad_(True))
    gs_model._rotation = nn.Parameter(torch.tensor(rots.clone().detach(), dtype=torch.float, device="cuda").requires_grad_(True))

    gs_model.active_sh_degree = gs_model.max_sh_degree
    
    return gs_model

##### this is used for rigid body transformation, modify gaussian in place
# gaussians is a dictionary
def translate_gaussian(gaussians, translation, selected_indices):
    """
    translation: (3,) or (B, 3)
    Returns: (B, N, 3) or (N, 3) if no batch
    """
    selected_pts = gaussians._xyz[selected_indices]  # (N, 3)

    if translation.dim() == 1:  # (3,)
        result = selected_pts + translation  # (N, 3)
    elif translation.dim() == 2:  # (B, 3)
        B = translation.size(0)
        N = selected_pts.size(0)
        result = selected_pts.unsqueeze(0).expand(B, N, 3) + translation.unsqueeze(1)  # (B, N, 3)
    else:
        raise ValueError(f"Unexpected translation shape {translation.shape}")

    return result # xyz


def inverse_sigmoid(x):
    return torch.log(x / (1 - x))


def change_gaussian_opacity(gaussians, new_opacity, selected_indices):
    """
    new_opacity: scalar or (B,)
    Returns: (B, N) or (N,)
    """
    selected_opacities = gaussians._opacity[selected_indices]  # (N,)
    mean_threshold = selected_opacities.mean() * 5
    mask = selected_opacities < mean_threshold  # (N,)
    if new_opacity.dim() == 0:  # scalar
        result = selected_opacities.clone()
        result[mask] = new_opacity
    elif new_opacity.dim() == 1:  # (B,)
        B = new_opacity.size(0)
        N = selected_opacities.size(0)
        # Broadcast both mask and opacities
        result = selected_opacities.unsqueeze(0).expand(B, N).clone()  # (B, N)
        mask_expanded = mask.unsqueeze(0).expand(B, N)                 # (B, N)
        result[mask_expanded] = new_opacity.unsqueeze(1).expand(B, N)[mask_expanded]
    else:
        raise ValueError(f"Unexpected new_opacity shape {new_opacity.shape}")

    return result # opacity


def scale_gaussian(gaussians, scale, selected_indices):
    """
    scale: scalar or (B,)
    Returns:
        xyz_scaled: (B, N, 3) or (N, 3)
        scaling_scaled: (B, N, D) or (N, D), D = gaussians._scaling.shape[-1]
    """
    xyz = gaussians._xyz[selected_indices]        # (N, 3)
    scaling = gaussians._scaling[selected_indices]  # (N, D)

    if scale.dim() == 0:  # scalar
        xyz_scaled = xyz * scale
        scaling_scaled = inverse_sigmoid(torch.exp(scaling) * scale)
    elif scale.dim() == 1:  # (B,)
        B = scale.size(0)
        N = xyz.size(0)
        D = scaling.size(1)

        xyz_scaled = xyz.unsqueeze(0).expand(B, N, 3) * scale.unsqueeze(1).unsqueeze(-1)  # (B, N, 3)
        scaling_scaled = inverse_sigmoid(
            torch.exp(scaling.unsqueeze(0).expand(B, N, D)) * scale.unsqueeze(1).unsqueeze(-1)
        )  # (B, N, D)
    else:
        raise ValueError(f"Unexpected scale shape {scale.shape}")

    return xyz_scaled, scaling_scaled # xyz, scale


def get_gaussian_rotation_quat_pytorch3d(converted_quat, r):
    norm = r.norm(dim=-1, keepdim=True)
    norm_q = r / norm
    # final_q = pytorch3d.transforms.quaternion_multiply(converted_quat, norm_q)
    final_q = quaternion_multiply(converted_quat, norm_q)

    return final_q * norm

def rotate_gaussian(gaussians, rot_mat, selected_indices):
    quat_r = matrix_to_quaternion(rot_mat)
    selected_pts = gaussians._xyz[selected_indices]

    # Case 1: one rotation for all selected points
    if rot_mat.size(0) == 1:
        rot_mat_expanded = rot_mat.expand(selected_pts.size(0), 3, 3)
        selected_pts = torch.bmm(rot_mat_expanded, selected_pts.unsqueeze(-1)).squeeze(-1)
    # Case 2: one-to-one batch correspondence
    elif rot_mat.size(0) == selected_pts.size(0):
        selected_pts = torch.bmm(rot_mat, selected_pts.unsqueeze(-1)).squeeze(-1)
    # Case 3: B rotations applied to all N points -> (B, N, 3)
    elif rot_mat.size(0) > 1 and rot_mat.size(0) != selected_pts.size(0):
        B = rot_mat.size(0)
        N = selected_pts.size(0)
        # Expand points and rotations for broadcasting
        pts_expanded = selected_pts.unsqueeze(0).expand(B, N, 3)        # (B, N, 3)
        rot_expanded = rot_mat.unsqueeze(1).expand(B, N, 3, 3)          # (B, N, 3, 3)
        # Apply batched matmul
        selected_pts = torch.matmul(rot_expanded, pts_expanded.unsqueeze(-1)).squeeze(-1)  # (B, N, 3)
    else:
        raise ValueError(
            f"Batch mismatch: rot_mat batch {rot_mat.size(0)} vs selected points {selected_pts.size(0)}"
        )

    r = gaussians._rotation[selected_indices]
    if r.numel() > 0:
        B = quat_r.size(0)
        N = r.size(0)

        if B == 1:  # one quaternion for all
            quat_r_expanded = quat_r.expand(N, 4)  # (N, 4)
            r_out = get_gaussian_rotation_quat_pytorch3d(quat_r_expanded, r)  # (N, 4)

        elif B == N:  # one-to-one mapping
            r_out = get_gaussian_rotation_quat_pytorch3d(quat_r, r)  # (N, 4)

        else:  # Case 3: B rotations applied to all N -> (B, N, 4)
            quat_r_expanded = quat_r.unsqueeze(1).expand(B, N, 4)  # (B, N, 4)
            r_expanded = r.unsqueeze(0).expand(B, N, 4)            # (B, N, 4)
            # Flatten to batch all pairs (B*N, 4)
            quat_r_flat = quat_r_expanded.reshape(B * N, 4)
            r_flat = r_expanded.reshape(B * N, 4)
            r_out = get_gaussian_rotation_quat_pytorch3d(quat_r_flat, r_flat)
            r_out = r_out.view(B, N, 4)  # reshape back

    return selected_pts, r_out # xyz, quat


# unified batched transformation
def transform_gaussians(
    gaussians,
    selected_indices,
    scale=None,              # scalar or (B,)
    rot_mat=None,            # (1,3,3), (N,3,3), or (B,3,3)
    translation=None,        # (3,) or (B,3)
    new_opacity=None         # scalar or (B,)
):
    """
    Apply transformations to selected Gaussians in the order:
    [scale -> rotate -> translate -> change opacity].

    Returns:
        xyz_out: (N, 3) or (B, N, 3)
        scaling_out: (N, 3) or (B, N, 3)
        rotation_out: (N, 4) or (B, N, 4)
        opacity_out: (N,) or (B, N)
    """
    xyz = gaussians._xyz[selected_indices]            # (N, 3)
    scaling = gaussians._scaling[selected_indices]    # (N, 3)
    rotation = gaussians._rotation[selected_indices]  # (N, 4)
    opacities = gaussians._opacity[selected_indices]  # (N,)

    # ================= Scale =================
    if scale is not None:
        if scale.dim() == 0:  # scalar
            xyz = xyz * scale
            scaling = inverse_sigmoid(torch.exp(scaling) * scale)
        elif scale.dim() == 1:  # (B,)
            B, N, D = scale.size(0), xyz.size(0), scaling.size(1)
            xyz = xyz.unsqueeze(0).expand(B, N, 3) * scale[:, None, None]
            scaling = inverse_sigmoid(
                torch.exp(scaling.unsqueeze(0).expand(B, N, 3)) * scale[:, None, None]
            )
        else:
            raise ValueError(f"Unexpected scale shape {scale.shape}")

    # ================= Rotate =================
    if rot_mat is not None:
        quat_r = matrix_to_quaternion(rot_mat)

        if rot_mat.size(0) == 1:  # one rotation for all points
            if xyz.dim() == 2:  # (N,3)
                rot_exp = rot_mat.expand(xyz.size(0), 3, 3)
                xyz = torch.bmm(rot_exp, xyz.unsqueeze(-1)).squeeze(-1)
            else:  # (B,N,3)
                B, N = xyz.shape[:2]
                rot_exp = rot_mat.expand(B, N, 3, 3)
                xyz = torch.matmul(rot_exp, xyz.unsqueeze(-1)).squeeze(-1)

        elif rot_mat.size(0) == xyz.size(0) and xyz.dim() == 2:
            xyz = torch.bmm(rot_mat, xyz.unsqueeze(-1)).squeeze(-1)
        else:  # (B,3,3) applied to all N
            B, N = rot_mat.size(0), xyz.size(-2)
            pts = xyz if xyz.dim() == 3 else xyz.unsqueeze(0).expand(B, N, 3)
            rot_exp = rot_mat.unsqueeze(1).expand(B, N, 3, 3)
            xyz = torch.matmul(rot_exp, pts.unsqueeze(-1)).squeeze(-1)

        # rotations
        if rotation.numel() > 0:
            # if quat_r.size(0) == 1: # TODO bug
            #     r_out = get_gaussian_rotation_quat_pytorch3d(quat_r.expand(rotation.size(0), 4), rotation)
            if quat_r.size(0) == rotation.size(0) and xyz.dim() == 2:
                r_out = get_gaussian_rotation_quat_pytorch3d(quat_r, rotation)
            else:  # B rotations applied to all N
                B, N = quat_r.size(0), rotation.size(0)
                quat_r_exp = quat_r[:, None, :].expand(B, N, 4)
                r_exp = rotation[None, :, :].expand(B, N, 4)
                r_out = get_gaussian_rotation_quat_pytorch3d(
                    quat_r_exp.reshape(B * N, 4), r_exp.reshape(B * N, 4)
                ).view(B, N, 4)
            rotation = r_out

    # ================= Translate =================
    if translation is not None:
        if translation.dim() == 1:  # (3,)
            xyz = xyz + translation
        elif translation.dim() == 2:  # (B,3)
            if xyz.dim() == 2:  # (N,3)
                B, N = translation.size(0), xyz.size(0)
                xyz = xyz.unsqueeze(0).expand(B, N, 3) + translation[:, None, :]
            else:  # (B,N,3)
                xyz = xyz + translation[:, None, :]
        else:
            raise ValueError(f"Unexpected translation shape {translation.shape}")

    # ================= Opacity =================
    if new_opacity is not None:
        mean_threshold = opacities.mean() * 5
        mask = opacities < mean_threshold
        if new_opacity.dim() == 0:  # scalar
            result = opacities.clone()
            result[mask] = new_opacity
        elif new_opacity.dim() == 1:  # (B,)
            B, N = new_opacity.size(0), opacities.size(0)
            result = opacities[None, :].expand(B, N).clone()
            mask_exp = mask[None, :].expand(B, N)
            result[mask_exp] = new_opacity[:, None].expand(B, N)[mask_exp]
        else:
            raise ValueError(f"Unexpected new_opacity shape {new_opacity.shape}")
        opacities = result

    return xyz, scaling, rotation, opacities


def is_rigid_transformation(T: torch.Tensor, tol: float = 1e-5):
    if T.ndim == 2 and T.shape == (4, 4):
        T = T.unsqueeze(0)  # promote to (1,4,4)
        single_input = True
    elif T.ndim == 3 and T.shape[1:] == (4, 4):
        single_input = False
    else:
        raise ValueError("Input must be (4,4) or (N,4,4)")

    R = T[:, :3, :3]  # (N,3,3)
    I = torch.eye(3, device=T.device, dtype=T.dtype)

    # Orthogonality: R^T R ≈ I
    RtR = torch.matmul(R.transpose(-1, -2), R)  # (N,3,3)
    orthogonality_check = torch.allclose(RtR, I.expand_as(RtR), atol=tol, rtol=0)

    # Determinant ≈ 1
    dets = torch.linalg.det(R)
    det_check = torch.allclose(dets, torch.ones_like(dets), atol=tol, rtol=0)

    if single_input:
        return bool(orthogonality_check and det_check)
    else:
        ortho_ok = torch.isclose(RtR, I, atol=tol, rtol=0).all(dim=(-1, -2))
        det_ok = torch.isclose(dets, torch.tensor(1.0, device=T.device, dtype=T.dtype), atol=tol, rtol=0)
        return ortho_ok & det_ok
