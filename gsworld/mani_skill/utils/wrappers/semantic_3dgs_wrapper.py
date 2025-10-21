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

import torch
import numpy as np
from torch import nn
import os
import json
from plyfile import PlyData, PlyElement
from utils.system_utils import mkdir_p

# Import the original GaussianModel
try:
    # Try direct import first
    from scene.gaussian_model import GaussianModel
except ImportError:
    # Fallback to sys.path approach
    import sys
    import os
    gaussian_splatting_path = os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', 'submodules', 'gaussian-splatting')
    if gaussian_splatting_path not in sys.path:
        sys.path.append(gaussian_splatting_path)
    from scene.gaussian_model import GaussianModel


class Semantic3DGSWrapper(GaussianModel):
    """
    A wrapper around the original GaussianModel that adds semantic functionality.
    This provides the same features as SemanticGaussianModel without modifying
    the original third-party code.
    """
    
    def __init__(self, sh_degree, optimizer_type="default"):
        super().__init__(sh_degree, optimizer_type)
        # Add semantic parameter
        self._semantics = torch.empty(0)
    
    def capture(self):
        """Override capture to include semantics"""
        base_capture = super().capture()
        return base_capture + (self._semantics,)
    
    def restore(self, model_args, training_args):
        """Override restore to handle semantics"""
        # Extract semantics from the end of model_args
        *base_args, self._semantics = model_args
        super().restore(base_args, training_args)
    
    @property
    def get_semantics(self):
        """Get semantic information"""
        return self._semantics
    
    def create_from_pcd(self, pcd, cam_infos, spatial_lr_scale):
        """Override to initialize semantics"""
        super().create_from_pcd(pcd, cam_infos, spatial_lr_scale)
        # Initialize semantics with zeros
        num_points = self.get_xyz.shape[0]
        self._semantics = torch.zeros((num_points, 1), dtype=torch.float, device="cuda")
    
    def construct_list_of_attributes(self):
        """Override to include semantics in attribute list"""
        l = super().construct_list_of_attributes()
        l.append('semantics')
        return l
    
    def save_ply(self, path):
        """Override save_ply to include semantics"""
        mkdir_p(os.path.dirname(path))
        xyz = self._xyz.detach().cpu().numpy() # (N, 3)
        normals = np.zeros_like(xyz) # (N, 3)
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy() # (N, 3)
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy() # (N, 45)
        opacities = self._opacity.detach().cpu().numpy().reshape(self._opacity.shape[0], -1) # (N, 1)
        scale = self._scaling.detach().cpu().numpy() # (N, 3)
        rotation = self._rotation.detach().cpu().numpy() # (N, 4)
        
        # Include semantics if available
        if hasattr(self, '_semantics') and self._semantics.numel() > 0:
            semantics = self._semantics.detach().cpu().numpy()
        else:
            semantics = np.zeros_like(opacities)

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation, semantics), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)
    
    def load_ply(self, path, use_train_test_exp=False):
        """Override load_ply to handle semantics"""
        plydata = PlyData.read(path)
        if use_train_test_exp:
            exposure_file = os.path.join(os.path.dirname(path), os.pardir, os.pardir, "exposure.json")
            if os.path.exists(exposure_file):
                with open(exposure_file, "r") as f:
                    exposures = json.load(f)
                self.pretrained_exposures = {image_name: torch.FloatTensor(exposures[image_name]).requires_grad_(False).cuda() for image_name in exposures}
                print(f"Pretrained exposures loaded.")
            else:
                print(f"No exposure to be loaded at {exposure_file}")
                self.pretrained_exposures = None

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]
        
        # Try to load semantics, create zeros if not available
        try:
            semantics = np.asarray(plydata.elements[0]["semantics"])[..., np.newaxis]
        except:
            semantics = np.zeros_like(opacities)

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
        assert len(extra_f_names)==3*(self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda",requires_grad=False))
        self._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda", requires_grad=False)).transpose(1, 2).contiguous()
        self._features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda", requires_grad=False)).transpose(1, 2).contiguous()
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda",requires_grad=False))[..., None]
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda",requires_grad=False))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda",requires_grad=False))
        self._semantics = nn.Parameter(torch.tensor(semantics, dtype=torch.float, device="cuda",requires_grad=False))

        self._xyz.requires_grad = False
        self._features_dc = self._features_dc.detach()
        self._features_rest = self._features_rest.detach()
        self._opacity = self._opacity.detach()
        self._scaling.requires_grad = False
        self._rotation.requires_grad = False
        self._semantics.requires_grad = False
        
        self.active_sh_degree = self.max_sh_degree
    
    def prune_points(self, mask):
        """Override prune_points to handle semantics"""
        super().prune_points(mask)
        if hasattr(self, '_semantics') and self._semantics.numel() > 0:
            valid_points_mask = ~mask
            self._semantics = self._semantics[valid_points_mask]
    
    def densification_postfix(self, new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation, new_tmp_radii):
        """Override densification_postfix to handle semantics"""
        super().densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation, new_tmp_radii)
        
        # Add semantics for new points (initialize with zeros)
        if hasattr(self, '_semantics') and self._semantics.numel() > 0:
            num_new_points = new_xyz.shape[0]
            new_semantics = torch.zeros((num_new_points, 1), dtype=torch.float, device="cuda")
            self._semantics = torch.cat((self._semantics, new_semantics), dim=0)
    
    def set_semantics(self, semantics):
        """Set semantic information for all points"""
        if isinstance(semantics, np.ndarray):
            semantics = torch.tensor(semantics, dtype=torch.float, device="cuda")
        self._semantics = semantics
    
    def get_semantic_mask(self, semantic_id):
        """Get a boolean mask for points with specific semantic ID"""
        if not hasattr(self, '_semantics') or self._semantics.numel() == 0:
            return torch.zeros(self.get_xyz.shape[0], dtype=torch.bool, device="cuda")
        return (self._semantics.squeeze() == semantic_id)
    
    def filter_by_semantics(self, semantic_ids):
        """Filter points by semantic IDs"""
        if not hasattr(self, '_semantics') or self._semantics.numel() == 0:
            return torch.ones(self.get_xyz.shape[0], dtype=torch.bool, device="cuda")
        
        if isinstance(semantic_ids, int):
            semantic_ids = [semantic_ids]
        
        mask = torch.zeros(self.get_xyz.shape[0], dtype=torch.bool, device="cuda")
        for semantic_id in semantic_ids:
            mask |= (self._semantics.squeeze() == semantic_id)
        
        return mask
    
    def get_semantic_statistics(self):
        """Get statistics about semantic distribution"""
        if not hasattr(self, '_semantics') or self._semantics.numel() == 0:
            return {}
        
        unique_semantics, counts = torch.unique(self._semantics.squeeze(), return_counts=True)
        return {
            'unique_semantics': unique_semantics.cpu().numpy(),
            'counts': counts.cpu().numpy(),
            'total_points': self.get_xyz.shape[0]
        }
    
    def create_semantic_from_labels(self, labels):
        """Create semantic tensor from label array/list"""
        if isinstance(labels, (list, tuple)):
            labels = np.array(labels)
        if isinstance(labels, np.ndarray):
            labels = torch.tensor(labels, dtype=torch.float, device="cuda")
        
        if labels.dim() == 1:
            labels = labels.unsqueeze(1)
        
        self._semantics = labels
    
    def update_semantics_for_points(self, point_indices, semantic_values):
        """Update semantics for specific points"""
        if not hasattr(self, '_semantics') or self._semantics.numel() == 0:
            # Initialize semantics if not exists
            self._semantics = torch.zeros((self.get_xyz.shape[0], 1), dtype=torch.float, device="cuda")
        
        if isinstance(semantic_values, (int, float)):
            semantic_values = [semantic_values] * len(point_indices)
        
        if isinstance(semantic_values, (list, tuple)):
            semantic_values = torch.tensor(semantic_values, dtype=torch.float, device="cuda")
        
        if semantic_values.dim() == 1:
            semantic_values = semantic_values.unsqueeze(1)
        
        self._semantics[point_indices] = semantic_values
    
    def get_points_by_semantic(self, semantic_id):
        """Get all points with a specific semantic ID"""
        mask = self.get_semantic_mask(semantic_id)
        return {
            'xyz': self.get_xyz[mask],
            'features': self.get_features[mask],
            'opacity': self.get_opacity[mask],
            'scaling': self.get_scaling[mask],
            'rotation': self.get_rotation[mask],
            'indices': torch.where(mask)[0]
        }
