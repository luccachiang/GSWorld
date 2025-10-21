import os
import open3d as o3d
import numpy as np
from plyfile import PlyData, PlyElement
import torch
import json
import cv2

def load_semantic_pointcloud(ply_filename):
    # Load point cloud
    pcd = o3d.io.read_point_cloud(ply_filename)
    
    # Load semantic indices
    semantics_filename = ply_filename.replace('.ply', '_semantics.npy')
    semantic_indices = np.load(semantics_filename)
    
    return pcd, semantic_indices

# Previous utility functions remain the same
def read_ply_points(ply_path):
    """Extract XYZ coordinates from PLY file."""
    plydata = PlyData.read(ply_path)
    vertex = plydata['vertex']
    points = np.column_stack((vertex['x'], vertex['y'], vertex['z']))
    return points

def create_o3d_point_cloud(points):
    """Convert numpy array to Open3D point cloud"""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    return pcd

def construct_list_of_attributes(splats):
    l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
    # All channels except the 3 DC
    for i in range(splats["sh0"].shape[1]*splats["sh0"].shape[2]):
        l.append('f_dc_{}'.format(i))
    for i in range(splats["shN"].shape[1]*splats["shN"].shape[2]):
        l.append('f_rest_{}'.format(i))
    l.append('opacity')
    for i in range(splats["scales"].shape[1]):
        l.append('scale_{}'.format(i))
    for i in range(splats["quats"].shape[1]):
        l.append('rot_{}'.format(i))
    if "semantics" in splats:
        l.append('semantics')
    return l

def save_ply(splats, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)

    xyz = splats["means"].detach().cpu().numpy()
    normals = np.zeros_like(xyz)
    f_dc = splats["sh0"].detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
    f_rest = splats["shN"].detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()

    opacities = splats["opacities"].detach().reshape(splats["opacities"].shape[0], -1).cpu().numpy()
    scale = splats["scales"].detach().cpu().numpy()
    rotation = splats["quats"].detach().cpu().numpy()

    dtype_full = [(attribute, 'f4') for attribute in construct_list_of_attributes(splats)]

    elements = np.empty(xyz.shape[0], dtype=dtype_full)
    if "semantics" in splats:
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation, splats["semantics"].detach().reshape(splats["semantics"].shape[0], -1).cpu().numpy()), axis=1)
    else:
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
    elements[:] = list(map(tuple, attributes)) # (1468850, 63)
    el = PlyElement.describe(elements, 'vertex')
    PlyData([el]).write(path)

def load_ply_to_splats(path, use_train_test_exp = False, device="cpu"):
    '''
    Code adapted from official gaussian splatting repo.
    '''
    plydata = PlyData.read(path)
    max_sh_degree = 3

    xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                    np.asarray(plydata.elements[0]["y"]),
                    np.asarray(plydata.elements[0]["z"])),  axis=1)
    opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

    features_dc = np.zeros((xyz.shape[0], 3, 1))
    features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
    features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
    features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

    extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
    extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
    assert len(extra_f_names)==3*(max_sh_degree + 1) ** 2 - 3
    features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
    for idx, attr_name in enumerate(extra_f_names):
        features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
    # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
    features_extra = features_extra.reshape((features_extra.shape[0], 3, (max_sh_degree + 1) ** 2 - 1))

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

    # process semantic
    if "semantics" in plydata.elements[0]:
        print(f"Loading pre-stored semantic labels")
        semantic = torch.from_numpy(plydata.elements[0]["semantics"].astype(np.int32)).to(device)
    else:
        print(f"Gaussian model does not have semantics. Initializing them to zeros.")
        semantic = torch.zeros_like(torch.from_numpy(opacities), dtype=torch.int32).to(device) # .any() returns false

    # Create splats dictionary
    splats = {
        "means": torch.tensor(xyz, dtype=torch.float, device=device),
        "sh0": torch.tensor(features_dc, dtype=torch.float, device=device).contiguous(),
        "shN": torch.tensor(features_extra, dtype=torch.float, device=device).contiguous(),
        "scales": torch.tensor(scales, dtype=torch.float, device=device),
        "quats": torch.tensor(rots, dtype=torch.float, device=device),
        "opacities": torch.tensor(opacities, dtype=torch.float, device=device),
        "semantics": semantic # gs point label
    }
    
    return splats

def save_checkpoint(splats, output_path, step=0):
    """Save the splats data as a checkpoint file."""
    checkpoint = {
        'step': step,
        'splats': splats
    }
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.save(checkpoint, output_path)

def read_gaussian(
    gaussian_folder,
    charuco_dict,
    board,
    dist_coeffs,
    RECOMPUTE=False,
    reg=None,
    marker_size=0.1,
    sample_num=1000,
    mesh=None,
):
    gaussian_cameras_json_path = gaussian_folder / "cameras.json"
    with open(gaussian_cameras_json_path, "r") as f:
        gaussian_cameras = json.load(f)
    if RECOMPUTE or not os.path.exists(
        gaussian_folder / "to_marker_22222_transform.npy"
    ):
        gaussian_camera_poses = []
        marker_camera_poses = []
        image_names = []
        for gaussian_camera in gaussian_cameras:
            position = gaussian_camera["position"]
            rotation = gaussian_camera["rotation"]
            fx = gaussian_camera["fx"]
            fy = gaussian_camera["fy"]
            cx = gaussian_camera["width"] / 2
            cy = gaussian_camera["height"] / 2
            gaussian_camera_pose = np.eye(4)
            gaussian_camera_pose[:3, :3] = np.array(rotation)
            gaussian_camera_pose[:3, 3] = np.array(position)
            # gaussian_camera_pose = np.linalg.inv(gaussian_camera_pose)
            camera_params = [fx, fy, cx, cy]
            image_path = f"{gaussian_camera['image_path']}"
            image = cv2.imread(str(image_path))
            intrinsics_matrix = np.array(
                [
                    [camera_params[0], 0, camera_params[2]],
                    [0, camera_params[1], camera_params[3]],
                    [0, 0, 1],
                ]
            )
            dist_coeffs = (
                np.zeros((5,)) if dist_coeffs is None else np.array(dist_coeffs)
            )
            marker_pose = estimate_pose(
                image, charuco_dict, intrinsics_matrix, dist_coeffs, board
            )
            gaussian_camera_poses.append(gaussian_camera_pose)
            marker_camera_poses.append(marker_pose)
            image_names.append(gaussian_camera["img_name"])
        gaussian2marker, camera_poses_in_marker = get_transform_between_two_frames2(
            gaussian_camera_poses,
            marker_camera_poses,
            transform_type=True,
            img_names=image_names,
            reg=reg,
            sample_num=sample_num,
        )
        np.save(gaussian_folder / "gs_to_marker.npy", gaussian2marker)
    else:
        gaussian_camera_poses = []
        camera_poses_in_marker = []
        gaussian2marker = np.load(gaussian_folder / "gs_to_marker.npy")
        for gaussian_camera in gaussian_cameras:
            position = gaussian_camera["position"]
            rotation = gaussian_camera["rotation"]
            fx = gaussian_camera["fx"]
            fy = gaussian_camera["fy"]
            cx = gaussian_camera["width"] / 2
            cy = gaussian_camera["height"] / 2
            gaussian_camera_pose = np.eye(4)
            gaussian_camera_pose[:3, :3] = np.array(rotation)
            gaussian_camera_pose[:3, 3] = np.array(position)
            camera_params = [fx, fy, cx, cy]
            gaussian_camera_poses.append(gaussian_camera_pose)
            camera_poses_in_marker.append(gaussian2marker @ gaussian_camera_pose)
        marker_camera_poses = None
    if mesh is not None:
        mesh.transform(gaussian2marker)
    np.save(gaussian_folder / "gs_to_marker.npy", gaussian2marker)
    return camera_poses_in_marker, marker_camera_poses, mesh
    # return gaussian_camera_poses, marker_camera_poses, mesh

def extract_rigid_transform(M: torch.Tensor):
    if M.ndim == 2 and M.shape == (4, 4):
        M = M.unsqueeze(0)  # promote to (1,4,4)
        single_input = True
    elif M.ndim == 3 and M.shape[1:] == (4, 4):
        single_input = False
    else:
        raise ValueError("Input must be (4,4) or (N,4,4)")

    A = M[:, :3, :3]   # (N,3,3)
    t = M[:, :3, 3]    # (N,3)

    # SVD for scale estimation
    U, S, Vh = torch.linalg.svd(A)
    scales = S.mean(dim=1)  # average singular values (uniform scale assumption)

    # Polar decomposition to get rigid rotation
    # Method: R = U Vh
    R_rigid = torch.matmul(U, Vh)

    # Build rigid 4x4 matrices
    M_rigid = torch.eye(4, device=M.device, dtype=M.dtype).repeat(M.shape[0], 1, 1)
    M_rigid[:, :3, :3] = R_rigid
    M_rigid[:, :3, 3] = t

    if single_input:
        return M_rigid[0], scales[0], R_rigid[0], t[0]
    else:
        return M_rigid, scales, R_rigid, t
