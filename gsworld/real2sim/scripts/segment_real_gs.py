# This script is used to segment the real-world 3dgs ply.
# We would like to first align sim and real pts,
# and then use some voting algo to segment, given sim pcd has semantic index.

import os
import numpy as np
import argparse

from gsworld.utils.pcd_utils import load_ply_to_splats, load_semantic_pointcloud, read_ply_points, create_o3d_point_cloud, save_ply
from real2sim_utils import transfer_labels_with_bbox, visualize_results, get_semantic_bounding_boxes
import trimesh
from gsworld.constants import sim2gs_arm_trans, sim2gs_xarm_trans, ASSET_DIR, sim2gs_r1_trans
import torch

# Example usage
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Transfer semantic labels from URDF to real Gaussian Splatting")
    parser.add_argument("--robot-uid", "-r", type=str,
                        help="Robot UID (e.g., galaxea_r1)")
    parser.add_argument("--target-name", "-t", type=str,
                        help="Target GS scan filename with .ply extension (e.g., 0425_r1.ply)")
    parser.add_argument("--transform-matrix-name", "-m", type=str,
                        help="Transformation matrix name from gsworld.constants (e.g., sim2gs_r1_trans)")
    parser.add_argument("--bbox-threshold", "-b", type=float,
                        help="Bounding box distance threshold (e.g., 0.04)")
    
    args = parser.parse_args()
    
    # If arguments provided, use them; otherwise use hardcoded config
    if args.robot_uid and args.target_name and args.transform_matrix_name:
        from pathlib import Path
        import gsworld.constants as constants
        
        robot_assets_dir = f'{args.robot_uid}_assets'
        target_name = args.target_name[:-4]
        bbox_threshold = args.bbox_threshold if args.bbox_threshold else 0.04
        
        # Auto-find URDF filename (file without date prefix)
        robot_dir = Path(ASSET_DIR) / robot_assets_dir
        ply_files = [f for f in robot_dir.glob("*.ply") if "semantic" not in f.name.lower()]
        urdf_candidates = [f for f in ply_files if not f.stem[0].isdigit() and f.stem != target_name]
        
        if len(urdf_candidates) != 1:
            raise ValueError(f"Expected 1 URDF .ply file in {robot_dir}, found {len(urdf_candidates)}: {[f.name for f in urdf_candidates]}")
        
        file_name = urdf_candidates[0].stem
        transformation_matrix = getattr(constants, args.transform_matrix_name)
        
        print(f"Using command-line arguments:")
        print(f"  Robot: {robot_assets_dir}")
        print(f"  Auto-found URDF file: {file_name}")
        print(f"  Target name: {target_name}")
        print(f"  Bbox threshold: {bbox_threshold}")
    else:
        # ----- OPTION 1: FR3 -----
        # file_name = 'fr3_umi'
        # robot_assets_dir = 'fr3_umi_wrist435_cam_mount_assets'
        # target_name = '0411_fr3_haoran_v1'
        # transformation_matrix = sim2gs_arm_trans
        # bbox_threshold = 0.025

        # ----- OPTION 2: XArm6 -----
        # file_name = 'xarm6_uf_gripper_wrist435'
        # robot_assets_dir = 'xarm6_uf_gripper_wrist435_assets'
        # target_name = 'xarm6_v3'
        # transformation_matrix = sim2gs_xarm_trans
        # bbox_threshold = 0.04

        # ----- OPTION 3: R1 (currently active) -----
        file_name = 'r1'
        robot_assets_dir = 'galaxea_r1_assets'
        target_name = '0425_r1'
        transformation_matrix = sim2gs_r1_trans
        bbox_threshold = 0.04
    
    source_ply = os.path.join(ASSET_DIR, f'{robot_assets_dir}/{file_name}.ply')  # URDF sampled PLY
    target_ply = os.path.join(ASSET_DIR, f'{robot_assets_dir}/{target_name}.ply')  # Real GS scene
    mesh = trimesh.load(os.path.join(ASSET_DIR, f"{robot_assets_dir}/{file_name}_visual_mesh.obj"), process=False)
    semantic_idx = np.load(os.path.join(ASSET_DIR, f'{robot_assets_dir}/{file_name}_semantics_trimesh.npy'))

    # Load your data as before
    source_pcd, src_sem_idx = load_semantic_pointcloud(source_ply)
    target_points = read_ply_points(target_ply)
    target_pcd = create_o3d_point_cloud(target_points)
    
    # Get semantic bounding boxes (using your existing code)
    semantic_bboxes = get_semantic_bounding_boxes(mesh, semantic_idx)
    
    # Transfer labels with bbox validation
    new_labels, distances = transfer_labels_with_bbox(
        np.asarray(source_pcd.points),
        src_sem_idx,
        target_points,
        transformation_matrix,
        semantic_bboxes,
        bbox_distance_threshold=bbox_threshold,
    )
    
    # Visualize results
    visualize_results(source_pcd, target_pcd, new_labels, transformation_matrix)

    # Save the new labels to robot assets directory
    output_path = os.path.join(ASSET_DIR, f'{robot_assets_dir}/{target_name}_semantics_gs.npy')
    np.save(output_path, new_labels)
    print(f"Successfully saved real-world semantics to {output_path}")
    