import numpy as np
import copy
import open3d as o3d
import argparse

def visualize_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp])

def pick_points(pcd):
    print("")
    print(
        "1) Please pick at least three correspondences using [shift + left click]"
    )
    print("   Press [shift + right click] to undo point picking")
    print("2) After picking points, press 'Q' to close the window")
    vis = o3d.visualization.VisualizerWithEditing()
    vis.create_window()
    vis.add_geometry(pcd)
    vis.run()  # user picks points
    vis.destroy_window()
    print("")
    return vis.get_picked_points()

def manual_registration(source_path, target_path):
    source = o3d.io.read_point_cloud(source_path)
    target = o3d.io.read_point_cloud(target_path)
    print("Visualization of two point clouds before manual alignment")
    visualize_registration_result(source, target, np.identity(4))

    # pick points from two point clouds and builds correspondences
    picked_id_source = pick_points(source)
    picked_id_target = pick_points(target)
    assert (len(picked_id_source) >= 3 and len(picked_id_target) >= 3)
    assert (len(picked_id_source) == len(picked_id_target))
    corr = np.zeros((len(picked_id_source), 2))
    corr[:, 0] = picked_id_source
    corr[:, 1] = picked_id_target

    # estimate rough transformation using correspondences
    print("Compute a rough transform using the correspondences given by user")
    p2p = o3d.pipelines.registration.TransformationEstimationPointToPoint(with_scaling=True)
    trans_init = p2p.compute_transformation(source, target,
                                            o3d.utility.Vector2iVector(corr))

    # point-to-point ICP for refinement
    print("Perform point-to-point ICP refinement")
    threshold = 0.03  # 3cm distance threshold
    reg_p2p = o3d.pipelines.registration.registration_icp(
        source, target, threshold, trans_init,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(with_scaling=True))
    print(np.asarray(reg_p2p.correspondence_set).shape)
    print(reg_p2p.inlier_rmse)
    print(reg_p2p.transformation)
    visualize_registration_result(source, target, reg_p2p.transformation)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Manual ICP registration for point clouds")
    parser.add_argument("--robot-uid", "-r", type=str, required=True,
                        help="Robot UID (e.g., galaxea_r1)")
    parser.add_argument("--target", "-t", type=str, required=True,
                        help="Target point cloud filename (e.g., cropped_arm.ply)")
    
    args = parser.parse_args()
    
    # Setup paths
    import os
    from pathlib import Path
    script_dir = Path(__file__).parent
    assets_dir = script_dir.parent.parent / "assets"
    robot_dir = assets_dir / f"{args.robot_uid}_assets"
    
    # Auto-find source URDF file (file without date prefix)
    ply_files = [f for f in robot_dir.glob("*.ply") if "semantic" not in f.name.lower()]
    urdf_candidates = [f for f in ply_files if not f.stem[0].isdigit()]
    
    if len(urdf_candidates) != 1:
        parser.error(f"Expected 1 URDF .ply file in {robot_dir}, found {len(urdf_candidates)}: {[f.name for f in urdf_candidates]}")
    
    source_path = urdf_candidates[0]
    print(f"Auto-found source URDF: {source_path}")
    
    # Target file should be in the same robot directory
    target_path = robot_dir / args.target
    if not target_path.exists():
        parser.error(f"Target file not found: {target_path}")
    
    print(f"Using target file: {target_path}")
    
    manual_registration(str(source_path), str(target_path))
