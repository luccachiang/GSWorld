"""
Instantiates a empty environment with a floor, and attempts to place any given robot in there (default origin).

This scripts is to extract dense pointcloud from urdf with maniskill.
"""

import argparse
import numpy as np
import gymnasium as gym
import mani_skill
import trimesh
from typing import TYPE_CHECKING, Dict, List, Tuple, Union
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.utils.geometry.trimesh_utils import get_component_meshes, get_render_body_meshes
import open3d as o3d
import matplotlib as mpl
from gsworld.constants import robot_scan_qpos
from gsworld.constants import *

def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--robot-uid", type=str, default="fr3_umi_wrist435", help="The id of the robot to place in the environment")
    parser.add_argument("-b", "--sim-backend", type=str, default="auto", help="Which simulation backend to use. Can be 'auto', 'cpu', 'gpu'")
    parser.add_argument("-c", "--control-mode", type=str, default="pd_joint_pos", help="The control mode to use. Note that for new robots being implemented if the _controller_configs is not implemented in the selected robot, we by default provide two default controllers, 'pd_joint_pos' and 'pd_joint_delta_pos' ")
    parser.add_argument("-k", "--keyframe", type=str, help="The name of the keyframe of the robot to display")
    parser.add_argument("-f", "--filename", type=str, default="test", help="The name of the pcd file saved")
    parser.add_argument("--shader", default="default", type=str, help="Change shader used for rendering. Default is 'default' which is very fast. Can also be 'rt' for ray tracing and generating photo-realistic renders. Can also be 'rt-fast' for a faster but lower quality ray-traced renderer")
    parser.add_argument("--keyframe-actions", action="store_true", help="Whether to use the selected keyframe to set joint targets to try and hold the robot in its position")
    parser.add_argument("--random-actions", action="store_true", help="Whether to sample random actions to control the agent. If False, no control signals are sent and it is just rendering.")
    parser.add_argument("--none-actions", action="store_true", help="If set, then the scene and rendering will update each timestep but no joints will be controlled via code. You can use this to control the robot freely via the GUI.")
    parser.add_argument("--zero-actions", action="store_true", help="Whether to send zero actions to the robot. If False, no control signals are sent and it is just rendering.")
    parser.add_argument("--sim-freq", type=int, default=100, help="Simulation frequency")
    parser.add_argument("--control-freq", type=int, default=20, help="Control frequency")
    parser.add_argument("--save-pcd", action="store_true", help="Whether to save pcd or not")
    parser.add_argument(
        "-s",
        "--seed",
        type=int,
        help="Seed the random actions and environment. Default is no seed",
    )
    args = parser.parse_args()
    return args

def merge_meshes(meshes: List[trimesh.Trimesh], process=True):
    n, vs, fs = 0, [], []
    for mesh in meshes:
        v, f = mesh.vertices, mesh.faces
        vs.append(v)
        fs.append(f + n)
        n = n + v.shape[0]
    if n:
        # print(n)
        # print(np.vstack(vs).shape)
        # print(np.vstack(fs).shape)
        return trimesh.Trimesh(np.vstack(vs), np.vstack(fs), process=process)
    else:
        return None

def get_visual_meshes_w_link_labels(env, to_world_frame: bool = True, first_only: bool = False, with_semantics: bool = True) -> List[trimesh.Trimesh]:
        """
        CHANGELOG
        - 2024-11-19(roger): adapt from https://github.com/haosulab/ManiSkill/blob/a105ffa0c81e26909239c54528c2af21682be28b/mani_skill/utils/structs/articulation.py#L315

        Returns the collision mesh of each managed articulation object. Note results of this are not cached or optimized at the moment
        so this function can be slow if called too often

        Args:
            to_world_frame (bool): Whether to transform the collision mesh pose to the world frame
            first_only (bool): Whether to return the collision mesh of just the first articulation managed by this object. If True,
                this also returns a single Trimesh.Mesh object instead of a list
        """
        robot = env.agent.robot
        scene = env.scene
        assert (
            not robot.merged
        ), "Currently you cannot fetch collision meshes of merged articulations as merged articulations only share a root link"
        if robot.scene.gpu_sim_enabled:
            assert (
                robot.scene._gpu_sim_initialized
            ), "During GPU simulation link pose data is not accessible until after \
                initialization, and link poses are needed to get the correct collision mesh of an entire articulation"
        else:
            robot._objs[0].pose = robot._objs[0].pose
        # TODO (stao): Can we have a batched version of trimesh?
        meshes: List[trimesh.Trimesh] = []
        v_meshes: List[trimesh.Trimesh] = []

        assert len(robot.get_links()) == len(env.scene._get_all_render_bodies()), f"Link and visual mesh number mismatch, {len(robot.get_links())}, {len(env.scene._get_all_render_bodies())}"

        # visual_meshes = []
        # for idx, rb in enumerate(env.scene._get_all_render_bodies()): # 14
        #     if rb[0] is None:
        #         visual_meshes.append(rb[0])
        #     else:
        #         visual_meshes.append(merge_meshes(get_render_body_meshes(rb[0])))
        # visual_mesh = merge_meshes(visual_meshes)

        scene_render_bodies = env.scene._get_all_render_bodies()
        assert len(robot._objs) == 1, f"Only support one robot right now, got {len(robot._objs)}."
        for i, art in enumerate(robot._objs): # gq: only one
            art_meshes = []
            art_part_semantics = []
            visual_meshes = []
            for link_idx, link in enumerate(art.links): # 14
                link_mesh = merge_meshes(get_component_meshes(link))
                visual_mesh = scene_render_bodies[link_idx][0]
                if visual_mesh is not None:
                    visual_mesh = merge_meshes(get_render_body_meshes(visual_mesh))
                if link_mesh is not None:
                    if to_world_frame:
                        pose = robot.links[link.index].pose[i]
                        link_mesh.apply_transform(pose.sp.to_transformation_matrix())
                        visual_mesh.apply_transform(pose.sp.to_transformation_matrix())
                    art_part_semantics.append(link_idx)
                    art_meshes.append(link_mesh)
                    visual_meshes.append(visual_mesh)
            mesh = merge_meshes(art_meshes)
            v_mesh = merge_meshes(visual_meshes, process=False) # gq we lose 46 points here, set process to False to prevent this problem

            # Merge semantics
            semantic_list = []
            for i, part_mesh in enumerate(visual_meshes):
                semantic_list += [art_part_semantics[i]] * len(part_mesh.vertices)
            assert v_mesh.vertices.shape[0] == len(semantic_list), f"Visual mesh vertice and semantic number mismatch, {v_mesh.vertices.shape[0]}, {len(semantic_list)}" # TODO mismatch here

            if with_semantics:
                meshes.append((mesh, semantic_list))
                v_meshes.append((v_mesh, semantic_list))
            else:
                meshes.append(mesh)
                v_meshes.append(v_mesh)
            if first_only:
                break
        if to_world_frame:
            mat = robot.pose
            for i, mesh in enumerate(meshes):
                v_mesh = v_meshes[i]
                if with_semantics:
                    mesh = mesh[0]
                    v_mesh = v_mesh[0]
                if mat is not None:
                    if len(mat) > 1:
                        mesh.apply_transform(mat[i].sp.to_transformation_matrix())
                        v_mesh.apply_transform(mat[i].sp.to_transformation_matrix())
                    else:
                        mesh.apply_transform(mat.sp.to_transformation_matrix())
                        v_mesh.apply_transform(mat.sp.to_transformation_matrix())
        if first_only:

            return v_meshes[0], meshes[0], visual_meshes[0]
        return v_meshes, meshes, visual_meshes # also return a list of link visual mesh

def sample_points_from_links(visual_meshes, points_per_link=1000, total_points=300000):
    """
    Sample points from each link mesh separately to preserve semantic labels.
    
    Args:
        visual_meshes: List of (trimesh.Trimesh, semantic_idx) tuples for each link
        points_per_link: Number of points to sample per link, or None to use proportional sampling
    
    Returns:
        combined_pcd: Combined point cloud
        semantic_indices: Semantic indices for all points
    """
    
    all_points = []
    all_semantics = []
    
    # First pass: calculate total surface area for proportional sampling
    if points_per_link is None:
        total_area = 0
        link_areas = []
        for link_mesh, _ in visual_meshes:
            if link_mesh is not None:
                area = np.sum([triangle_area(link_mesh.vertices[face]) 
                             for face in link_mesh.faces])
                link_areas.append(area)
                total_area += area
            else:
                link_areas.append(0)
    
    for idx, (link_mesh, semantic_idx) in enumerate(visual_meshes):
        if link_mesh is None:
            continue
            
        # Calculate number of points for this link
        if points_per_link is None:
            # Proportional sampling based on surface area
            n_points = int((link_areas[idx] / total_area) * total_points)
            if n_points == 0:
                continue
        else:
            n_points = points_per_link
            
        # Convert trimesh to open3d mesh
        o3d_mesh = o3d.geometry.TriangleMesh()
        o3d_mesh.vertices = o3d.utility.Vector3dVector(link_mesh.vertices)
        o3d_mesh.triangles = o3d.utility.Vector3iVector(np.asarray(link_mesh.faces, dtype=np.int32))
        o3d_mesh.compute_vertex_normals()
        
        # Sample points from this link's mesh
        pcd = o3d_mesh.sample_points_uniformly(number_of_points=n_points)
        points = np.asarray(pcd.points)
        
        # Add points and semantics
        all_points.append(points)
        all_semantics.extend([semantic_idx] * len(points))
    
    # Combine all points
    combined_points = np.vstack(all_points)
    combined_pcd = o3d.geometry.PointCloud()
    combined_pcd.points = o3d.utility.Vector3dVector(combined_points)
    
    return combined_pcd, np.array(all_semantics)

def triangle_area(vertices):
    """Calculate area of a triangle given its vertices."""
    v0, v1, v2 = vertices
    return 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0))

def save_semantic_pointcloud(pcd, dense_semantic_indices, trimesh_semantic_indices, filename="semantic_pointcloud.ply"):
    o3d.io.write_point_cloud(filename, pcd)
    print(f"Successfully save pcd to {filename}")
    np.save(filename.replace('.ply', '_semantics.npy'), dense_semantic_indices)
    print(f"Successfully save dense semantic indices to {filename.replace('.ply', '_semantics.npy')}")
    np.save(filename.replace('.ply', '_semantics_trimesh.npy'), trimesh_semantic_indices)
    print(f"Successfully save trimesh semantic indices to {filename.replace('.ply', '_semantics_trimesh.npy')}")

def load_semantic_pointcloud(ply_filename):
    """
    Load point cloud and its semantic indices.
    
    Args:
        ply_filename: path to the PLY file
        
    Returns:
        points: numpy array of point coordinates
        colors: numpy array of point colors
        semantic_indices: numpy array of semantic indices
    """
    # Load point cloud
    pcd = o3d.io.read_point_cloud(ply_filename)
    
    # # Get points and colors
    # points = np.asarray(pcd.points)
    # colors = np.asarray(pcd.colors)
    
    # Load semantic indices
    semantics_filename = ply_filename.replace('.ply', '_semantics.npy')
    semantic_indices = np.load(semantics_filename)
    
    return pcd, semantic_indices

# Visualize results
def visualize_comparison(mesh, pcd):
    # Create separate windows for mesh and point cloud
    print("Displaying original mesh...")
    o3d.visualization.draw_geometries([mesh])
    
    print("Displaying sampled point cloud...")
    o3d.visualization.draw_geometries([pcd])
    
    # Optional: Display both side by side
    print("Displaying both together...")
    mesh_copy = o3d.geometry.TriangleMesh(mesh)
    mesh_copy.paint_uniform_color([0.8, 0.8, 0.8])  # Light gray
    o3d.visualization.draw_geometries([mesh_copy, pcd])

if __name__ == "__main__":

    import mani_skill.envs

    # ========== From mani_skill/examples/demo_robot.py ==========

    args = parse_args()
    file_name = args.filename # 'fr3_umi'
    env = gym.make(
        "Empty-v1",
        obs_mode="none",
        reward_mode="none",
        enable_shadow=True,
        control_mode=args.control_mode,
        robot_uids=args.robot_uid,
        sensor_configs=dict(shader_pack=args.shader),
        human_render_camera_configs=dict(shader_pack=args.shader),
        viewer_camera_configs=dict(shader_pack=args.shader),
        render_mode="human",
        sim_config=dict(sim_freq=args.sim_freq, control_freq=args.control_freq),
        sim_backend=args.sim_backend,
    )
    env.reset(seed=0)
    env: BaseEnv = env.unwrapped
    print(f"Selected robot {args.robot_uid}. Control mode: {args.control_mode}")
    print("Selected Robot has the following keyframes to view: ")
    print(env.agent.keyframes.keys())
    # qpos_vec = env.agent.robot.qpos * 0
    # Use hardcoded https://github.com/jimazeyu/franka_grasp_baseline/raw/main/assets/ui.png
    # qpos_vec = np.array([0.572, 0.860, -0.603, -1.686, 0.897, 1.966, -0.514, 0, 0], dtype=np.float32)
    qpos_vec = robot_scan_qpos[args.robot_uid]
    env.agent.robot.set_qpos(qpos_vec)
    visual_meshes, collision_meshes, _ = get_visual_meshes_w_link_labels(env, to_world_frame=True, first_only=False, with_semantics=True)
    visual_mesh, semantic_idx = visual_meshes[0] # trimesh, list 1023 of idx
    print(f"Visual mesh vertice number: {len(visual_mesh.vertices)}")
    os.makedirs(os.path.join(ASSET_DIR, f"{args.robot_uid}_assets"), exist_ok=True)
    collision_meshes[0][0].export(os.path.join(ASSET_DIR, f"{args.robot_uid}_assets/{file_name}_collision_mesh.obj")) # similar to visual mesh
    visual_mesh.export(os.path.join(ASSET_DIR, f"{args.robot_uid}_assets/{file_name}_visual_mesh.obj"))

    # TODO potential visual mesh
    # o3d.visualization.draw_geometries([merge_meshes(get_render_body_meshes(env.scene._get_all_render_bodies()[1][0]))])
    # for idx, rb in enumerate(env.scene._get_all_render_bodies()):
    #     print(idx, rb)
    #     if env.scene._get_all_render_bodies()[idx][0] is None:
    #         continue
    #     collision_mesh = merge_meshes(get_render_body_meshes(env.scene._get_all_render_bodies()[idx][0]))
    #     collision_mesh.export(f"{idx}.obj")
    #     mesh = o3d.io.read_triangle_mesh(f"{idx}.obj")
    #     o3d.visualization.draw_geometries([mesh])

    # Load and prepare mesh
    mesh = o3d.io.read_triangle_mesh(os.path.join(ASSET_DIR, f"{args.robot_uid}_assets/{file_name}_visual_mesh.obj"))
    mesh.compute_vertex_normals()
    # o3d.visualization.draw_geometries([mesh])

    # Create dense point cloud
    # gq: visual mesh already dense
    visual_meshes_list = []
    for link_idx in range(len(env.agent.robot.links)):
        print(link_idx, env.agent.robot.links[link_idx].name)
        if env.scene._get_all_render_bodies()[link_idx][0] is not None:
            visual_mesh = merge_meshes(get_render_body_meshes(env.scene._get_all_render_bodies()[link_idx][0]))
            visual_mesh.apply_transform(env.agent.robot.links[link_idx].pose.sp.to_transformation_matrix())
            visual_meshes_list.append((visual_mesh, link_idx))
        else:
            visual_meshes_list.append((None, link_idx))
    dense_pcd, dense_semantic_idx = sample_points_from_links(visual_meshes_list, points_per_link=None, total_points=300000) # TODO may slightly get less pts, also do we need to remove pts inside the mesh?
    print(f"Final pcd pt number {len(dense_pcd.points)}")
    # dense_semantic_idx = semantic_idx
    cmap = mpl.cm.tab20
    norm = mpl.colors.Normalize(vmin=min(dense_semantic_idx), vmax=max(dense_semantic_idx)) # norm to [0,1]
    mapped_colors = cmap(norm(dense_semantic_idx))

    # dense_pcd = o3d.geometry.PointCloud()
    # dense_pcd.points = o3d.utility.Vector3dVector(visual_mesh.vertices)
    dense_pcd.colors = o3d.utility.Vector3dVector(mapped_colors[:, :3]) # number_of_points, 4 [r,g,b,1]

    # save and load
    if args.save_pcd:
        save_semantic_pointcloud(dense_pcd, dense_semantic_idx, semantic_idx, os.path.join(ASSET_DIR, f"{args.robot_uid}_assets/{file_name}.ply"))
        pcd, sem_idx = load_semantic_pointcloud(os.path.join(ASSET_DIR, f"{args.robot_uid}_assets/{file_name}.ply"))
        visualize_comparison(mesh, pcd)
    else:
        visualize_comparison(mesh, dense_pcd)