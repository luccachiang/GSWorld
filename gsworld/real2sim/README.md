# Build Your Own Scene with GSWorld (Real2Sim)

This folder contains the Real2Sim utilities used to (1) reconstruct a metric-scale 3D Gaussian Splatting (3DGS)
point cloud from real images, and (2) align it with a simulated robot to transfer per-link semantic labels.

## Prerequisites
- Activate the `gsworld` conda env (see root `README.md`).
- `colmap` available in `PATH` (or pass `--colmap` to the script).
- A capture folder `data/<capture>/images/` containing RGB images.
- An ArUco marker of known size visible during capture (used to recover metric scale).

## 1) Train metric-scale 3DGS (COLMAP + ArUco scaling)
From the repo root:
```bash
conda activate gsworld
bash gsworld/real2sim/scripts/colmap_and_gs.sh --data_dir data/<capture> --gpu 0 --aruco_size 0.100 --export_ply assets/<robot_uid>_assets/<scene_name>.ply
```

Run `bash gsworld/real2sim/scripts/colmap_and_gs.sh --help` for all available arguments.

## 2) Sample points from simulated URDF meshes
```bash
python gsworld/real2sim/scripts/uniform_pcd_from_urdf_visual_mesh.py --robot-uid <robot_uid> --filename <robot_name> --save-pcd
```

This writes files into `assets/<robot_uid>_assets/`, including:
- `<robot_name>.ply`, `<robot_name>_semantics.npy`, `<robot_name>_semantics_trimesh.npy`
- `<robot_name>_visual_mesh.obj` (and `<robot_name>_collision_mesh.obj`)

## 3) Crop the robot from the real 3DGS reconstruction
Open `assets/<robot_uid>_assets/<scene_name>.ply` in a point cloud editor (e.g. Supersplat) and export the
robot-only point cloud to:

`assets/<robot_uid>_assets/<scene_name>_cropped.ply`

## 4) Align sim ↔ real (manual correspondences + ICP)
Pick at least 3 correspondences in each point cloud, then press `Q` to run ICP and print the 4x4 transform matrix:
```bash
python gsworld/real2sim/scripts/open3d_alignment.py --robot-uid <robot_uid> --target <scene_name>_cropped.ply
```

Copy the printed matrix into `gsworld/constants.py` (e.g. `sim2gs_<robot>_trans`).

## 5) Transfer semantic labels to the real 3DGS point cloud
```bash
python gsworld/real2sim/scripts/segment_real_gs.py --robot-uid <robot_uid> --target-name <scene_name>.ply --transform-matrix-name sim2gs_<robot>_trans --bbox-threshold 0.04
```

Output:
- `assets/<robot_uid>_assets/<scene_name>_semantics_gs.npy`
