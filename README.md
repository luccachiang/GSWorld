# GSWorld

![GSWorld](resources/teaser.webp)

## Assets Download
Go to the [HuggingFace repo](https://huggingface.co/datasets/GqJiang/gsworld) to download the assets, unzip, and put it under `assets/`.

## Installation
```bash
git submodule update --init --recursive

conda create -y -n gsworld python=3.11 && conda activate gsworld
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124
conda install -c "nvidia/label/cuda-12.4.0" cuda-toolkit -y

pip install plyfile open3d

pip install -e <path2gsworld>

pip install mani_skill==3.0.0b15

# install gaussian splatting
cd submodules/gaussian-splatting
git submodule update --init --recursive

# for real2sim 3dgs
# using conda to install colmap should also be feasible
sudo apt update
sudo apt install colmap
pip install opencv-python opencv-contrib-python==4.6.0.66 matplotlib open3d colmap-wrapper pycolmap==0.5.0 ipykernel 
```

Modify the frustum, from 0.2f to 0.05f in [auxiliary.h](submodules/gaussian-splatting/submodules/diff-gaussian-rasterization/cuda_rasterizer/auxiliary.h)

```bash
# if rebuilding, first rm -rf build/
pip install -e submodules/diff-gaussian-rasterization && pip install -e submodules/simple-knn && pip install -e submodules/fused-ssim
```

## Examples from GSWorld
### Random actions
```bash
cd examples/maniskill

# xarm6
python gsworld_rand_action_tabletop.py --robot_uids xarm6_uf_gripper --scene_cfg_name xarm6_align --record_dir ./exp_log/xarm6_align --ep_len 10 --env_id AlignXArmEnv-v1
python gsworld_rand_action_tabletop.py --robot_uids xarm6_uf_gripper --scene_cfg_name xarm6_rot_banana --record_dir ./exp_log/xarm6_banana --ep_len 10 --env_id BananaRotationXArmEnv-v1
python gsworld_rand_action_tabletop.py --robot_uids xarm6_uf_gripper --scene_cfg_name xarm6_spoon2board --record_dir ./exp_log/xarm6_spoon --ep_len 10 --env_id SpoonOnBoardXArmEnv-v1

# fr3
python gsworld_rand_action_tabletop.py --robot_uids fr3_umi --scene_cfg_name fr3_align --record_dir ./exp_log/fr3_align --ep_len 10 --env_id AlignFr3Env-v1
python gsworld_rand_action_tabletop.py --robot_uids fr3_umi --scene_cfg_name fr3_pnp_box --record_dir ./exp_log/fr3_pnp_box --ep_len 10 --env_id PnpBoxFr3Env-v1
python gsworld_rand_action_tabletop.py --robot_uids fr3_umi --scene_cfg_name fr3_pour --record_dir ./exp_log/fr3_pour --ep_len 10 --env_id PourMustardFr3Env-v1
python gsworld_rand_action_tabletop.py --robot_uids fr3_umi --scene_cfg_name fr3_stack --record_dir ./exp_log/fr3_stack --ep_len 10 --env_id StackFr3Env-v1
```

### Motion Plannings
```bash
cd gsworld/mani_skill/examples/motionplanning
# xarm6
python xarm6/run_with_gs.py -e AlignXArmEnv-v1 -n 1 --vis --scene_cfg_name xarm6_align
python xarm6/run_with_gs.py -e BananaRotationXArmEnv-v1 -n 1 --vis --scene_cfg_name xarm6_rot_banana
python xarm6/run_with_gs.py -e SpoonOnBoardXArmEnv-v1 -n 1 --vis --scene_cfg_name xarm6_spoon2board

# franka
python franka/run_with_gs.py -e AlignFr3Env-v1 -n 1 --vis --scene_cfg_name fr3_align
python franka/run_with_gs.py -e PnpBoxFr3Env-v1 -n 1 --vis --scene_cfg_name fr3_pnp_box
python franka/run_with_gs.py -e PourMustardFr3Env-v1 -n 1 --vis --scene_cfg_name fr3_pour
python franka/run_with_gs.py -e StackFr3Env-v1 -n 1 --vis --scene_cfg_name fr3_stack
```

## Build Your Own Scene with GSWorld
The Real2Sim pipeline (`gsworld/real2sim`) reconstructs a metric-scale 3D Gaussian Splatting (3DGS) scene from real
images (COLMAP + ArUco scaling), then aligns it to the simulation robot to transfer per-link semantic labels.

```bash
# 1) Train a metric-scale 3DGS scene from real images (COLMAP + ArUco scaling)
#    Expected input: data/<your_capture>/images/*.png (or *.jpg)
bash gsworld/real2sim/scripts/colmap_and_gs.sh --data_dir data/<your_capture> --gpu 0 --aruco_size 0.100 --export_ply assets/<robot_uid>_assets/<scene_name>.ply

# 2) Sample a semantic point cloud from the simulation robot URDF meshes
python gsworld/real2sim/scripts/uniform_pcd_from_urdf_visual_mesh.py --robot-uid <robot_uid> --filename <robot_name> --save-pcd

# 3) Crop the robot from the reconstructed point cloud (e.g. Supersplat) and save:
#    assets/<robot_uid>_assets/<scene_name>_cropped.ply

# 4) Manual alignment + ICP refinement (prints a 4x4 transform matrix)
python gsworld/real2sim/scripts/open3d_alignment.py --robot-uid <robot_uid> --target <scene_name>_cropped.ply

# 5) Copy the printed matrix into gsworld/constants.py (e.g. sim2gs_<robot>_trans), then segment:
python gsworld/real2sim/scripts/segment_real_gs.py --robot-uid <robot_uid> --target-name <scene_name>.ply --transform-matrix-name sim2gs_<robot>_trans --bbox-threshold 0.04
```

See `gsworld/real2sim/README.md` for a more detailed walkthrough and expected file layout.

## BibTex
If you find this project helpful, please give us a star and cite:
```
@article{jiang2025gsworld,
title={GSWorld: Closed-Loop Photo-Realistic Simulation Suite for Robotic Manipulation},
author={Jiang, Guangqi and Chang, Haoran and Qiu, Ri-Zhao and Liang, Yutong and Ji, Mazeyu and Zhu, Jiyue and Dong, Zhao and Zou, Xueyan and Wang, Xiaolong},
journal={arXiv preprint arXiv:2510.20813},
year={2025}
}
```
