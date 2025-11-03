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
```

### Motion Plannings
```bash
cd gsworld/mani_skill/examples/motionplanning
# xarm6
python xarm6/run_with_gs.py -e AlignXArmEnv-v1 -n 1 --vis --scene_cfg_name xarm6_align
python xarm6/run_with_gs.py -e BananaRotationXArmEnv-v1 -n 1 --vis --scene_cfg_name xarm6_rot_banana
python xarm6/run_with_gs.py -e SpoonOnBoardXArmEnv-v1 -n 1 --vis --scene_cfg_name xarm6_spoon2board
```

## Build Your Own Scene with GSWorld
WIP

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