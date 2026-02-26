# Real2Sim Pipeline Guide

Complete guide for creating simulated robotic environments from real-world scans using Gaussian Splatting.

---


## Detailed Steps

> **Note:** All robot-related files are stored in `assets/{robot_uid}_assets/` directory.  
> Steps 2-4 automatically look for files in this location based on the `--robot-uid` argument.

### Step 1: 3D Reconstruction

We follow the standard [3D Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting) or [2D Gaussian Splatting](https://github.com/hbb1/2d-gaussian-splatting) repository.

---

### Step 2: Generate URDF Point Cloud

```bash
cd real2sim/scripts
python uniform_pcd_from_urdf_visual_mesh.py \
    --robot-uid galaxea_r1 \  # Registered robot in gsworld/mani_skill/agents/robots/
    --filename r1 \            # Base name for output files
    --save-pcd
```

The `--robot-uid` must match a registered robot in `gsworld/mani_skill/agents/robots/`. The script automatically finds the URDF and creates output in `assets/{robot_uid}_assets/`.

**Output:** `assets/{robot_uid}_assets/`
- `{filename}.ply` - Point cloud (300k points)
- `{filename}_semantics.npy` - Link labels per point
- `{filename}_visual_mesh.obj` - Combined mesh  

---

### Step 3: Align URDF to Real Robot

**3a.(optional)** Crop robot in [Supersplat](https://supersplat.io/) → export as `cropped_arm.ply`  
       Place the file in `assets/{robot_uid}_assets/` (e.g., `assets/galaxea_r1_assets/cropped_arm.ply`)

**3b.** Compute alignment:
```bash
python open3d_alignment.py \
    --robot-uid galaxea_r1 \  # Looks in assets/galaxea_r1_assets/
    -t cropped_arm.ply         # Filename of cropped robot
# IMPORTANT: Copy the printed transformation matrix → save to gsworld/constants.py as sim2gs_{robot}_trans
```

**Output:** 4x4 transformation matrix (printed to terminal)  

---

### Step 4: Transfer Semantic Labels

```bash
python segment_real_gs.py \
    --robot-uid galaxea_r1 \        # Looks in assets/galaxea_r1_assets/
    --target-name 0425_r1.ply \     # Filename of real GS scan
    --transform-matrix-name sim2gs_r1_trans  # Transformation matrix name from Step 3
    # --bbox-threshold 0.04         # (Optional) Distance threshold (default: 0.04 for R1/XArm6, 0.025 for FR3)
```

**Output:** `assets/{robot_uid}_assets/{target_name}_semantics_gs.npy`

---

### Step 5: Create Scene Config

Create JSON file in `assets/` directory:

```json
{
    "models": [
        {
            "data_path": "./galaxea_r1_assets/0425_r1.ply",
            "semantic_labels": "./galaxea_r1_assets/0425_r1_semantics_gs.npy",
            "transformation": []
        }
    ]
}
```

**Fields:** `data_path` (GS .ply), `semantic_labels` (.npy or integer), `transformation` (optional 4x4)

---

### Step 6: Write Environment

Update `gsworld/constants.py` with semantic label mappings, then write ManiSkill environment class.

---

