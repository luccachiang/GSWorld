import os
import numpy as np
from pathlib import Path

UFGRIPPER_CLOSED_THRESHOLD = 0.1

FILE_DIR = os.path.dirname(os.path.abspath(__file__)) # gsworld
ASSET_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../assets") # assets 
DTC_DIR = Path(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../assets/dtc_assets")) # dtc
GS_DIR = Path(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../submodules/gaussian-splatting")) # gaussian repo
PROJ_DIR = os.path.join(FILE_DIR, "..") # gsworld
CFG_DIR = os.path.join(PROJ_DIR, "configs") # configs

x_180_deg_rot = np.array([
    [1, 0, 0],
    [0, -1, 0],
    [0, 0, -1]
])
y_180_deg_rot = np.array([
    [-1, 0, 0],
    [0, 1, 0],
    [0, 0, -1]
])
z_180_deg_rot = np.array([
    [-1, 0, 0],
    [0, -1, 0],
    [0, 0, 1]
])

sim2gs_arm_trans = matrix = np.array([
    [ 0.65203872,  0.70075277,  0.03073432, -0.08619287],
    [ 0.03194594,  0.01225097, -0.95706996, -0.75944751],
    [-0.70069858,  0.65264769, -0.01503433,  0.25320947],
    [ 0.        ,  0.        ,  0.        ,  1.        ]
], dtype=np.float32)

sim2gs_xarm_trans = np.array([
    [-0.97002696,  0.2247966 ,  0.10835464,  0.32787871],
    [ 0.05080531,  0.60369423, -0.7976206 ,  0.37823396],
    [-0.24432164, -0.76697216, -0.59605971,  0.45637834],
    [ 0.        ,  0.        ,  0.        ,  1.        ]
], dtype=np.float32)

sim2gs_r1_trans = np.array([
    [ 2.70573310e-01, -8.82001081e-01,  3.56843272e-03, -1.01723192e-02],
    [-2.38923961e-04, -3.80586011e-03, -9.22569247e-01, -6.73930139e-02],
    [ 8.82008267e-01,  2.70570074e-01, -1.34459800e-03, -1.45273889e-02],
    [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00]
], dtype=np.float32)

fr3_umi_gs_init_qpos = np.array([
        0.00879998,
        -0.62698067,
        -0.00591884,
        -2.25830053,
        -0.00557862,
        1.63165594,
        0.78539816,
        4.04319502e-02, 
        4.04319502e-02,
    ], dtype=np.float32)

fr3_umi_task_init_qpos = np.array([
        0.00879998,
        -0.62698067,
        -0.00591884,
        -2.25830053,
        -0.00557862,
        1.63165594,
        0.78539816,
        4.04319502e-02, 
        4.04319502e-02,
    ], dtype=np.float32)

xarm_gs_qpos = np.array([
        0.0,  # joint1
        0.0,  # joint2 
        -np.pi / 4,  # joint3
        0.0,  # joint4
        np.pi / 4,  # joint5
        0.0,  # joint6
        0.0,  # drive_joint (gripper)
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    ], dtype=np.float32)

xarm_task_init_qpos = np.array([
        0.0,  # joint1
        0.0,  # joint2 
        -np.pi / 3,  # joint3
        0.0,  # joint4
        np.pi / 3,  # joint5
        0.0,  # joint6
        0.0,  # drive_joint (gripper)
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    ], dtype=np.float32)

r1_task_init_qpos = np.array([
        0.0, # servo_joint1
        0.0, # servo_joint2
        0.0, # servo_joint3
        0.0, # torso_joint1
        0.0, # wheel_joint1
        0.0, # wheel_joint2
        0.0, # wheel_joint3
        0.0, # torso_joint2
        0.0, # torso_joint3
        0.0, # torso_joint4
        np.pi / 2 + 0.05, # left_arm_joint1
        -np.pi / 2 - 0.05, # right_arm_joint1 # reversed from left joint1
        np.pi * 3 / 4 + 0.1, # left_arm_joint2
        np.pi * 3 / 4, # right_arm_joint2
        -np.pi * 2 / 3 + 0.01, # left_arm_joint3
        -np.pi * 2 / 3 + 0.05, # right_arm_joint3
        0.0, # left_arm_joint4
        0.0, # right_arm_joint4
        0.0, # left_arm_joint5
        0.0, # right_arm_joint5
        0.0, # left_arm_joint6
        0.0, # right_arm_joint6
        0.02, # left_gripper_axis1
        0.02, # left_gripper_axis2:
        0.02, # right_gripper_axis1
        0.02, # right_gripper_axis2:
    ], dtype=np.float32)

r1_arm_heart_qpos = np.array([
        0.0, # servo_joint1
        0.0, # servo_joint2
        0.0, # servo_joint3
        0.0, # torso_joint1
        0.0, # wheel_joint1
        0.0, # wheel_joint2
        0.0, # wheel_joint3
        0.0, # torso_joint2
        0.0, # torso_joint3
        0.0, # torso_joint4
        -1.5703, # left_arm_joint1
        1.5703, # right_arm_joint1 # reversed from left joint1
        1.98, # left_arm_joint2
        1.98, # right_arm_joint2
        -1.21, # left_arm_joint3
        -1.21, # right_arm_joint3
        -1.5703, # left_arm_joint4
        -1.5703, # right_arm_joint4
        0.95, # left_arm_joint5
        0.95, # right_arm_joint5
        0.13, # left_arm_joint6
        0.13, # right_arm_joint6
        0.02, # left_gripper_axis1
        0.02, # left_gripper_axis2:
        0.02, # right_gripper_axis1
        0.02, # right_gripper_axis2:
    ], dtype=np.float32)

robot_scan_qpos = {
    "fr3_umi_wrist435_cam_mount": fr3_umi_gs_init_qpos,
    "fr3_umi_wrist435": fr3_umi_gs_init_qpos,
    "fr3_umi": fr3_umi_gs_init_qpos,
    "xarm6_uf_gripper": xarm_task_init_qpos,
    "xarm6_uf_gripper_wrist435": xarm_gs_qpos,
    "galaxea_r1": r1_task_init_qpos,
}

robot_task_init_qpos = {
    "fr3_umi_wrist435_cam_mount": fr3_umi_task_init_qpos,
    "fr3_umi_wrist435": fr3_umi_task_init_qpos,
    "fr3_umi": fr3_umi_task_init_qpos,
    "xarm6_uf_gripper": xarm_task_init_qpos,
    "xarm6_uf_gripper_wrist435": xarm_task_init_qpos,
    "galaxea_r1": r1_task_init_qpos,
}

sim2gs_mustard_trans = np.array([
    [0.510841, -0.618467, -0.0283021, 0.000536782],
    [-0.0116977, 0.0270441, -0.802118, -0.299513],
    [0.619003, 0.51091, 0.00819852, -0.0257972],
    [0, 0, 0, 1]
], dtype=np.float32)

sim2gs_snack_box_trans = np.array([
        [ 0.167587 , -0.728182 , -0.664579 ,  0.0772496],
        [-0.961955 , -0.268324 ,  0.051428 , -0.137463 ],
        [-0.215772 ,  0.630676 , -0.745446 ,  0.262004 ],
        [ 0.       ,  0.       ,  0.       ,  1.       ] 
], dtype=np.float32)

sim2gs_bread_slice_trans = np.array([
        [3.98912, 10.5397, 0.649514, 0.345585],
        [-7.51925, 2.34766, 8.08519, 0.941237],
        [7.41409, -3.28989, 7.8504, 1.70994],
        [ 0.       ,  0.       ,  0.       ,  1.       ] 
], dtype=np.float32)
sim2gs_bread_slice_trans[:3, :3] = sim2gs_bread_slice_trans[:3, :3] @ x_180_deg_rot

# =============================================
sim2gs_spice_rack_trans = np.array([
    [0.18858,   -0.91295,   0.000928738,  0.124793],
    [0.0308756,  0.00542988, -0.931696,   -0.191401],
    [0.912428,   0.188504,   0.0313357,    0.0207375],
    [0.0,        0.0,        0.0,          1.0]
], dtype=np.float32)

sim2gs_tomato_soup_can_trans = np.array([
    [-0.000393405, -0.892165,  -0.000564396, -0.0584909],
    [ 0.00982369,  0.000560062, -0.892111,  -0.294318],
    [ 0.892111,  -0.0003996,  0.00982345, -0.045191],
    [ 0.0,     0.0,     0.0,     1.0]
], dtype=np.float32)

sim2gs_baseball_trans = np.array([
    [ 0.85493,  -0.426098,  0.127043,  -0.0459754],
    [ 0.132547,  0.50705,   0.808663,  -0.298154],
    [-0.424418, -0.699961,  0.508457,  -0.0455954],
    [ 0.0,       0.0,       0.0,        1.0]
], dtype=np.float32)
sim2gs_baseball_trans[:3, :3] = sim2gs_baseball_trans[:3, :3] @ x_180_deg_rot

sim2gs_gelatin_box_trans = np.array([
    [ 0.431306,  -0.0535783,  0.761947,  -0.00255462],
    [-0.00114077, -0.875071, -0.0608872, -0.110748],
    [ 0.763827,   0.0289468, -0.430335,   0.00802863],
    [ 0.0,        0.0,        0.0,         1.0]
], dtype=np.float32)

sim2gs_lemon_trans = np.array([
    [ 0.636018,  0.431956, -0.198637, -0.225826],
    [-0.453441,  0.650861, -0.0365169, -0.445538],
    [ 0.142947,  0.142675,  0.767966, -0.191521],
    [ 0.0,       0.0,       0.0,        1.0]
], dtype=np.float32)

sim2gs_banana_trans = np.array([
    [  4.29100891, -15.59620731,   7.29994834,   0.18308148],
    [ -9.23809578,   4.26401532,  14.54026979,   1.12816099],
    [-14.53232654,  -7.31574853,  -7.08766496,   3.25415401],
    [  0.,           0.,           0.,           1.        ]
], dtype=np.float32)

sim2gs_cleanser_trans = np.array([
    [ 7.65158271e-01, -2.36258082e-01,  2.10928566e-02,  2.41788604e-03],
    [ 1.43967208e-02, -2.48475971e-02, -8.00565613e-01, -3.42199773e-01],
    [ 2.36760479e-01,  7.65045544e-01, -1.94874332e-02, -6.91950050e-04],
    [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00]
], dtype=np.float32)

sim2gs_tennis_ball_trans = np.array([
    [ 0.840371,  0.257109,  0.114891, -0.0154867],
    [ 0.241193, -0.470482, -0.711338, -0.274127],
    [-0.145365,  0.705741, -0.516069, -0.080588],
    [ 0.0,       0.0,       0.0,       1.0]
], dtype=np.float32)


def euler2mat(x, y, z):
    Rx = np.array([
    [1, 0, 0],
    [0, np.cos(x), -np.sin(x)],
    [0, np.sin(x), np.cos(x)]
    ])
    Ry = np.array([
    [np.cos(y), 0, np.sin(y)],
    [0, 1, 0],
    [-np.sin(y), 0, np.cos(y)]
    ])
    Rz = np.array([
    [np.cos(z), -np.sin(z), 0],
    [np.sin(z), np.cos(z), 0],
    [0, 0, 1]
    ])
    return Rz @ Ry @ Rx
cylinder_fix = np.eye(4)
cylinder_fix[:3, :3] = euler2mat(0, -np.pi/2, 0)

# dtc collision
sim2gs_dtc_green_can = np.array([
    [-7.33941866e-01,  3.36139655e-02, -5.37589582e-01,  4.31403401e-02],
    [-2.65421561e-02, -9.09764955e-01, -2.06484809e-02, -2.37379279e-01],
    [-5.37985103e-01, -9.73211127e-04,  7.34420997e-01, -1.01842246e-01],
    [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00]
], dtype=np.float32)

sim2gs_dtc_spoon = np.array([
    [-0.56167979, -0.01032282, -0.0277326,   0.015921  ],
    [ 0.02844451, -0.33368127, -0.4518933,   0.1459936 ],
    [-0.00815888, -0.45266995,  0.33374119,  0.12412568],
    [ 0.0,         0.0,         0.0,         1.0       ]
], dtype=np.float32)

sim2gs_dtc_green_cutting_board = np.array([
    [-6.45884193, -0.79723217, 10.77221029,  0.28603743],
    [10.1708243,   3.77760417,  6.37783447,  1.29758983],
    [-3.63736535, 11.9786001,  -1.29439011,  1.91922799],
    [0.0,          0.0,         0.0,         1.0]
], dtype=np.float32)

sim2gs_dtc_red_tomato_can = np.array([
    [ 0.893085,   -0.00686968,  0.0312505,   0.0118606],
    [ 0.0312641,   0.00166396, -0.893109,   -0.26732],
    [ 0.00680727,  0.89363,     0.00190322, -0.0275778],
    [ 0,           0,           0,           1]
], dtype=np.float32)
sim2gs_dtc_red_tomato_can = sim2gs_dtc_red_tomato_can @ cylinder_fix
# =============================================

# TODO how to link with ycb model id
sim2gs_object_transforms = {
    "005_tomato_soup_can": sim2gs_tomato_soup_can_trans,
    "005_tomato_soup_can-0": sim2gs_tomato_soup_can_trans,
    "005_tomato_soup_can-1": sim2gs_tomato_soup_can_trans,
    "006_mustard_bottle": sim2gs_mustard_trans,
    "006_mustard_bottle-0": sim2gs_mustard_trans,
    "006_mustard_bottle-1": sim2gs_mustard_trans,
    "snack_box": sim2gs_snack_box_trans,
    "spice_rack": sim2gs_spice_rack_trans,
    "055_baseball": sim2gs_baseball_trans,
    "009_gelatin_box": sim2gs_gelatin_box_trans,
    "011_banana": sim2gs_banana_trans,
    "014_lemon": sim2gs_lemon_trans,
    "021_bleach_cleanser": sim2gs_cleanser_trans,
    "056_tennis_ball": sim2gs_tennis_ball_trans,
    "dtc_green_can": sim2gs_dtc_green_can,
    "dtc_red_tomato_can": sim2gs_dtc_red_tomato_can,
    "bread_slice": sim2gs_bread_slice_trans,
    "dtc:Kitchen_Spoon_B008H2JLP8_LargeWooden": sim2gs_dtc_spoon,
    "dtc:Cutting_Board_B005CZ90HM_LimeGreen": sim2gs_dtc_green_cutting_board,
}

# fine-tuning gs alignment
object_offset = {
    # "006_mustard_bottle": [-0.02, 0.01, 0.02],
    "005_tomato_soup_can": [0.0, 0.0, 0.01],
    "005_tomato_soup_can-0": [0.0, 0.0, 0.04], # stack goal
    "005_tomato_soup_can-1": [0.0, 0.0, 0.01],
    "006_mustard_bottle": [0.0, 0.0, 0.02],
    "006_mustard_bottle-0": [0.0, 0.0, 0.02],
    "006_mustard_bottle-1": [0.0, 0.0, 0.02],
    "white_box": [0.0, 0.0, 0.02],
    "plate": [0.0, 0.0, 0.02],
    "snack_box": [0.0, 0.0, 0.035],
    "spice_rack": [0.0, 0.0, 0.045],
    "055_baseball": [0.0, 0.0, 0.02],
    "009_gelatin_box": [0.0, 0.0, 0.035],
    "011_banana": [0.0, 0.0, -0.03],
    "014_lemon": [0.0, 0.0, 0.02],
    "021_bleach_cleanser": [0.0, 0.0, 0.02],
    "056_tennis_ball": [0.0, 0.0, 0.02],
    "xarm_arm": [0.0, 0.0, 0.05],
    "dtc_green_can": [-0.04, 0.0, -0.03],
    "dtc_red_tomato_can": [0.0, -0.015, 0.04],
    "bread_slice": [0.0, 0.0, 0.04],
    "dtc:Kitchen_Spoon_B008H2JLP8_LargeWooden": [0.0, 0.0, 0.0],
}
object_scale = {
    "005_tomato_soup_can": 1,
    "005_tomato_soup_can-0": 1,
    "005_tomato_soup_can-1": 1,
    "006_mustard_bottle": 1,
    "006_mustard_bottle-0": 1,
    "006_mustard_bottle-1": 1,
    "white_box": 1,
    "plate": 1,
    "snack_box": 1,
    "spice_rack": 1,
    "055_baseball": 1,
    "009_gelatin_box": 1,
    "011_banana": 1,
    "014_lemon": 1,
    "021_bleach_cleanser": 1,
    "056_tennis_ball": 1,
    "dtc_green_can": 1,
    "dtc_red_tomato_can": 1,
    "bread_slice": 0.95,
    "r1table": 1,
    "dtc:Kitchen_Spoon_B008H2JLP8_LargeWooden": 1,
    "dtc:Cutting_Board_B005CZ90HM_LimeGreen": 1,
}


### gaussian model semantic idx label, -1 is bg
fr3_gs_semantics = {
    "base": 0,
    "fr3_link0": 1,
    "fr3_link1": 2,
    "fr3_link2": 3,
    "fr3_link3": 4,
    "fr3_link4": 5,
    "fr3_link5": 6,
    "fr3_link6": 7,
    "fr3_link7": 8,
    "fr3_link8": 9,
    "fr3_hand": [10, 14, 15],
    "fr3_hand_tcp": 11,
    "fr3_leftfinger": 12,
    "fr3_rightfinger": 13,
    # "camera_base_link": 14,
    # "camera_link": 15,
}

obj_gs_semantics = {
    "006_mustard_bottle": 100,
    "006_mustard_bottle-0": 100,
    "006_mustard_bottle-1": 1001,
    "white_box": 101,
    "plate": 102,
    "snack_box": 103,
    "055_baseball": 104,
    "009_gelatin_box": 105,
    "011_banana": 114,
    "014_lemon": 106,
    "021_bleach_cleanser": 107,
    "056_tennis_ball": 108,
    "spice_rack": 109,
    "005_tomato_soup_can": 110,
    "005_tomato_soup_can-0": 110,
    "005_tomato_soup_can-1": 1010,
    "dtc_green_can": 201,
    "dtc_red_tomato_can": 202,
    "dtc:Cutting_Board_B005CZ90HM_LimeGreen": 203,
    "bread_slice": 111,
    "r1table": 112,
    "dtc:Kitchen_Spoon_B008H2JLP8_LargeWooden": 204,
}


#################################
xarm_gs_semantics = {
    "world": 0,
    "link_base": 1,
    "link1": 2,
    "link2": 3,
    "link3": 4,
    "link4": 5,
    "link5": 6,
    "link6": [7, 8], # for urdf xarm6 without camera, we need to move the camera
    # "link_eef": 8, # xarm6 with camera
    "xarm_gripper_base_link": 9,
    "left_outer_knuckle": 10,
    "left_inner_knuckle": 11,
    "right_outer_knuckle": 12,
    "right_inner_knuckle": 13,
    "xarm_hand_tcp": 14,
    "left_finger": 15,
    "right_finger": 16,
}

r1_gs_semantics = {
    "base_link": 0,
    "servo_link1": 1,
    "servo_link2": 2,
    "servo_link3": 3,
    "torso_link1": 4,
    "wheel_link1": 5,
    "wheel_link2": 6,
    "wheel_link3": 7,
    "torso_link2": 8,
    "torso_link3": 9,
    "torso_link4": 10,
    "zed_link": 11,
    "left_arm_link1": 12,
    "right_arm_link1": 13,
    "left_arm_link2": 14,
    "right_arm_link2": 15,
    "left_arm_link3": 16,
    "right_arm_link3": 17,
    "left_arm_link4": 18,
    "right_arm_link4": 19,
    "left_arm_link5": 20,
    "right_arm_link5": 21,
    "left_arm_link6": 22,
    "right_arm_link6": 23,
    "left_realsense_link": 24,
    "left_gripper_tcp": 25,
    "left_gripper_link1": 26,
    "left_gripper_link2": 27,
    "l_hand_keypoint": 28,
    "right_realsense_link": 29,
    "right_gripper_tcp": 30,
    "right_gripper_link1": 31,
    "right_gripper_link2": 32,
    "r_hand_keypoint": 33,
}

wrist2eef = np.array([
    [ 0.00561756, -0.99991452,  0.01180684,  0.0691971],
    [ 0.99993738,  0.00573118,  0.00961197, 0.02580245],
    [-0.00967881,  0.01175211,  0.9998841,  -0.1056441],
    [ 0.        ,  0.        ,  0.        ,  1.        ]
], dtype=np.float32)

rs_d435i_rgb_k = np.array([
    [606.12145996, 0.0, 318.3548584],
    [0.0, 605.1428833, 242.92498779],
    [0.0, 0.0, 1.0]
], dtype=np.float32)

right2base = np.array([
[-0.025185470710454363, 0.9003537485256276, -0.43442930331751733, 0.8003658631290567],
[0.9990845637502204, 0.007637667199582072, -0.04209157297821219, 0.014761293894194942],
[-0.034579279071787865, -0.4350917070636938, -0.8997218903101533, 0.8497237283025128],
[0.0, 0.0, 0.0, 1.0],
], dtype=np.float32)

xarm_right2base = np.array([
    [-0.99815940,  0.02312000,  0.05609515,  0.38209513],
    [-0.00610404,  0.88159275, -0.47197380,  0.40018010],
    [-0.06036488, -0.47144645, -0.87982790,  0.46095666],
    [ 0.00000000,  0.00000000,  0.00000000,  1.00000000],
], dtype=np.float32)

xarm_wrist2base = np.array([
    [-0.0375638,  -0.9982628,  -0.04539683,  0.01998455],
    [ 0.99928665, -0.03734544, -0.00564907, -0.00621691],
    [ 0.00394388, -0.04557664,  0.99895304,  -0.0705968],
    [ 0.0,         0.0,         0.0,         1.0]
], dtype=np.float32)