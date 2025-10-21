# This demo is to do camera real2sim
# We use the realworld calibration cam2base

import os
import time
import gymnasium as gym
import numpy as np
from mani_skill.utils import common
import sapien

from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.utils import gym_utils
from mani_skill.utils.wrappers import RecordEpisode
from gsworld.mani_skill.utils.wrappers import GSWorldWrapper
from mani_skill.utils.visualization.misc import images_to_video
from gsworld.utils.io_utils import save_image_frames

from argparse import ArgumentParser
from gsworld.utils.gs_utils import ModelParams, PipelineParams, get_combined_args
# from gsworld.constants import sim2gs_arm_trans, fr3_umi_task_init_qpos

def main(args: ArgumentParser, scene_cfg_name, pipeline): # TODO scene_cfg_name is not used
    np.set_printoptions(suppress=True, precision=3)
    verbose = not args.quiet
    if isinstance(args.seed, int):
        args.seed = [args.seed]
    if args.seed is not None:
        np.random.seed(args.seed[0])
    parallel_in_single_scene = args.render_mode == "human"
    if args.render_mode == "human" and args.obs_mode in ["sensor_data", "rgb", "rgbd", "depth", "point_cloud"]:
        print("Disabling parallel single scene/GUI render as observation mode is a visual one. Change observation mode to state or state_dict to see a parallel env render")
        parallel_in_single_scene = False
    if args.render_mode == "human" and args.num_envs == 1:
        parallel_in_single_scene = False

    # cam_configs = dict(
    #     wrist_cam=dict(width=640, height=480),
    #     right_cam=dict(width=640, height=480),
    #     )
    sim_config = dict()
    if args.control_freq:
        sim_config["control_freq"] = args.control_freq
    if args.sim_freq:
        sim_config["sim_freq"] = args.sim_freq
    env_kwargs = dict(
        obs_mode=args.obs_mode,
        reward_mode=None,
        control_mode=args.control_mode,
        render_mode=args.render_mode,
        # sensor_configs=cam_configs,
        human_render_camera_configs=dict(shader_pack=args.shader),
        viewer_camera_configs=dict(shader_pack=args.shader),
        num_envs=args.num_envs,
        sim_backend=args.sim_backend,
        enable_shadow=True,
        parallel_in_single_scene=parallel_in_single_scene,
        max_episode_steps=args.ep_len, # TODO if render mode is human, this does not work, also record will get error
        sim_config=sim_config,
    )
    if args.robot_uids is not None:
        env_kwargs["robot_uids"] = tuple(args.robot_uids.split(","))
    env: BaseEnv = gym.make(
        args.env_id,
        **env_kwargs
    )

    env = GSWorldWrapper(
        env=env,
        scene_gs_cfg_name=scene_cfg_name,
        robot_pipe=pipeline.extract(args), # TODO this is needed for render
        device="cuda",
        log_state = True,
        state_log_path="./exp_log/tttest"
    )

    record_dir = args.record_dir
    if record_dir:
        record_dir = record_dir.format(env_id=args.env_id)
        env = RecordEpisode(env, record_dir, info_on_video=False, save_trajectory=False, 
                            max_steps_per_video=gym_utils.find_max_episode_steps_value(env),
                            video_fps=10)

    if verbose:
        print("Observation space", env.observation_space)
        print("Action space", env.action_space)
        if env.unwrapped.agent is not None:
            print("Control mode", env.unwrapped.control_mode)
        print("Reward mode", env.unwrapped.reward_mode)

    obs, _ = env.reset(seed=args.seed, options=dict(reconfigure=True))
    # print(f"Pre-defined task init qpos {fr3_umi_task_init_qpos}")
    # print(f"Arm tcp pose {env.unwrapped.agent.tcp.pose.raw_pose}")
    # print(f"Arm qpos {env.unwrapped.agent.robot.get_qpos()}")
    gsworld_imgs_all = {cam_name: [] for cam_name in obs["sensor_data"].keys()}
    for cam_name, sensor in obs["sensor_data"].items():
        img = common.unbatch(common.to_numpy(sensor["rgb"]))
        gsworld_imgs_all[cam_name].append(img)

    if args.seed is not None and env.action_space is not None:
            env.action_space.seed(args.seed[0])
    if args.render_mode is not None:
        viewer = env.render() # TODO or we can first modify the render function?
        if isinstance(viewer, sapien.utils.Viewer):
            viewer.paused = args.pause
        env.render()
        
    prev_time = time.time()
    frame_count = 0
    while True:
        action = env.action_space.sample() if env.action_space is not None else None
        # action = np.zeros_like(env.action_space.sample().shape)
        print(action)
        obs, reward, terminated, truncated, info = env.step(action)
        for cam_name, sensor in obs["sensor_data"].items():
            img = common.unbatch(common.to_numpy(sensor["rgb"]))
            gsworld_imgs_all[cam_name].append(img)

        frame_count += 1
        if verbose:
            print("reward", reward)
            print("terminated", terminated)
            print("truncated", truncated)
            print("info", info)
        if args.render_mode is not None:
            env.render()
        if args.render_mode is None or args.render_mode != "human":
            if (terminated | truncated).any():
                break
    
    current_time = time.time()
    elapsed_time = current_time - prev_time
    fps = frame_count / elapsed_time
    print(f"FPS: {fps:.2f}")
    # env.save_state_log()
    env.close()

    if record_dir:
        for cam_name, imgs in gsworld_imgs_all.items():
            # Save frames
            # save_image_frames(imgs, os.path.join(record_dir, f"gsworld_{cam_name}"), verbose=False)
            # Save video
            images_to_video(imgs, record_dir, f"gsworld_{cam_name}", fps=10)
        print(f"Saving video to {record_dir}")


if __name__ == "__main__":
    # parser = tyro.cli(Args)

    parser = ArgumentParser(description="GS World")
    # model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser) # TODO this is needed for render
    # parser.add_argument("--iteration", default=-1, type=int)
    # parser.add_argument("--skip_train", action="store_true")
    # parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    # gaussian model
    parser.add_argument("--scene_cfg_name", default="xarm_table_test", type=str, help="Path to scene json config file")

    # maniskill
    parser.add_argument("--env_id", default="PickALignYCBXArmEnv-v1", type=str, help="The environment ID of the task you want to simulate")
    parser.add_argument("--obs_mode", default="rgb+segmentation", type=str, help="Observation mode")
    parser.add_argument("--robot_uids", default="xarm6_uf_gripper", type=str, help="Robot UID(s) to use. Can be a comma-separated list of UIDs or an empty string to have no agents. If not given, it defaults to the environment's default robot")
    parser.add_argument("--sim_backend", default="auto", type=str, help="Which simulation backend to use. Can be 'auto', 'cpu', 'gpu'")
    parser.add_argument("--control_mode", default="pd_joint_pos", type=str, help="Control mode")
    parser.add_argument("--render_mode", default="sensors", type=str, help="Render mode") # TODO human render mode will get an error in the eps end
    parser.add_argument("--record_dir", default="./exp_log/pour", type=str, help="Directory to save recordings")
    parser.add_argument("--pause", default=False, action="store_true", help="If using human render mode, auto pauses the simulation upon loading")
    parser.add_argument("--quiet_ms", default=False, action="store_true", help="Disable verbose output.")
    parser.add_argument("--seed", default=0, type=int, nargs="+", help="Seed(s) for random actions and simulator. Can be a single integer or a list of integers. Default is None (no seeds)")
    parser.add_argument("--reward_mode", default=None, type=str, help="Reward mode")
    parser.add_argument("--shader", default="default", type=str, help="Change shader used for all cameras in the environment for rendering. Default is 'minimal' which is very fast. Can also be 'rt' for ray tracing and generating photo-realistic renders. Can also be 'rt-fast' for a faster but lower quality ray-traced renderer")
    parser.add_argument("--num_envs", default=1, type=int, help="Number of environments to run.")
    parser.add_argument("--ep_len", default=30, type=int, help="Number of max episode length.") # TODO fix bug when ep len is long
    # define control freq, align with real
    parser.add_argument("--sim_freq", default=120, type=int, help="Simulator physics freq.")
    parser.add_argument("--control_freq", default=40, type=int, help=".step control freq.")

    # args = get_combined_args(parser)
    args = parser.parse_args()
    print("Building GS World with " + args.scene_cfg_name)

    main(args, scene_cfg_name=args.scene_cfg_name, pipeline=pipeline)
