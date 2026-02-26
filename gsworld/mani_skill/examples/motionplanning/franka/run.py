# gq: code adapted from panda motion planning
import multiprocessing as mp
import os
from copy import deepcopy
import time
import argparse
import gymnasium as gym
import numpy as np
from tqdm import tqdm
import os.path as osp
from mani_skill.utils.wrappers.record import RecordEpisode
from mani_skill.trajectory.merge_trajectory import merge_trajectories
from gsworld.mani_skill.examples.motionplanning.fr3_umi.solutions import *
from gsworld.constants import *
from gsworld.utils.gs_utils import PipelineParams, get_combined_args
from gsworld.utils.io_utils import save_images_to_mp4

MP_SOLUTIONS = {
    "PickCube-v1": solvePickCube,
    "StackCube-v1": solveStackCube,
    "PegInsertionSide-v1": solvePegInsertionSide,
    "PlugCharger-v1": solvePlugCharger,
    "PushCube-v1": solvePushCube,
    "PullCubeTool-v1": solvePullCubeTool,
    "LiftPegUpright-v1": solveLiftPegUpright,
    "PullCube-v1": solvePullCube,
    "PnpSingleYCBBox-v1": solvePnpYcbBox,
    "PourMustard-v1": solvePourMustard,
    "PnpAlignYCB-v1": solvePnpAlign,
    "PnpStackYCB-v1": solvePnpStack,
}

def parse_args(args=None):
    parser = argparse.ArgumentParser(description="GS World")
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")

    # maniskill
    parser.add_argument("--obs_mode", default="rgb+segmentation", type=str, help="Observation mode")
    parser.add_argument("--sim_backend", default="auto", type=str, help="Which simulation backend to use. Can be 'auto', 'cpu', 'gpu'")
    parser.add_argument("--control_mode", default="pd_joint_pos", type=str, help="Control mode")
    parser.add_argument("--render_mode", default="sensors", type=str, help="Render mode")
    parser.add_argument("--pause", default=False, action="store_true", help="If using human render mode, auto pauses the simulation upon loading")
    parser.add_argument("--quiet_ms", default=False, action="store_true", help="Disable verbose output.")
    parser.add_argument("--seed", default=0, type=int, nargs="+", help="Seed(s) for random actions and simulator. Can be a single integer or a list of integers. Default is None (no seeds)")
    parser.add_argument("--reward_mode", default=None, type=str, help="Reward mode")
    parser.add_argument("--num_envs", default=1, type=int, help="Number of environments to run.")
    parser.add_argument("--ep_len", default=800, type=int, help="Number of max episode length.")
    parser.add_argument("-e", "--env-id", type=str, default="PickCube-v1", help=f"Environment to run motion planning solver on. Available options are {list(MP_SOLUTIONS.keys())}")
    parser.add_argument("-n", "--num-traj", type=int, default=10, help="Number of trajectories to generate.")
    parser.add_argument("-r", "--robot-uid", type=str, default="fr3_umi_wrist435", help="Robot uid.")
    parser.add_argument("--only-count-success", action="store_true", help="If true, generates trajectories until num_traj of them are successful and only saves the successful trajectories/videos")
    parser.add_argument("-b", "--sim-backend", type=str, default="auto", help="Which simulation backend to use. Can be 'auto', 'cpu', 'gpu'")
    parser.add_argument("--render-mode", type=str, default="rgb_array", help="can be 'sensors' or 'rgb_array' which only affect what is saved to videos")
    parser.add_argument("--vis", action="store_true", help="whether or not to open a GUI to visualize the solution live")
    parser.add_argument("--save-video", action="store_true", help="whether or not to save videos locally")
    parser.add_argument("--traj-name", type=str, help="The name of the trajectory .h5 file that will be created.")
    parser.add_argument("--shader", default="default", type=str, help="Change shader used for rendering. Default is 'default' which is very fast. Can also be 'rt' for ray tracing and generating photo-realistic renders. Can also be 'rt-fast' for a faster but lower quality ray-traced renderer")
    parser.add_argument("--record-dir", type=str, default="./demos", help="where to save the recorded trajectories")
    parser.add_argument("--num_procs", type=int, default=1, help="Number of processes to use to help parallelize the trajectory replay process. This uses CPU multiprocessing and only works with the CPU simulation backend at the moment.")

    parser.add_argument("--scene_cfg_name", default=None, type=str, help="Path to scene json config file")
    # define control freq, align with real
    parser.add_argument("--sim_freq", default=100, type=int, help="Simulator physics freq.")
    parser.add_argument("--control_freq", default=20, type=int, help=".step control freq.")
    # recovery state
    parser.add_argument("--recovery_state_logger_path", default=None, type=str, help="Path to directory to stores (multiple) state loggers for recovery. If not given, the env will randomly initialized.")

    return parser.parse_args(), pipeline

def _main(args, pipeline, proc_id: int = 0, start_seed: int = 0) -> str:
    # start_seed can be the state recovery idx
    env_id = args.env_id
    sim_config = dict()
    if args.control_freq:
        sim_config["control_freq"] = args.control_freq
    if args.sim_freq:
        sim_config["sim_freq"] = args.sim_freq
    env = gym.make(
        env_id,
        robot_uids=args.robot_uid, # gq, add support for other robots
        obs_mode=args.obs_mode,
        control_mode=args.control_mode,
        render_mode=args.render_mode,
        reward_mode=None, # "dense" if args.reward_mode is None else args.reward_mode, # TODO cause env to get wrong?
        # sensor_configs=cam_configs,
        human_render_camera_configs=dict(shader_pack=args.shader),
        viewer_camera_configs=dict(shader_pack=args.shader),
        sim_backend=args.sim_backend,
        parallel_in_single_scene=False,
        sim_config=sim_config,
        max_episode_steps=args.ep_len,
    )
    
    if env_id not in MP_SOLUTIONS:
        raise RuntimeError(f"No already written motion planning solutions for {env_id}. Available options are {list(MP_SOLUTIONS.keys())}")
    
    if not args.traj_name:
        new_traj_name = time.strftime("%Y%m%d_%H%M%S")
    else:
        new_traj_name = args.traj_name

    if args.num_procs > 1:
        new_traj_name = new_traj_name + "." + str(proc_id)

    env = RecordEpisode(
        env,
        output_dir=osp.join(args.record_dir, env_id, "motionplanning"),
        trajectory_name=new_traj_name, save_video=args.save_video,
        source_type="motionplanning",
        source_desc="official motion planning solution from ManiSkill contributors",
        video_fps=30,
        save_on_reset=False
    )

    restore_states = []
    if args.recovery_state_logger_path is not None:
        for state_logger_file_path in os.listdir(args.recovery_state_logger_path):
            if state_logger_file_path.endswith(".h5") or state_logger_file_path.endswith(".hdf5"):
                restore_states.append(state_logger_file_path)

    output_h5_path = env._h5_file.filename
    solve = MP_SOLUTIONS[env_id]
    print(f"Motion Planning Running on {env_id}")
    pbar = tqdm(range(args.num_traj), desc=f"proc_id: {proc_id}")
    seed = start_seed
    successes = []
    solution_episode_lengths = []
    failed_motion_plans = 0
    passed = 0
    while True: # main demo collection loop
        # !!! This try block hugely cause trouble in debug
        # try:
            # TODO add state recovery, call set_state_recovery_cfg in gswrapper, then reset
            # recover inside the solve function, since we need to decide which step to restore based on the env logged info
        if args.recovery_state_logger_path is None:
            res = solve(env, seed=seed, debug=False, vis=True if args.vis else False)
        else:
            # sample a state logger file
            state_file_idx = seed % len(restore_states)
            res = solve(env, seed=seed, debug=False, vis=True if args.vis else False, state_recover_file=os.path.join(args.recovery_state_logger_path, restore_states[state_file_idx]))
        # except Exception as e:
        #     print(f"Cannot find valid solution because of an error in motion planning solution: {e}")
        #     res = -1
        # print(f"{res=}")

        if res == -1:
            success = False
            failed_motion_plans += 1
        else:
            success = res[-1]["success"].item()
            elapsed_steps = res[-1]["elapsed_steps"].item()
            solution_episode_lengths.append(elapsed_steps)
        successes.append(success)
        if args.only_count_success and not success:
            seed += 1
            env.flush_trajectory(save=False)
            if args.save_video:
                env.flush_video(save=False)
            continue
        else:
            env.flush_trajectory()
            if args.save_video:
                env.flush_video()
            pbar.update(1)
            pbar.set_postfix(
                dict(
                    success_rate=np.mean(successes),
                    failed_motion_plan_rate=failed_motion_plans / (seed + 1),
                    avg_episode_length=np.mean(solution_episode_lengths),
                    max_episode_length=np.max(solution_episode_lengths),
                    # min_episode_length=np.min(solution_episode_lengths)
                )
            )
            seed += 1
            passed += 1
            if passed == args.num_traj:
                break
    env.close()
    return output_h5_path

def main(args, pipeline):
    # args = parser.parse_args()
    if args.num_procs > 1 and args.num_procs <= args.num_traj:
        if args.num_traj < args.num_procs: # TODO why subprocess doesnt support multiple processes?
            raise ValueError("Number of trajectories should be greater than or equal to number of processes")
        args.num_traj = args.num_traj // args.num_procs
        # seeds = [*range(0, args.num_procs * args.num_traj, args.num_traj)] # always start from 0, so we may use this as the state recovery idx
        # seeds = [*range(args.seed[0], args.seed[0] + args.num_procs * args.num_traj, args.num_traj)] # always start from 0, so we may use this as the state recovery idx
        
        if isinstance(args.seed, list):
            seeds = [*range(args.seed[0], args.seed[0] + args.num_procs * args.num_traj, args.num_traj)]
        else:
            seeds = [*range(args.seed, args.seed + args.num_procs * args.num_traj, args.num_traj)]
            
        # seeds = [*range(args.seed, args.seed + args.num_procs * args.num_traj, args.num_traj)] # always start from 0, so we may use this as the state recovery idx
        pool = mp.Pool(args.num_procs)
        proc_args = [(deepcopy(args), pipeline, i, seeds[i]) for i in range(args.num_procs)]
        res = pool.starmap(_main, proc_args)
        pool.close()
        # Merge trajectory files
        output_path = res[0][: -len("0.h5")] + "h5"
        merge_trajectories(output_path, res)
        for h5_path in res:
            tqdm.write(f"Remove {h5_path}")
            os.remove(h5_path)
            json_path = h5_path.replace(".h5", ".json")
            tqdm.write(f"Remove {json_path}")
            os.remove(json_path)
        # save_images_to_mp4(output_path, output_path.split('.')[0]) # TODO -2 -> 0/1/... assume regular path doesnt contain .
    else:
        if isinstance(args.seed, list):
            output_path = _main(args, pipeline, start_seed=args.seed[0])
        else:
            output_path = _main(args, pipeline, start_seed=args.seed)
        # save_images_to_mp4(output_path, output_path.split('.')[0])
    save_images_to_mp4(hdf5_path=output_path, output_dir=output_path[: -len(".h5")])
    print(f"Output hdf5 file {output_path}")

if __name__ == "__main__":
    # start = time.time()
    mp.set_start_method("spawn")
    args, pp = parse_args()
    main(args, pp)
    # print(f"Total time taken: {time.time() - start}")
