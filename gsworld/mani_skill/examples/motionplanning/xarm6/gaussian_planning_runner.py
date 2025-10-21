import os
import argparse
from gsworld.utils.gs_utils import PipelineParams, get_combined_args
from gsworld.constants import *
from copy import deepcopy

class GaussianPlannningRunner:
    """
    Runner class for ManiSkill motion planning experiments.
    Encapsulates all parameters and provides methods to run experiments.
    """
    def __init__(self):
        self.parser, self.pipeline = self._create_parser()
        self.args = None
    
    def _create_parser(self):
        """Create argument parser with all necessary parameters."""
        parser = argparse.ArgumentParser(description="GS World")
        pipeline = PipelineParams(parser)
        
        # General settings
        parser.add_argument("--iteration", default=-1, type=int)
        parser.add_argument("--skip_train", action="store_true")
        parser.add_argument("--skip_test", action="store_true")
        parser.add_argument("--quiet", action="store_true")

        # ManiSkill settings
        parser.add_argument("--obs_mode", default="rgb+segmentation", type=str, help="Observation mode")
        parser.add_argument("--sim_backend", default="auto", type=str, help="Which simulation backend to use. Can be 'auto', 'cpu', 'gpu'")
        parser.add_argument("--control_mode", default="pd_joint_pos", type=str, help="Control mode")
        parser.add_argument("--render_mode", default="sensors", type=str, help="Render mode")
        parser.add_argument("--pause", default=False, action="store_true", help="If using human render mode, auto pauses the simulation upon loading")
        parser.add_argument("--quiet_ms", default=False, action="store_true", help="Disable verbose output.")
        parser.add_argument("--seed", default=0, type=int, nargs="+", help="Seed(s) for random actions and simulator. Can be a single integer or a list of integers. Default is None (no seeds)")
        parser.add_argument("--reward_mode", default=None, type=str, help="Reward mode")
        parser.add_argument("--num_envs", default=1, type=int, help="Number of environments to run.")
        parser.add_argument("--ep_len", default=100, type=int, help="Number of max episode length.")
        
        # Environment and task settings
        parser.add_argument("-e", "--env-id", type=str, default="PickALignYCBXArmEnv-v1", help="Environment to run motion planning solver on")
        parser.add_argument("-n", "--num-traj", type=int, default=1, help="Number of trajectories to generate.")
        parser.add_argument("-r", "--robot-uid", type=str, default="xarm6_uf_gripper", help="Robot uid.")
        parser.add_argument("--only-count-success", action="store_true", help="If true, generates trajectories until num_traj of them are successful and only saves the successful trajectories/videos")
        parser.add_argument("-b", "--sim-backend", type=str, default="auto", help="Which simulation backend to use. Can be 'auto', 'cpu', 'gpu'")
        
        # Rendering and output settings
        parser.add_argument("--render-mode", type=str, default="rgb_array", help="can be 'sensors' or 'rgb_array' which only affect what is saved to videos")
        parser.add_argument("--vis", action="store_true", help="whether or not to open a GUI to visualize the solution live")
        parser.add_argument("--save-video", action="store_true", help="whether or not to save videos locally")
        parser.add_argument("--traj-name", type=str, help="The name of the trajectory .h5 file that will be created.")
        parser.add_argument("--shader", default="default", type=str, help="Change shader used for rendering. Default is 'default' which is very fast. Can also be 'rt' for ray tracing and generating photo-realistic renders. Can also be 'rt-fast' for a faster but lower quality ray-traced renderer")
        parser.add_argument("--record-dir", type=str, default="/data/guangqi/maniskill/demos", help="where to save the recorded trajectories")
        
        # Parallel processing
        parser.add_argument("--num_procs", type=int, default=1, help="Number of processes to use to help parallelize the trajectory replay process. This uses CPU multiprocessing and only works with the CPU simulation backend at the moment.")

        # Scene configuration
        parser.add_argument("--scene_cfg_name", default=None, type=str, help="Path to scene json config file")
        
        # Simulation frequency settings
        parser.add_argument("--sim_freq", default=120, type=int, help="Simulator physics freq.")
        parser.add_argument("--control_freq", default=40, type=int, help=".step control freq.")
        
        # Recovery state
        parser.add_argument("--recovery_state_logger_path", default=None, type=str, help="Path to directory to stores (multiple) state loggers for recovery. If not given, the env will randomly initialized.")

        return parser, pipeline
    
    def parse_args(self, args=None):
        """Parse command line arguments."""
        self.args = self.parser.parse_args(args)
        return self.args
    
    def run(self, args=None):
        """Run the motion planning experiment with the provided or parsed arguments."""
        from gsworld.mani_skill.examples.motionplanning.xarm6.run_with_gs import main
        
        if args is not None:
            self.args = args
        elif self.args is None:
            self.parse_args()
            
        main(self.args, self.pipeline)
        
    def run_with_config(self, config_dict):
        """
        Run with a configuration dictionary that overrides default arguments.
        
        Args:
            config_dict (dict): Dictionary of argument name and value pairs
        """
        # Parse default args first
        args = self.parse_args([])
        # Override with provided config
        for key, value in config_dict.items():
            if hasattr(args, key):
                setattr(args, key, value)
            else:
                print(f"Warning: Unknown parameter '{key}'")
        
        # Run with the modified args
        self.run(args)


# Example usage
if __name__ == "__main__":
    import multiprocessing as mp
    mp.set_start_method("spawn")
    
    runner = GaussianPlannningRunner()
    
    # Method 1: Run with command line arguments
    # runner.run()
    
    # Method 2: Run with config dictionary
    config = {
        "env_id": "PickALignYCBXArmEnv-v1",
        "scene_cfg_name": "xarm6_icra_align", # 
        "num_traj": 2,
        "sim_backend": "cpu",
        "save_video": True,
        "only_count_success": False,
        "record_dir": os.path.join(os.path.dirname(os.path.abspath(__file__)), "exp_log/icra_align"),
        "num_procs": 2, # ~20cpu/procs
        "obs_mode": "rgb",
        "ep_len": 500,
        "vis": True,
    }
    runner.run_with_config(config)