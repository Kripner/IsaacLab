"""Record a video of a trained policy using the Newton OpenGL viewer.

Usage:
    python scripts/reinforcement_learning/rsl_rl/record_video.py \
        --task Isaac-Reach-Franka-Warp-v0 --num_envs 4 \
        --load_run <timestamp> --num_steps 500 --output video.mp4
"""

import argparse
import sys

from isaaclab.app import AppLauncher

import cli_args  # isort: skip

parser = argparse.ArgumentParser(description="Record a video of a trained RL policy.")
parser.add_argument("--num_envs", type=int, default=4, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--num_steps", type=int, default=500, help="Number of steps to record.")
parser.add_argument("--output", type=str, default="policy_video.mp4", help="Output video file path.")
parser.add_argument("--fps", type=int, default=30, help="Video frames per second.")
parser.add_argument("--width", type=int, default=1920, help="Video width.")
parser.add_argument("--height", type=int, default=1080, help="Video height.")
parser.add_argument("--cam_pos", type=float, nargs=3, default=None, help="Camera position (x y z).")
parser.add_argument("--cam_pitch", type=float, default=None, help="Camera pitch in degrees.")
parser.add_argument("--cam_yaw", type=float, default=None, help="Camera yaw in degrees.")
parser.add_argument("--cam_fov", type=float, default=None, help="Camera field of view in degrees.")
parser.add_argument("--resample_time", type=float, default=None, help="Time between target changes (seconds).")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# Force headless + newton visualizer
# Don't set headless -- it disables visualizers. The newton visualizer
# will open a window but we only need it for get_frame() capture.
args_cli.visualizer = ["newton"]

sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import numpy as np
import os
import time
import torch
import warp as wp

from rsl_rl.runners import OnPolicyRunner

from isaaclab.utils import close_simulation, is_simulation_running
from isaaclab.utils.timer import Timer

Timer.enable = False
Timer.enable_display_output = False

import isaaclab_tasks_experimental  # noqa: F401

from isaaclab.envs import DirectRLEnvCfg, ManagerBasedRLEnvCfg
from isaaclab.utils.assets import retrieve_file_path

from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper, export_policy_as_jit, export_policy_as_onnx

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Record video of trained policy."""
    task_name = args_cli.task.split(":")[-1]
    train_task_name = task_name.replace("-Play", "")

    agent_cfg: RslRlBaseRunnerCfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # Override command resampling time if requested
    if args_cli.resample_time is not None and hasattr(env_cfg, 'commands'):
        for cmd_name in dir(env_cfg.commands):
            cmd = getattr(env_cfg.commands, cmd_name, None)
            if hasattr(cmd, 'resampling_time_range'):
                cmd.resampling_time_range = (args_cli.resample_time, args_cli.resample_time)

    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")

    if args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    # Create environment
    env = gym.make(args_cli.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(resume_path)
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    # Reset and get initial observations
    dt = env.unwrapped.step_dt
    obs = env.get_observations()

    # Get the Newton viewer from the initialized visualizers
    sim = env.unwrapped.sim
    if not sim._visualizers:
        sim.initialize_visualizers()

    viewer = None
    for v in sim._visualizers:
        if hasattr(v, '_viewer') and v._viewer is not None:
            viewer = v._viewer
            break

    if viewer is None or not hasattr(viewer, 'get_frame'):
        print("[ERROR] Could not find Newton viewer. Visualizers:", sim._visualizers)
        env.close()
        return

    # Set camera position and FOV (use CLI args if provided)
    if args_cli.cam_pos is not None:
        cam_pos = wp.vec3(*args_cli.cam_pos)
        cam_pitch = args_cli.cam_pitch if args_cli.cam_pitch is not None else -2.8
        cam_yaw = args_cli.cam_yaw if args_cli.cam_yaw is not None else -180.8
        viewer.set_camera(pos=cam_pos, pitch=cam_pitch, yaw=cam_yaw)
    if args_cli.cam_fov is not None:
        viewer.camera.fov = args_cli.cam_fov

    print(f"[INFO]: Using initialized Newton viewer for frame capture")

    # Collect frames
    raw_env = env.unwrapped
    frames = []

    print(f"[INFO]: Recording {args_cli.num_steps} steps to {args_cli.output}...")
    for step in range(args_cli.num_steps):
        with torch.inference_mode():
            actions = policy(obs)
            obs, _, _, _ = env.step(actions)

        # Draw target markers at fingertip position (offset from panda_hand along target orientation)
        if hasattr(raw_env, 'command_manager'):
            for term in raw_env.command_manager._terms.values():
                if hasattr(term, 'pose_command_w'):
                    target_pos = term.pose_command_w[:, :3]  # (n, 3)
                    target_quat = term.pose_command_w[:, 3:]  # (n, 4) as xyzw or wxyz

                    # Offset along the target's local z-axis by wrist-to-fingertip distance
                    # panda_hand z-axis points along the fingers; offset ~0.1m to reach fingertip
                    from isaaclab.utils.math import quat_apply
                    local_offset = torch.tensor([0.0, 0.0, 0.1], device=target_pos.device).expand_as(target_pos)
                    world_offset = quat_apply(target_quat, local_offset)
                    fingertip_pos = target_pos + world_offset

                    n = fingertip_pos.shape[0]
                    target_pos_wp = wp.from_torch(fingertip_pos.contiguous(), dtype=wp.vec3)
                    radii_wp = wp.full(n, 0.03, dtype=wp.float32, device=target_pos_wp.device)
                    colors_np = np.tile(np.array([1.0, 0.2, 0.2], dtype=np.float32), (n, 1))
                    colors_wp = wp.array(colors_np, dtype=wp.vec3, device=target_pos_wp.device)
                    viewer.log_points("targets", target_pos_wp, radii=radii_wp, colors=colors_wp)

        frame_wp = viewer.get_frame()
        if frame_wp is not None:
            frame_np = frame_wp.numpy()
            frames.append(frame_np.copy())

        if (step + 1) % 100 == 0:
            print(f"[INFO]: Frame {step + 1}/{args_cli.num_steps}")

    # Write video
    if frames:
        try:
            import imageio
            print(f"[INFO]: Writing {len(frames)} frames to {args_cli.output}...")
            writer = imageio.get_writer(args_cli.output, fps=args_cli.fps)
            for frame in frames:
                writer.append_data(frame)
            writer.close()
            print(f"[INFO]: Video saved: {args_cli.output} ({os.path.getsize(args_cli.output) / 1024 / 1024:.1f} MB)")
        except ImportError:
            print("[ERROR] imageio not installed. Install with: pip install imageio[ffmpeg]")
    else:
        print("[WARNING] No frames captured.")

    env.close()


if __name__ == "__main__":
    main()
    close_simulation(simulation_app)
