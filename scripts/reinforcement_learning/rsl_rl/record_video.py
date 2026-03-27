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
parser.add_argument("--width", type=int, default=1280, help="Video width.")
parser.add_argument("--height", type=int, default=720, help="Video height.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# Force headless + newton visualizer
args_cli.headless = True
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

    # Create a standalone Newton viewer for frame capture
    from newton.viewer import ViewerGL
    from isaaclab_newton.physics import NewtonManager

    model = NewtonManager.get_model()

    viewer = ViewerGL(width=args_cli.width, height=args_cli.height, headless=True)
    viewer.set_model(model)

    print(f"[INFO]: Newton viewer created ({args_cli.width}x{args_cli.height}, headless)")

    # Collect frames
    frames = []

    print(f"[INFO]: Recording {args_cli.num_steps} steps to {args_cli.output}...")
    for step in range(args_cli.num_steps):
        with torch.inference_mode():
            actions = policy(obs)
            obs, _, _, _ = env.step(actions)

        # Update viewer with current state and capture frame
        state = NewtonManager.get_state_0()
        viewer.begin_frame(step * dt)
        viewer.log_state(state)

        # Draw target markers if the env has a command manager with pose commands
        raw_env = env.unwrapped
        if hasattr(raw_env, 'command_manager'):
            for term in raw_env.command_manager._terms.values():
                if hasattr(term, 'pose_command_w'):
                    target_pos = term.pose_command_w[:, :3]  # (num_envs, 3)
                    n = target_pos.shape[0]
                    target_pos_wp = wp.from_torch(target_pos.contiguous(), dtype=wp.vec3)
                    radii_wp = wp.full(n, 0.03, dtype=wp.float32, device=target_pos_wp.device)
                    colors_np = np.tile(np.array([1.0, 0.2, 0.2], dtype=np.float32), (n, 1))
                    colors_wp = wp.array(colors_np, dtype=wp.vec3, device=target_pos_wp.device)
                    viewer.log_points("targets", target_pos_wp, radii=radii_wp, colors=colors_wp)

        viewer.end_frame()

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
