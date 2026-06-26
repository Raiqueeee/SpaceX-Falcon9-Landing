"""Render demo videos from a trained PPO checkpoint.

Runs the policy in the training env to collect a trajectory (qpos/qvel),
then replays it in a demo MuJoCo XML with two camera angles:
  - Aerial:   fixed high camera looking down at the pad
  - Tracking: closer camera that follows the rocket

Usage:
    python env/demo_render.py --checkpoint training/checkpoints/ppo_final.pt
    python env/demo_render.py --checkpoint training/checkpoints/ppo_final.pt --resolution 1080
"""

import argparse
import os
import sys

import imageio
import mujoco
import numpy as np
import torch
from torch import nn
from tensordict.nn import AddStateIndependentNormalScale, TensorDictModule
from torchrl.envs import GymWrapper, TransformedEnv, Compose
from torchrl.envs.transforms import DoubleToFloat, InitTracker, ObservationNorm, RewardSum, StepCounter
from torchrl.envs.utils import ExplorationType, set_exploration_type
from torchrl.modules import MLP, ProbabilisticActor
from torchrl.modules.distributions import TanhNormal

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

# Must match OBS_SCALE in training/utils_ppo.py
_OBS_SCALE = torch.tensor(
    [20., 20., 15., 90., 90., 180., 15., 15., 15., 10., 10., 10.],
    dtype=torch.float32,
)

from env.rocket_landing import RocketLander

DEMO_XML = os.path.join(os.path.dirname(__file__), "xml_files", "demo_v0.xml")


def build_env(rocket_design, curriculum_height=None, max_episode_steps=1000):
    """Create the training env (no pixels) for trajectory collection."""
    os.chdir(ROOT_DIR)
    rocket_env = RocketLander(
        rocket_design=rocket_design,
        render_mode="rgb_array",
        width=64,
        height=64,
    )
    if curriculum_height is not None:
        rocket_env.set_curriculum_height(curriculum_height)

    env = GymWrapper(rocket_env, device="cpu", from_pixels=False)
    env = TransformedEnv(
        env,
        Compose(
            StepCounter(max_steps=max_episode_steps),
            InitTracker(),
            DoubleToFloat(),
            ObservationNorm(
                in_keys=["observation"],
                loc=torch.zeros(12, dtype=torch.float32),
                scale=_OBS_SCALE,
                standard_normal=True,
            ),
            RewardSum(),
        ),
    )
    return env, rocket_env


def build_ppo_actor(env, hidden_sizes, activation, device):
    """Reconstruct PPO actor (ProbabilisticActor with AddStateIndependentNormalScale)."""
    action_spec = env.action_spec
    act_cls = {"relu": nn.ReLU, "tanh": nn.Tanh, "leaky_relu": nn.LeakyReLU}[activation]
    num_outputs = action_spec.shape[-1]
    input_shape = env.observation_spec["observation"].shape

    policy_mlp = MLP(
        in_features=input_shape[-1],
        activation_class=act_cls,
        out_features=num_outputs,
        num_cells=hidden_sizes,
        device=device,
    )

    policy_mlp = nn.Sequential(
        policy_mlp,
        AddStateIndependentNormalScale(num_outputs, scale_lb=1e-8).to(device),
    )

    actor = ProbabilisticActor(
        TensorDictModule(
            module=policy_mlp,
            in_keys=["observation"],
            out_keys=["loc", "scale"],
        ),
        in_keys=["loc", "scale"],
        spec=action_spec,
        distribution_class=TanhNormal,
        distribution_kwargs={
            "low": action_spec.space.low,
            "high": action_spec.space.high,
            "tanh_loc": False,
        },
        return_log_prob=True,
        default_interaction_type=ExplorationType.RANDOM,
    )

    with torch.no_grad(), set_exploration_type(ExplorationType.RANDOM):
        td = env.fake_tensordict().to(device)
        actor(td)

    return actor


def collect_trajectory(env, rocket_env, actor, max_steps, linger_steps=240):
    """Step the policy and record qpos/qvel at each timestep.

    Returns (trajectory, final_altitude, landed) where landed=True if the
    episode ended with crash_report==1 (successful landing).
    """
    trajectory = []
    landed = False
    final_alt = float("inf")

    with set_exploration_type(ExplorationType.DETERMINISTIC), torch.no_grad():
        td = env.reset()
        for _ in range(max_steps):
            td = actor(td)
            td = env.step(td)

            qpos = rocket_env.data.qpos.copy()
            qvel = rocket_env.data.qvel.copy()
            trajectory.append((qpos, qvel))
            final_alt = float(qpos[2])

            done = td["next", "done"].item()
            if done:
                # Check if we actually landed (crash_report == 1)
                if ("next", "crash_report") in td.keys(True):
                    landed = int(td["next", "crash_report"].item()) == 1
                for _ in range(linger_steps):
                    mujoco.mj_step(rocket_env.model, rocket_env.data)
                    trajectory.append((rocket_env.data.qpos.copy(), rocket_env.data.qvel.copy()))
                break
            td = td["next"]

    return trajectory, final_alt, landed


def collect_best_trajectory(env, rocket_env, actor, max_steps, linger_steps=240, attempts=1):
    """Run the policy `attempts` times and return the trajectory with the lowest final altitude.

    Prefers successful landings (crash_report==1) over close-but-not-landed attempts.
    """
    best_traj, best_alt, best_landed = None, float("inf"), False

    for i in range(attempts):
        traj, final_alt, landed = collect_trajectory(env, rocket_env, actor, max_steps, linger_steps)
        status = "LANDED" if landed else f"alt={final_alt:.1f}m"
        print(f"  Attempt {i + 1}/{attempts}: {status}")

        if best_traj is None:
            best_traj, best_alt, best_landed = traj, final_alt, landed
            continue

        # Prefer a successful landing over a lower final altitude from a crash
        if landed and not best_landed:
            best_traj, best_alt, best_landed = traj, final_alt, landed
        elif not landed and not best_landed and final_alt < best_alt:
            best_traj, best_alt, best_landed = traj, final_alt, landed

    print(f"  Best: {'LANDED' if best_landed else f'alt={best_alt:.1f}m'}")
    return best_traj


def make_camera(lookat, distance, azimuth, elevation):
    """Create an MjvCamera with the given parameters."""
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = lookat
    cam.distance = distance
    cam.azimuth = azimuth
    cam.elevation = elevation
    return cam


def render_trajectory(trajectory, resolution, fps, output_dir, camera_distance=80):
    """Render the trajectory from aerial and tracking cameras."""
    model = mujoco.MjModel.from_xml_path(DEMO_XML)
    data = mujoco.MjData(model)

    width = int(resolution * 16 / 9)
    height = resolution
    renderer = mujoco.Renderer(model, height=height, width=width)

    aerial_frames = []
    tracking_frames = []

    for qpos, qvel in trajectory:
        data.qpos[:] = qpos
        data.qvel[:] = qvel
        mujoco.mj_forward(model, data)

        rocket_pos = qpos[:3]

        # --- Aerial camera ---
        lookat_z = rocket_pos[2] * 0.4
        cam = make_camera([0.0, 0.0, lookat_z], camera_distance, 135, -25)
        renderer.update_scene(data, camera=cam)
        aerial_frames.append(renderer.render().copy())

        # --- Tracking camera: offset above and to the side, looking down ---
        cam = make_camera(rocket_pos.copy(), 25, 135, -30)
        renderer.update_scene(data, camera=cam)
        tracking_frames.append(renderer.render().copy())

    renderer.close()

    os.makedirs(output_dir, exist_ok=True)

    aerial_path = os.path.join(output_dir, "demo_aerial.mp4")
    tracking_path = os.path.join(output_dir, "demo_tracking.mp4")

    save_video(aerial_frames, aerial_path, fps)
    save_video(tracking_frames, tracking_path, fps)

    return aerial_path, tracking_path


def save_video(frames, path, fps):
    """Save frames as MP4."""
    if not frames:
        print(f"  No frames to save for {path}")
        return
    writer = imageio.get_writer(path, fps=fps, quality=8, codec="libx264", macro_block_size=1)
    for frame in frames:
        writer.append_data(frame)
    writer.close()
    size_mb = os.path.getsize(path) / (1024 * 1024)
    print(f"  Saved: {path} ({len(frames)} frames, {size_mb:.1f} MB)")


def main():
    parser = argparse.ArgumentParser(description="Render demo videos from a trained PPO checkpoint")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to PPO checkpoint .pt file")
    parser.add_argument("--resolution", type=int, default=1080, help="Video height in pixels (default: 1080)")
    parser.add_argument("--fps", type=int, default=24, help="Video FPS (default: 24, lower = longer video)")
    parser.add_argument("--max-steps", type=int, default=1000, help="Max steps per episode")
    parser.add_argument("--linger", type=int, default=240, help="Extra frames to record after landing (default: 240)")
    parser.add_argument("--attempts", type=int, default=3, help="Number of rollouts to try; best (lowest altitude) is rendered (default: 3)")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory (default: videos/)")
    parser.add_argument("--rocket-design", type=str, default=None, help="Override rocket design (v0/v1)")
    parser.add_argument("--height", type=float, default=None, help="Override starting height")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for initial conditions")
    args = parser.parse_args()

    device = torch.device("cpu")

    print(f"Loading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)

    ckpt_config = checkpoint.get("config", {})
    rocket_design = args.rocket_design
    if rocket_design is None:
        rocket_design = ckpt_config.get("env", {}).get("rocket", {}).get("design", "v0")
    curriculum_height = args.height
    if curriculum_height is None:
        curriculum_height = checkpoint.get("curriculum_height", None)

    net_cfg = ckpt_config.get("network", {})
    hidden_sizes = net_cfg.get("hidden_sizes", [256, 256])
    activation = net_cfg.get("activation", "relu")

    print(f"  Rocket design: {rocket_design}")
    print(f"  Resolution: {int(args.resolution * 16 / 9)}x{args.resolution}")
    print(f"  Network: {hidden_sizes}, {activation}")
    if curriculum_height is not None:
        print(f"  Starting height: {curriculum_height}m")

    # Build training env for trajectory collection
    env, rocket_env = build_env(
        rocket_design=rocket_design,
        curriculum_height=curriculum_height,
        max_episode_steps=args.max_steps,
    )

    # Build and load actor
    actor = build_ppo_actor(env, hidden_sizes, activation, device)
    actor.load_state_dict(checkpoint["actor_state_dict"])
    actor.eval()

    frames_trained = checkpoint.get("collected_frames", "unknown")
    print(f"  Trained for {frames_trained} frames")

    # Set seed for reproducible initial conditions
    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

    # Collect trajectory — run multiple attempts and pick the one that gets closest to landing
    print(f"\nCollecting trajectory ({args.attempts} attempt(s))...")
    trajectory = collect_best_trajectory(
        env, rocket_env, actor, args.max_steps,
        linger_steps=args.linger, attempts=args.attempts,
    )
    print(f"  Total frames recorded: {len(trajectory)}")

    if trajectory:
        final_pos = trajectory[-1][0][:3]
        print(f"  Final position: x={final_pos[0]:.2f}, y={final_pos[1]:.2f}, z={final_pos[2]:.2f}")

    # Render from demo XML
    output_dir = args.output_dir or os.path.join(ROOT_DIR, "videos")
    print(f"\nRendering demo videos...")
    aerial_path, tracking_path = render_trajectory(
        trajectory, args.resolution, args.fps, output_dir
    )

    print(f"\nDone! Videos saved to {output_dir}/")
    env.close()


if __name__ == "__main__":
    main()
