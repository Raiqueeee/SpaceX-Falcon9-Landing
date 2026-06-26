# SpaceX Falcon 9 Landing Simulator

Train a reinforcement learning agent to land a rocket on a drone ship, SpaceX Falcon 9 style. Built with [TorchRL](https://github.com/pytorch/rl) for training.

<p align="center">
  <img src="docs/demo_aerial.png" width="600" alt="Rocket landing aerial view">
</p>

## Features

- **MuJoCo physics** with 6-DOF free joint rocket dynamics
- **PPO** (Proximal Policy Optimization) with GAE advantages
- **Domain randomization** for robust policies
- **Demo video renderer** with aerial and tracking camera angles

## Installation

Requires Python 3.10+. Runs on CPU — no GPU required.

```bash
# Clone and install dependencies
git clone https://github.com/BY571/SpaceX-Falcon9.git
cd SpaceX-Falcon9
uv sync

# Activate environment
source .venv/bin/activate  # Linux/macOS
# or: .venv\Scripts\activate  # Windows
```

## Training

PPO training on CPU:

```bash
cd training && python train_ppo.py
```

Switch between rocket designs via config:

```bash
# Detailed tripod with legs (default)
python train_ppo.py env.rocket.design=demo

# Simple cylinder (faster physics, no legs)
python train_ppo.py env.rocket.design=v0
```

Training logs to [Weights & Biases](https://wandb.ai). Override any config parameter via CLI:

```bash
# Change seed
python train_ppo.py env.seed=123

# More environments
python train_ppo.py env.num_envs=8192

# Longer training
python train_ppo.py collector.total_frames=100_000_000

# Offline logging (no W&B upload)
python train_ppo.py logger.mode=offline
```

Checkpoints are saved to `training/checkpoints/`.

## Demo Videos

Generate high-quality demo videos from a trained checkpoint:

```bash
python env/demo_render.py --checkpoint training/checkpoints/ppo_final.pt
```

This produces two videos in `videos/`:

| Aerial view | Tracking view |
|:-----------:|:-------------:|
| <img src="docs/demo_aerial.png" width="380"> | <img src="docs/demo_tracking.png" width="380"> |

Options:

```bash
# Custom resolution and FPS
python env/demo_render.py --checkpoint <path> --resolution 720 --fps 30

# Override starting height
python env/demo_render.py --checkpoint <path> --height 30

# Custom output directory
python env/demo_render.py --checkpoint <path> --output-dir my_videos/
```

## Environment

### Observation Space (12-dim continuous)

| Index | Component | Description |
|-------|-----------|-------------|
| 0-2 | `pos_x, pos_y, pos_z` | Position in meters (target pad at origin) |
| 3-5 | `roll, pitch, yaw` | Orientation in degrees |
| 6-8 | `vel_x, vel_y, vel_z` | Linear velocity in m/s |
| 9-11 | `ang_x, ang_y, ang_z` | Angular velocity in rad/s |

### Action Space (3-dim continuous)

| Index | Component | Range | Description |
|-------|-----------|-------|-------------|
| 0 | `thrust_x` | [-1, 1] | Lateral thrust X |
| 1 | `thrust_y` | [-1, 1] | Lateral thrust Y |
| 2 | `thrust_z` | [0, 1] | Main engine |

### Reward Function

**Per-step shaping** (weighted sum of exponential components, all in [0, 1]):

| Component | Weight | Formula | Purpose |
|-----------|--------|---------|---------|
| Distance | 0.6 | `exp(-0.05 * dist_3d)` | Approach the landing pad |
| Velocity | 0.25 | `exp(-coeff * excess_vel)` | Slow down near ground (altitude-dependent safe speed) |
| Upright | 0.1 | `exp(-2.0 * tilt_rad)` | Stay vertical |
| Angular | 0.05 | `exp(-0.5 * ang_mag)` | Minimize rotation |
| Time penalty | — | `-0.125` per step | Incentivize landing quickly |

**Terminal rewards:**

| Event | Reward | Condition |
|-------|--------|-----------|
| Success | `+2000 * exp(-1 * approach_vel)` | Near pad, slow, upright, at target height |
| Crash | -10 | Below z=0.5m or hard landing (>5 m/s near surface) |
| Tipover | -10 | Roll or pitch exceeds 70° |

### Episode Termination

- **Success**: within 2m of pad center, altitude near target height, velocity < 1 m/s, tilt < 15°
- **Crash**: altitude < 0.5m, or near-surface contact above 5 m/s
- **Tipover**: roll or pitch > 70°
- **Out of bounds**: horizontal distance > 20m from pad
- **Truncation**: 1000 steps (25s at 40Hz control)

### Domain Randomization

Initial conditions are randomized at each episode reset to produce robust policies:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `reset_pos_noise` | 3.0 m | XY position offset from pad center |
| `reset_vel_noise` | 3.0 m/s | Linear velocity perturbation |
| `reset_ang_noise` | 0.15 rad (~8°) | Initial orientation tilt |
| `reset_angvel_noise` | 0.3 rad/s | Angular velocity perturbation |

The rocket starts at 50m altitude with these perturbations applied, so it must learn to handle off-center starts, initial tilt, and nonzero velocity.

## Rocket Design

Two models share the same physics (mass, actuators, joint):

| v0 — Training (fast) | v1 — Demo (full detail) |
|:---------------------:|:-----------------------:|
| <img src="rocket_designs/screenshots/design_v0.png" width="300"> | <img src="rocket_designs/screenshots/design_v1.png" width="300"> |
| Simple cylinder | Tripod with landing legs |

Both have: 10 kg body, free joint (6 DOF), lateral thrust (x/y, 25N), main engine (z, 200N). The policy trained on v0 transfers to v1 since the legs are passive geometry that don't affect the control problem.
