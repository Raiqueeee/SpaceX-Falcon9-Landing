# Falcon 9 Landing Simulator

A physics-accurate rocket landing simulation with two control approaches: a **reinforcement learning agent** trained with PPO and a **deterministic classical controller** modelled on the real Falcon 9 flight computer. Both controllers operate a 6-DOF MuJoCo rocket from altitude to a precision pad landing.

<p align="center">
  <img src="docs/rocket_preview.png" width="680" alt="Falcon 9 landing simulation">
</p>

---

## Two Control Approaches

### Approach 1 — Reinforcement Learning (PPO)

A neural network policy trained from scratch using Proximal Policy Optimization. The agent observes position, velocity, orientation and angular rates, and learns to fire three actuators (lateral x/y + main engine) through trial and error over millions of simulated episodes.

- **Library**: [TorchRL](https://github.com/pytorch/rl) with GAE advantages
- **Training**: CPU-only, ~3M frames, ~79 minutes at ~1000 fps
- **Logging**: Weights & Biases
- **Status**: Agent reaches the pad area and descends but impact velocity (~3 m/s) exceeds the 1 m/s landing threshold — requires longer training or reward tuning to achieve clean touchdowns

### Approach 2 — Classical FSM Controller

A deterministic finite state machine controller that requires no training. Designed to mirror the actual Falcon 9 first-stage guidance logic: the rocket free-falls to build velocity, then fires a precisely-timed "suicide burn" to decelerate and land.

- **5 flight phases**: `FREE_FALL → GUIDANCE → ALIGNMENT → BURN → TOUCHDOWN`
- **Physics-timed ignition**: burn trigger computed from stopping distance `v²/(2a_net)` + margin
- **Cascade PID guidance**: position → desired velocity → tilt setpoint → nozzle gimbal
- **Energy-based TOUCHDOWN**: thrust = `m(v²−v_tgt²)/(2d) + mg` at every step
- **Status**: 100% landing success with no training required

---

## Results

| Metric | PPO (3M frames) | Classical FSM |
|---|:---:|:---:|
| Landing success rate | ~0% | **100%** |
| Touchdown velocity | ~3 m/s | < 0.6 m/s |
| Training required | ~79 min | None |
| Robustness (±3m offset) | — | 90% |
| Failure scenario tolerance | — | 3 / 8 survived |

---

## Rocket Models

Two models share identical physics — same mass, actuators, and free joint:

| `v0` — Training | `demo` — Cinematic |
|:---:|:---:|
| <img src="rocket_designs/screenshots/design_v0.png" width="280"> | <img src="rocket_designs/screenshots/design_v1.png" width="280"> |
| Simple cylinder, fast physics | Tripod with deployable legs |

Policies trained on `v0` transfer to `demo` — the legs are passive geometry and do not affect the control problem.

---

## Environment

### Observation Space (12-dim)

| Index | Signal | Unit |
|---|---|---|
| 0–2 | `pos_x, pos_y, pos_z` | m (pad at origin) |
| 3–5 | `roll, pitch, yaw` | degrees |
| 6–8 | `vel_x, vel_y, vel_z` | m/s (world frame) |
| 9–11 | `ang_x, ang_y, ang_z` | rad/s (body frame) |

### Action Space (3-dim continuous)

| Index | Actuator | Range | Force |
|---|---|---|---|
| 0 | `thrust_x` | [−1, 1] | ±25 N lateral |
| 1 | `thrust_y` | [−1, 1] | ±25 N lateral |
| 2 | `thrust_z` | [0, 1] | 0–200 N main engine |

The nozzle site is 1.5 m below the centre of mass. Lateral thrust creates torques that tilt the rocket; the main engine vector then produces horizontal force. This mirrors real TVC (thrust vector control).

### Reward Function (PPO)

| Component | Weight | Formula |
|---|---|---|
| Distance to pad | 0.60 | `exp(−0.05 · dist_3d)` |
| Velocity control | 0.25 | `exp(−k · excess_vel)` |
| Upright attitude | 0.10 | `exp(−2.0 · tilt_rad)` |
| Angular stability | 0.05 | `exp(−0.5 · ω_mag)` |
| Time penalty | — | −0.125 per step |

Terminal: **+2000 · exp(−approach_vel)** on success · **−10** on crash or tipover.

### Episode Termination

| Condition | Trigger |
|---|---|
| **Success** | z < 1.98 m · vel < 1 m/s · dist < 2 m · tilt < 15° |
| **Hard crash** | z < 0.5 m or near-surface contact > 5 m/s |
| **Tipover** | roll or pitch > 70° |
| **Out of bounds** | horizontal distance > 20 m |
| **Truncation** | 1000 steps (25 s at 40 Hz) |

---

## Installation

Requires Python 3.10+. Runs entirely on CPU.

```bash
git clone https://github.com/BY571/SpaceX-Falcon9.git
cd SpaceX-Falcon9
uv sync
```

```bash
# Activate
source .venv/bin/activate        # Linux / macOS
.venv\Scripts\activate           # Windows
```

---

## Training (PPO)

```bash
cd training && python train_ppo.py
```

Key overrides (Hydra CLI):

```bash
python train_ppo.py env.rocket.design=v0          # fast cylinder (default)
python train_ppo.py env.rocket.design=demo         # detailed model with legs
python train_ppo.py collector.total_frames=50_000_000
python train_ppo.py env.num_envs=8192
python train_ppo.py env.seed=123
python train_ppo.py logger.mode=offline            # no W&B upload
```

Checkpoints are saved to `training/checkpoints/`. Training logs stream to [Weights & Biases](https://wandb.ai).

### Domain Randomisation

Each episode reset applies noise so the policy learns to handle off-nominal starts:

| Parameter | Default |
|---|---|
| XY position offset | ±3.0 m |
| Linear velocity perturbation | ±3.0 m/s |
| Initial tilt | ±0.15 rad (~8°) |
| Angular velocity | ±0.3 rad/s |

---

## Classical Controller Demo

No checkpoint required — runs immediately:

```bash
python env/demo_classical.py
```

Options:

```bash
python env/demo_classical.py --height 15          # starting altitude (m)
python env/demo_classical.py --attempts 5          # rollouts; best is rendered
python env/demo_classical.py --resolution 1080     # output resolution
python env/demo_classical.py --no-overlay          # disable HUD
python env/demo_classical.py --diagnose            # verbose phase trace
```

Renders two videos to `videos/classical/`:

| Aerial view | Tracking camera |
|:---:|:---:|
| `classical_aerial.mp4` | `classical_tracking.mp4` |

The HUD overlay shows flight phase (colour-coded), throttle, lateral error, vertical speed, and a burn indicator.

---

## Failure Scenarios

Eight real-world failure modes, each selectable with `--scenario`:

```bash
python env/demo_classical.py --scenario ENGINE_RELIGHT
python env/demo_classical.py --scenario HYPERSONIC_TUMBLE --resolution 1080
```

| Scenario flag | Description | Outcome |
|---|---|---|
| `HYDRAULIC_EXHAUSTION` | Grid fins degrade after 60% of descent | SUCCESS (barely) |
| `ENGINE_RELIGHT` | Landing burn ignites 0.5 s late | MISSED_PAD |
| `HYPERSONIC_TUMBLE` | Starts at 30° pitch with 1.5 rad/s spin | TIPOVER |
| `ROUGH_SEAS` | Pad drifts ±2 m at 0.3 Hz | MISSED_PAD |
| `CROSSWIND` | Constant 8 m/s lateral wind | SUCCESS |
| `SENSOR_DRIFT` | GPS reading drifts 1.5 m over descent | SUCCESS |
| `LATE_SEPARATION` | Enters with 15 m/s horizontal velocity | MISSED_PAD |
| `PARTIAL_THRUSTER` | Lateral thruster X stuck at +30% | TIPOVER |

Each scenario renders its own video with a live disturbance status strip in the HUD:

```
[SCENARIO] Engine Relight Failure  |  ENGINE SUPPRESSED — RELIGHT IN 0.25s
```

Outputs go to `videos/scenarios/scenario_<name>_aerial.mp4`.

---

## PPO Demo

Generate demo video from a trained checkpoint:

```bash
python env/demo_render.py --checkpoint training/checkpoints/ppo_final.pt
python env/demo_render.py --checkpoint <path> --resolution 720 --height 30
```

---

## Project Structure

```
SpaceX-Falcon9/
├── env/
│   ├── rocket_landing.py        # MuJoCo Gym environment
│   ├── classical_controller.py  # Deterministic FSM controller (no training)
│   ├── demo_classical.py        # Classical demo renderer + scenario system
│   ├── demo_render.py           # PPO checkpoint renderer
│   └── xml_files/               # MuJoCo rocket models
├── training/
│   ├── train_ppo.py             # PPO training script
│   └── conf/                    # Hydra config files
├── rocket_designs/              # Blender source + screenshots
├── videos/
│   ├── classical/               # Baseline landing videos
│   └── scenarios/               # Failure scenario videos
└── docs/                        # Images for README
```

---

## Stack

| Component | Library |
|---|---|
| Physics simulation | [MuJoCo 3](https://mujoco.org/) |
| RL training | [TorchRL](https://github.com/pytorch/rl) + PyTorch |
| Config management | [Hydra](https://hydra.cc/) |
| Experiment tracking | [Weights & Biases](https://wandb.ai/) |
| Video rendering | [imageio](https://imageio.readthedocs.io/) + [Pillow](https://pillow.readthedocs.io/) |
| Package management | [uv](https://docs.astral.sh/uv/) |
