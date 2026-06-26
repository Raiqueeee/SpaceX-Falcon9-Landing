"""Utilities for CPU-based PPO training with RocketLander."""
import os
import sys

import torch
import torch.nn
from tensordict.nn import AddStateIndependentNormalScale, TensorDictModule
from torchrl.envs import Compose, ExplorationType, TransformedEnv
from torchrl.envs.transforms import StepCounter, InitTracker, RewardSum, DoubleToFloat, ObservationNorm
from torchrl.modules import MLP, ProbabilisticActor, TanhNormal, ValueOperator

# Add project root to path
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)


# ====================================================================
# General utils
# ====================================================================


def log_metrics(logger, metrics, step):
    for metric_name, metric_value in metrics.items():
        logger.log_scalar(metric_name, metric_value, step)


def get_activation(cfg):
    if cfg.network.activation == "relu":
        return torch.nn.ReLU
    elif cfg.network.activation == "tanh":
        return torch.nn.Tanh
    elif cfg.network.activation == "leaky_relu":
        return torch.nn.LeakyReLU
    else:
        raise NotImplementedError(f"Unknown activation: {cfg.network.activation}")


# ====================================================================
# Environment utils
# ====================================================================


def env_maker(cfg, curriculum_height=None, num_envs=None):
    """Create n RocketLander envs wrapped in SerialEnv for TorchRL (CPU)."""
    from env.rocket_landing import RocketLander
    from torchrl.envs import GymWrapper, SerialEnv

    n = num_envs if num_envs is not None else cfg.env.num_envs
    device = cfg.network.device or "cpu"

    # Map config reward weights to RocketLander's dict format
    reward_weights = {}
    if hasattr(cfg.env, "reward_weights"):
        for k in ["distance", "velocity", "upright", "angular", "success", "crash", "tipover", "time_penalty"]:
            if hasattr(cfg.env.reward_weights, k):
                reward_weights[k] = getattr(cfg.env.reward_weights, k)

    max_distance = 20.0
    max_angle = 70.0
    if hasattr(cfg.env, "termination"):
        if hasattr(cfg.env.termination, "max_distance"):
            max_distance = cfg.env.termination.max_distance
        if hasattr(cfg.env.termination, "max_angle"):
            max_angle = cfg.env.termination.max_angle

    rocket_design = "v0"
    if hasattr(cfg.env, "rocket") and hasattr(cfg.env.rocket, "design"):
        rocket_design = cfg.env.rocket.design

    # starting_height in config overrides the XML default (50 m); the explicit
    # curriculum_height argument (used by external callers) takes priority.
    config_height = None
    if hasattr(cfg.env, "starting_height"):
        config_height = float(cfg.env.starting_height)
    effective_height = curriculum_height if curriculum_height is not None else config_height

    def make_single_env():
        gym_env = RocketLander(
            rocket_design=rocket_design,
            reward_weights=reward_weights if reward_weights else None,
            max_episode_length=cfg.env.max_episode_steps,
            max_distance=max_distance,
            max_angle=max_angle,
        )
        if effective_height is not None:
            gym_env.set_curriculum_height(effective_height)
        return GymWrapper(gym_env, device=device)

    return SerialEnv(n, make_single_env)


# Fixed observation scale factors — divides each component into roughly [-1, 1].
# Must match OBS_SCALE in env/demo_render.py.
# Order: pos_x, pos_y, pos_z, roll, pitch, yaw, vel_x, vel_y, vel_z, av_x, av_y, av_z
OBS_SCALE = torch.tensor(
    [20., 20., 15., 90., 90., 180., 15., 15., 15., 10., 10., 10.],
    dtype=torch.float32,
)


def apply_env_transforms(env, max_episode_steps):
    transformed_env = TransformedEnv(
        env,
        Compose(
            StepCounter(max_steps=max_episode_steps),
            InitTracker(),
            DoubleToFloat(),
            # Normalise obs to ~[-1,1]: prevents large angle inputs (±180°) from
            # dominating the first network layer over position/velocity signals.
            ObservationNorm(
                in_keys=["observation"],
                loc=torch.zeros(12, dtype=torch.float32),
                scale=OBS_SCALE,
                standard_normal=True,
            ),
            RewardSum(),
        ),
    )
    return transformed_env


def make_environment(cfg, logger=None, curriculum_height=None):
    """Make environments for training and evaluation."""
    train_env = env_maker(cfg, curriculum_height=curriculum_height)
    train_env = apply_env_transforms(train_env, cfg.env.max_episode_steps)

    eval_env = env_maker(cfg, curriculum_height=curriculum_height, num_envs=1)
    eval_env = apply_env_transforms(eval_env, cfg.env.max_episode_steps)

    return train_env, eval_env


def make_render_env(cfg):
    """Create a CPU Gymnasium env with pixel rendering for video logging."""
    from env.rocket_landing import RocketLander
    from torchrl.envs import GymWrapper

    rocket_design = "v0"
    if hasattr(cfg.env, "rocket") and hasattr(cfg.env.rocket, "design"):
        rocket_design = cfg.env.rocket.design

    rocket_env = RocketLander(
        rocket_design=rocket_design,
        render_mode="rgb_array",
        width=256,
        height=256,
    )
    env = GymWrapper(rocket_env, device="cpu", from_pixels=True)
    env = TransformedEnv(
        env,
        Compose(
            StepCounter(max_steps=cfg.env.max_episode_steps),
            InitTracker(),
            DoubleToFloat(),
            RewardSum(),
        ),
    )
    return env


# ====================================================================
# PPO Model
# ---------


def make_ppo_models(cfg, train_env, device):
    """Build PPO actor and critic networks.

    Actor: MLP → AddStateIndependentNormalScale → ProbabilisticActor (returns log_prob)
    Critic: MLP → ValueOperator (predicts state_value)
    """
    input_shape = train_env.observation_spec["observation"].shape
    action_spec = train_env.action_spec
    if train_env.batch_size:
        action_spec = action_spec[(0,) * len(train_env.batch_size)]

    num_outputs = action_spec.shape[-1]
    activation_class = get_activation(cfg)
    hidden_sizes = cfg.network.hidden_sizes

    # --- Actor ---
    policy_mlp = MLP(
        in_features=input_shape[-1],
        activation_class=activation_class,
        out_features=num_outputs,
        num_cells=hidden_sizes,
        device=device,
    )

    # Orthogonal init (PPO standard)
    for layer in policy_mlp.modules():
        if isinstance(layer, torch.nn.Linear):
            torch.nn.init.orthogonal_(layer.weight, 1.0)
            layer.bias.data.zero_()

    policy_mlp = torch.nn.Sequential(
        policy_mlp,
        AddStateIndependentNormalScale(num_outputs, scale_lb=1e-8).to(device),
    )

    policy_module = ProbabilisticActor(
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

    # --- Critic ---
    value_mlp = MLP(
        in_features=input_shape[-1],
        activation_class=activation_class,
        out_features=1,
        num_cells=hidden_sizes,
        device=device,
    )

    # Orthogonal init with small scale for value head
    for layer in value_mlp.modules():
        if isinstance(layer, torch.nn.Linear):
            torch.nn.init.orthogonal_(layer.weight, 0.01)
            layer.bias.data.zero_()

    value_module = ValueOperator(
        value_mlp,
        in_keys=["observation"],
    )

    return policy_module, value_module
