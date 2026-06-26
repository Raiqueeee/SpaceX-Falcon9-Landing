"""Configuration dataclasses for RocketLander environment."""

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple


@dataclass
class CurriculumConfig:
    """Configuration for curriculum learning.

    Curriculum learning starts training from lower heights and progressively
    increases to full height as the agent improves.
    """

    enabled: bool = False
    heights: Tuple[float, ...] = (5.0, 10.0, 20.0, 35.0, 50.0)
    success_threshold: float = 0.7  # 70% success rate to advance
    window_size: int = 100  # Episodes to average over


@dataclass
class RocketDesignConfig:
    """Configuration for a specific rocket design."""
    xml_file: str
    target_height: float  # Varies by leg design
    name: str


# Registry of available rocket designs
# target_height = rocket center z-position when landed upright
ROCKET_DESIGNS: Dict[str, RocketDesignConfig] = {
    "v0": RocketDesignConfig("single_rocket_test.xml", 1.0, "cylinder"),
    "v1": RocketDesignConfig("rocket_v1_three_legs.xml", 1.93, "tripod"),
    "demo": RocketDesignConfig("demo_v0.xml", 1.93, "detailed_tripod"),
}


@dataclass
class DomainRandomizationConfig:
    """Configuration for domain randomization during training.

    Default values are tight for stable early training.
    Widen ranges gradually for robustness after initial convergence.
    """
    enabled: bool = False
    mass_range: Tuple[float, float] = (0.95, 1.05)  # ±5% mass variation
    thrust_range: Tuple[float, float] = (0.95, 1.05)  # ±5% thrust variation
    gravity_range: Tuple[float, float] = (9.7, 9.9)  # Tight around Earth's 9.81
    initial_height_range: Tuple[float, float] = (48.0, 52.0)  # ±4m around 50m
    initial_velocity_range: Tuple[float, float] = (-0.5, 0.5)  # Small perturbations
    initial_orientation_range: Tuple[float, float] = (-2.0, 2.0)  # ±2° tilt


@dataclass
class RewardWeights:
    """Weights for different reward components.

    Exponential shaping: per-step components in [0, 1].
    distance rewards low altitude + pad alignment.
    time_penalty is subtracted every step to punish hovering.
    """
    distance: float = 0.5    # altitude + horizontal alignment
    velocity: float = 0.15
    upright: float = 0.08
    angular: float = 0.02
    success: float = 5000.0
    crash: float = -50.0
    tipover: float = -30.0
    time_penalty: float = 0.0  # per-step cost; set via config to discourage hovering


@dataclass
class RocketEnvConfig:
    """Main configuration for the RocketLander environment."""
    design: str = "v0"
    domain_randomization: DomainRandomizationConfig = field(
        default_factory=DomainRandomizationConfig
    )
    reward_weights: RewardWeights = field(default_factory=RewardWeights)
    curriculum: CurriculumConfig = field(default_factory=CurriculumConfig)
    max_episode_length: int = 1000
    frame_skip: int = 5
    reset_noise_scale: float = 0.01
    verbose: int = 0
    # Termination conditions
    max_distance: float = 20.0  # Max horizontal distance from pad before termination
    max_angle: float = 70.0  # Max roll/pitch angle before termination

    def get_design_config(self) -> RocketDesignConfig:
        """Get the RocketDesignConfig for the current design."""
        if self.design not in ROCKET_DESIGNS:
            raise ValueError(
                f"Unknown rocket design '{self.design}'. "
                f"Available designs: {list(ROCKET_DESIGNS.keys())}"
            )
        return ROCKET_DESIGNS[self.design]
