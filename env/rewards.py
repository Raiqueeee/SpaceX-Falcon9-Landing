"""Modular reward calculator for RocketLander environment."""

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np

from env.config import RewardWeights


@dataclass
class RewardComponents:
    """Container for individual reward components for logging."""
    distance: float = 0.0
    velocity: float = 0.0
    upright: float = 0.0
    angular: float = 0.0
    bonus: float = 0.0
    total: float = 0.0

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary for logging."""
        return {
            "reward_distance": self.distance,
            "reward_velocity": self.velocity,
            "reward_upright": self.upright,
            "reward_angular": self.angular,
            "reward_bonus": self.bonus,
            "reward_total": self.total,
        }


class RewardCalculator:
    """Calculates rewards with exponential shaping and component tracking.

    All per-step components are in [0, 1] via exp(-k * x), matching the
    warp (GPU) environment reward function.
    """

    def __init__(
        self,
        weights: Optional[RewardWeights] = None,
        target_height: float = 1.02,
        starting_height: float = 50.0,
    ):
        self.weights = weights if weights is not None else RewardWeights()
        self.target_height = target_height
        self.starting_height = starting_height

    # Altitude gate in metres. Above this height every step bleeds reward
    # and zero positive signal is given. Forces the rocket to descend.
    DESCENT_GATE_M: float = 5.0

    def calculate(
        self,
        state: np.ndarray,
        crash_report: Optional[int] = None,
    ) -> tuple[float, RewardComponents]:
        """Calculate reward for the current rocket state.

        Reward structure:
          ABOVE 5 m  — flat penalty of -time_penalty every step, nothing positive.
          BELOW 5 m  — rich approach rewards (altitude, alignment, speed, attitude).
          Terminal    — large success bonus or crash/tipover penalty.

        Args:
            state: 12-dim array [pos(3), euler_deg(3), vel(3), ang_vel(3)]
            crash_report: 1=success, 2=crash, 3=roll_over, 4=pitch_over, 0/None=ongoing

        Returns:
            Tuple of (total_reward, RewardComponents)
        """
        (
            pos_x, pos_y, pos_z,
            roll_deg, pitch_deg, yaw_deg,
            vel_x, vel_y, vel_z,
            angular_vel_x, angular_vel_y, angular_vel_z,
        ) = state

        h_dist     = np.sqrt(pos_x**2 + pos_y**2)
        vel_mag    = np.sqrt(vel_x**2 + vel_y**2 + vel_z**2)
        tilt_rad   = (abs(roll_deg) + abs(pitch_deg)) * (np.pi / 180.0)
        ang_mag    = np.sqrt(angular_vel_x**2 + angular_vel_y**2 + angular_vel_z**2)

        components = RewardComponents()

        # --- Terminal bonuses apply at any altitude ---
        if crash_report == 1:
            # Quality-weighted landing bonus: slow + precise + upright
            r_speed = np.exp(-1.0 * vel_mag)    # 0.5 m/s → 0.61, 1 m/s → 0.37
            r_align = np.exp(-0.3 * h_dist)     # on pad → 1.0, 2 m off → 0.55
            r_up    = np.exp(-2.0 * tilt_rad)
            components.bonus = self.weights.success * r_speed * r_align * r_up
        elif crash_report == 2:
            components.bonus = self.weights.crash
        elif crash_report in (3, 4):
            components.bonus = self.weights.tipover

        # --- ABOVE gate: guided descent ---
        # Always negative (never profitable to stay up here), but the penalty SHRINKS
        # as the rocket descends at the right speed (target 5-8 m/s downward).
        # This gives the policy a concrete gradient: "go DOWN at a controlled rate."
        # Physics constraint: must arrive at gate ≤ 10 m/s to brake in 5 m.
        if pos_z > self.DESCENT_GATE_M:
            # descent_speed: 0 = hovering, positive = falling, capped at 8 m/s
            # (above 8 m/s there's no extra incentive — avoids rewarding free-fall)
            descent_speed = float(np.clip(-vel_z, 0.0, 8.0))
            descent_fraction = descent_speed / 8.0          # 0 → 1

            # Penalty shrinks from -time_penalty (hover) to -0.15*time_penalty (8 m/s)
            # Always stays negative — above the gate is always costly
            penalty = -self.weights.time_penalty * (1.0 - 0.85 * descent_fraction)

            components.total = penalty + components.bonus
            return components.total, components

        # --- BELOW gate: rich landing-approach rewards ---

        # Altitude: peaks at target_height (leg contact height), decays away.
        #   exp(-1.0 * |z - target|): 1.0 at target, 0.61 at ±0.5 m, 0.22 at ±1.5 m
        r_altitude   = np.exp(-1.0 * abs(pos_z - self.target_height))

        # Horizontal alignment: reward being directly above the pad.
        r_horizontal = np.exp(-0.4 * h_dist)

        components.distance = (0.5 * r_altitude + 0.5 * r_horizontal) * self.weights.distance

        # Speed: always penalise — must be slow to land safely.
        components.velocity = np.exp(-0.8 * vel_mag) * self.weights.velocity

        # Attitude
        components.upright = np.exp(-2.0 * tilt_rad) * self.weights.upright
        components.angular = np.exp(-0.5 * ang_mag)  * self.weights.angular

        # No time penalty below gate — let the agent take its time to land carefully.
        components.total = (
            components.distance
            + components.velocity
            + components.upright
            + components.angular
            + components.bonus
        )

        return components.total, components
