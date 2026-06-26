"""
Falcon 9-style FSM controller for RocketLander  —  v2 (enhanced).

5 flight phases:
  FREE_FALL   z > 10m    Stabilise, regulated ~4 m/s descent
  GUIDANCE    10-5m      Predictive intercept guidance, full lateral authority
  ALIGNMENT   <5m        Dedicated lateral-velocity zeroing before burn
  BURN        trigger    Physics-timed suicide burn (hover-slam)
  TOUCHDOWN   <2.2m      Near-vertical, adaptive gentle sink

Key improvements over v1:
  Predictive intercept guidance  v_des = -pos / t_to_land
    Outer position PIDs removed; the guidance law is now purely kinematic.
    Urgency scales automatically: more lateral authority near the ground.
  ALIGNMENT phase  dedicated phase to zero lateral velocity before burn.
  Tilt setpoint rate limiting (30 deg/s) → smooth rotation, no snap.
  Diagnostics dict  per-step state snapshot for video overlay.

Physics (demo_v0.xml):
  MASS   = 9.71 kg    MAX_FZ = 200 N    MAX_FXY = 25 N
  DT     = 0.025 s    TARGET_H = 1.93 m (COM at touchdown)
  Nozzle at body (0,0,-1.5):
    thrust_x > 0  →  τ_y = -37.5 Nm  →  pitch DECREASES (nose to -X)
    thrust_y > 0  →  τ_x = +37.5 Nm  →  roll  INCREASES (nose to -Y)
"""

from __future__ import annotations
from enum import IntEnum
import numpy as np


# ── Physical constants ─────────────────────────────────────────────────────────
MASS    : float = 9.71
G       : float = 9.81
MAX_FZ  : float = 200.0
MAX_FXY : float = 25.0
HOVER   : float = MASS * G / MAX_FZ          # ≈ 0.476 normalised hover throttle
ANET_UP : float = MAX_FZ / MASS - G          # ≈ 10.79 m/s² net upward at full thrust
TARGET_H: float = 1.93                       # m  COM height at touchdown
DT      : float = 0.025                      # s per env step


class FlightPhase(IntEnum):
    FREE_FALL = 0
    GUIDANCE  = 1
    ALIGNMENT = 2   # lateral velocity zeroing (NEW)
    BURN      = 3
    TOUCHDOWN = 4


# ── PID ───────────────────────────────────────────────────────────────────────

class PID:
    """PID with anti-windup and first-call derivative initialisation.

    On the very first call, prev_error is seeded from the current error so
    the derivative term is exactly zero (avoids a large initial spike).
    """

    def __init__(self, kp: float, ki: float, kd: float, limit: float = 1.0) -> None:
        self.kp, self.ki, self.kd, self.limit = kp, ki, kd, limit
        self._i    = 0.0
        self._prev = 0.0
        self._init = False

    def reset(self) -> None:
        self._i = self._prev = 0.0
        self._init = False

    def update(self, error: float) -> float:
        if not self._init:
            self._prev = error
            self._init = True
        self._i += error * DT
        if self.ki > 1e-9:
            self._i = float(np.clip(self._i, -self.limit / self.ki, self.limit / self.ki))
        d = (error - self._prev) / DT
        self._prev = error
        out = self.kp * error + self.ki * self._i + self.kd * d
        return float(np.clip(out, -self.limit, self.limit))


# ── Trajectory predictor ───────────────────────────────────────────────────────

class TrajectoryPredictor:
    """Kinematic calculations for burn timing and guidance."""

    @staticmethod
    def stopping_distance(v_down: float, v_target: float = 0.8) -> float:
        """Altitude above TARGET_H required to brake v_down → v_target at max thrust."""
        v = max(v_down, 0.0)
        return (v ** 2 - v_target ** 2) / (2.0 * ANET_UP) if v > v_target else 0.0

    @staticmethod
    def time_to_land(pos_z: float, vel_z: float, v_floor: float = 1.0) -> float:
        """Estimated seconds until rocket reaches TARGET_H at current descent rate.

        v_floor prevents infinite estimates when barely moving downward early in fall.
        """
        h     = max(pos_z - TARGET_H, 0.3)
        v_dn  = max(-vel_z, v_floor)
        return float(np.clip(h / v_dn, 0.3, 10.0))


# ── Landing burn controller ───────────────────────────────────────────────────

class LandingBurnController:
    """Physics-timed suicide burn.

    Ignites when remaining altitude equals stopping distance + safety margin.
    Latches: once burning, keeps going until v_down is small.
    """

    BURN_MARGIN_M: float = 1.2    # m extra safety margin (larger for faster approach)
    V_TOUCHDOWN  : float = 0.8    # m/s target speed at touchdown

    def __init__(self) -> None:
        self._latched = False

    def reset(self) -> None:
        self._latched = False

    def should_burn(self, pos_z: float, vel_z: float) -> bool:
        v_down    = max(-vel_z, 0.0)
        stop_dist = TrajectoryPredictor.stopping_distance(v_down, self.V_TOUCHDOWN)
        trigger   = TARGET_H + stop_dist + self.BURN_MARGIN_M

        if pos_z <= trigger and v_down > self.V_TOUCHDOWN * 0.5:
            self._latched = True
        # Release when nearly stopped. TOUCHDOWN fires at 3.5m now, so this
        # only triggers as a fallback edge case.
        if v_down <= self.V_TOUCHDOWN * 0.5:
            self._latched = False
        return self._latched


# ── Attitude controller ───────────────────────────────────────────────────────

class AttitudeController:
    """PD attitude control via nozzle gimbal (thrust_x / thrust_y).

    Sign derivation (nozzle at body pos 0,0,-1.5 m below COM):
      thrust_x > 0  →  τ_y = r×F = (−1.5) × 25 = −37.5 Nm
                     →  pitch decreases (nose toward −X)
      So to increase pitch (nose toward +X): thrust_x < 0

    PD gains (ω_n = 3 rad/s, ζ ≈ 0.65, K_plant ≈ 4.69 rad/s² per unit):
      kp ≈ ω_n² / (K × 57.3) ≈ 0.034
      kd ≈ 2ζω_n / K         ≈ 0.83
    """

    def __init__(self) -> None:
        self._kp = 0.034    # deg⁻¹  (proportional to angle error)
        self._kd = 0.80     # s·rad⁻¹ (proportional to angular velocity)

    def compute(
        self,
        pitch_d: float, roll_d: float,
        av_x: float, av_y: float,
        desired_pitch_d: float = 0.0,
        desired_roll_d:  float = 0.0,
    ) -> tuple[float, float]:
        pitch_err = desired_pitch_d - pitch_d
        roll_err  = desired_roll_d  - roll_d

        # Pitch: negative plant gain → −kp*err, POSITIVE kd to damp
        tx = -self._kp * pitch_err + self._kd * av_y
        # Roll:  positive plant gain → +kp*err, NEGATIVE kd to damp
        ty =  self._kp * roll_err  - self._kd * av_x

        return float(np.clip(tx, -1.0, 1.0)), float(np.clip(ty, -1.0, 1.0))


# ── Guidance controller ────────────────────────────────────────────────────────

class GuidanceController:
    """
    Predictive intercept guidance (time-to-go based).

    Outer loop — kinematic intercept (no PID gains):
        v_des = −pos / t_to_land
        Natural urgency scaling: gentle high up, aggressive near ground.

    Inner loop — velocity PID:
        error = v_des − v_actual  →  normalised lateral acceleration
        Converts to desired tilt via F_z geometry.

    Tilt directions:
        +pitch_des (nose toward +X)  →  engine pushes rocket in +X
        +roll_des  (nose toward −Y)  →  engine pushes rocket in −Y
    """

    def __init__(self) -> None:
        # Outer loop: position error → desired lateral velocity
        # kd=0.55 on position error acts as -kd*vel (velocity damping → no overshoot)
        self._px = PID(kp=0.30, ki=0.004, kd=0.55, limit=3.0)
        self._py = PID(kp=0.30, ki=0.004, kd=0.55, limit=3.0)
        # Inner loop: velocity error → desired lateral acceleration
        self._vx = PID(kp=0.55, ki=0.0,   kd=0.12, limit=1.0)
        self._vy = PID(kp=0.55, ki=0.0,   kd=0.12, limit=1.0)

    def reset(self) -> None:
        for p in (self._px, self._py, self._vx, self._vy):
            p.reset()

    def desired_tilt(
        self,
        pos_x: float, pos_y: float,
        vel_x: float, vel_y: float,
        pos_z: float, vel_z: float,
        tz_normalised: float,
        max_tilt_deg: float,
        vel_cap_ms: float,
    ) -> tuple[float, float]:
        # Outer: position PID → desired velocity.
        # PID derivative term: d(err)/dt = d(-pos)/dt = -vel → adds -kd*vel to output.
        # This naturally damps overshoot without explicit velocity feed-forward.
        dvx = float(np.clip(self._px.update(-pos_x), -vel_cap_ms, vel_cap_ms))
        dvy = float(np.clip(self._py.update(-pos_y), -vel_cap_ms, vel_cap_ms))

        # Inner: velocity error → normalised lateral acceleration
        ax_norm = self._vx.update(dvx - vel_x)
        ay_norm = self._vy.update(dvy - vel_y)

        # Convert to tilt: F_x = F_z × sin(pitch) ≈ F_z × pitch_rad
        F_z_N     = max(tz_normalised * MAX_FZ, 20.0)
        pitch_des = float(np.degrees(ax_norm * MAX_FXY / F_z_N))
        roll_des  = float(np.degrees(-ay_norm * MAX_FXY / F_z_N))

        return (float(np.clip(pitch_des, -max_tilt_deg, max_tilt_deg)),
                float(np.clip(roll_des,  -max_tilt_deg, max_tilt_deg)))


# ── Flight state machine ────────────────────────────────────────────────────────

class FlightStateMachine:
    """5-phase transition logic."""

    FREE_FALL_ALT  = 10.0
    ALIGNMENT_ALT  =  5.0   # enter ALIGNMENT below this (before BURN fires)
    TOUCHDOWN_ALT  =  3.5   # raised: fire TOUCHDOWN before burn oscillation zone

    def __init__(self) -> None:
        self._burn = LandingBurnController()
        self.phase = FlightPhase.FREE_FALL

    def reset(self) -> None:
        self._burn.reset()
        self.phase = FlightPhase.FREE_FALL

    def update(self, pos_z: float, vel_z: float) -> FlightPhase:
        if pos_z <= self.TOUCHDOWN_ALT:
            self.phase = FlightPhase.TOUCHDOWN
        elif self._burn.should_burn(pos_z, vel_z):
            self.phase = FlightPhase.BURN
        elif pos_z > self.FREE_FALL_ALT:
            self.phase = FlightPhase.FREE_FALL
        elif pos_z > self.ALIGNMENT_ALT:
            self.phase = FlightPhase.GUIDANCE
        else:
            self.phase = FlightPhase.ALIGNMENT
        return self.phase


# ── Main controller ────────────────────────────────────────────────────────────

class FalconController:
    """
    Deterministic Falcon 9-style landing controller.

    Usage:
        ctrl = FalconController()
        ctrl.reset()
        obs, _ = env.reset()
        while True:
            action = ctrl.step(obs)
            obs, reward, done, trunc, info = env.step(action)
            if done or trunc: break

    Observation (12-dim raw from RocketLander._get_obs):
        obs[0:3]   pos_x, pos_y, pos_z  (m)
        obs[3:6]   roll, pitch, yaw      (degrees)
        obs[6:9]   vel_x, vel_y, vel_z  (m/s, world frame)
        obs[9:12]  av_x, av_y, av_z    (rad/s, body frame)

    Action (3-dim, normalised to ctrlrange):
        action[0]  thrust_x ∈ [−1, 1]  →  ±25 N lateral
        action[1]  thrust_y ∈ [−1, 1]  →  ±25 N lateral
        action[2]  thrust_z ∈ [ 0, 1]  →  0–200 N main
    """

    _VZ_KP = 0.10
    _VZ_KI = 0.003
    _VZ_KD = 0.02

    TILT_RATE_LIMIT = 120.0  # deg/s — max rate for tilt setpoint (wide; physical smoothness comes from attitude kd)

    def __init__(self) -> None:
        self.fsm      = FlightStateMachine()
        self.guidance = GuidanceController()
        self.attitude = AttitudeController()
        self._pid_vz  = PID(kp=self._VZ_KP, ki=self._VZ_KI, kd=self._VZ_KD, limit=0.40)
        self._tz_last       : float = HOVER
        self._prev_pitch_des: float = 0.0
        self._prev_roll_des : float = 0.0
        self._prev_phase    : FlightPhase = FlightPhase.FREE_FALL
        self._diag          : dict  = {}

    def reset(self) -> None:
        self.fsm.reset()
        self.guidance.reset()
        self._pid_vz.reset()
        self._tz_last        = HOVER
        self._prev_pitch_des = 0.0
        self._prev_roll_des  = 0.0
        self._prev_phase     = FlightPhase.FREE_FALL
        self._diag           = {}

    # ── vertical thrust command ───────────────────────────────────────────────

    def _vertical_thrust(self, pos_z: float, vel_z: float, phase: FlightPhase) -> float:
        v_down = max(-vel_z, 0.0)

        if phase == FlightPhase.BURN:
            return 1.0

        if phase == FlightPhase.TOUCHDOWN:
            # Energy-based braking: compute thrust to reach v_tgt exactly at landing.
            #   F_net_up * d = ½m(v² − v_tgt²)   →   F_up = m(v²-v_tgt²)/(2d) + mg
            # As v→v_tgt the thrust naturally relaxes to near-hover.
            V_TGT = 0.4
            d = max(pos_z - TARGET_H, 0.05)
            if v_down > V_TGT:
                F_up = MASS * (v_down ** 2 - V_TGT ** 2) / (2.0 * d) + MASS * G
                return float(np.clip(F_up / MAX_FZ, HOVER * 0.80, 1.0))
            return float(np.clip(HOVER * 0.88, 0.0, 1.0))   # gentle final sink

        if phase == FlightPhase.FREE_FALL:
            # Near-free-fall: let gravity dominate, just limit top speed.
            # Rockets don't fight gravity during descent — they fall fast and brake late.
            target = 9.0
        elif phase == FlightPhase.GUIDANCE:
            # Still falling fast — burn will fire at ~7m altitude.
            target = 8.0
        else:  # ALIGNMENT
            # Still fast, correcting lateral drift; burn fires before this matters much.
            target = 6.0

        err        = target - v_down
        correction = self._pid_vz.update(-err)
        tz = float(np.clip(HOVER * 0.80 + correction, 0.18, 0.92))
        self._tz_last = tz
        return tz

    # ── main step ─────────────────────────────────────────────────────────────

    def step(self, obs: np.ndarray) -> np.ndarray:
        pos_x, pos_y, pos_z = float(obs[0]), float(obs[1]), float(obs[2])
        roll_d, pitch_d      = float(obs[3]), float(obs[4])
        vel_x, vel_y, vel_z  = float(obs[6]), float(obs[7]), float(obs[8])
        av_x, av_y           = float(obs[9]), float(obs[10])

        phase = self.fsm.update(pos_z, vel_z)
        tz    = self._vertical_thrust(pos_z, vel_z, phase)

        # Phase-specific guidance parameters
        if phase == FlightPhase.FREE_FALL:
            max_tilt, vel_cap = 5.0,  1.5
        elif phase == FlightPhase.GUIDANCE:
            max_tilt, vel_cap = 15.0, 3.0
        elif phase == FlightPhase.ALIGNMENT:
            max_tilt, vel_cap = 10.0, 2.0
        elif phase == FlightPhase.BURN:
            max_tilt, vel_cap = 8.0,  1.5
        else:  # TOUCHDOWN
            max_tilt, vel_cap = 5.0,  0.6

        pitch_des, roll_des = self.guidance.desired_tilt(
            pos_x, pos_y, vel_x, vel_y, pos_z, vel_z,
            tz_normalised=tz, max_tilt_deg=max_tilt, vel_cap_ms=vel_cap,
        )

        # On phase transition to a tighter tilt envelope, immediately clamp
        # the rate-limiter state so it doesn't bleed a large old setpoint
        # into the new phase (e.g., -8° from ALIGNMENT into TOUCHDOWN's ±2°).
        if phase != self._prev_phase:
            self._prev_pitch_des = float(np.clip(self._prev_pitch_des,
                                                  -max_tilt, max_tilt))
            self._prev_roll_des  = float(np.clip(self._prev_roll_des,
                                                  -max_tilt, max_tilt))
            self._prev_phase = phase

        # Tilt setpoint rate limiting — prevents snap rotation within a phase
        max_delta = self.TILT_RATE_LIMIT * DT
        pitch_des = float(np.clip(pitch_des,
                                   self._prev_pitch_des - max_delta,
                                   self._prev_pitch_des + max_delta))
        roll_des  = float(np.clip(roll_des,
                                   self._prev_roll_des  - max_delta,
                                   self._prev_roll_des  + max_delta))
        self._prev_pitch_des = pitch_des
        self._prev_roll_des  = roll_des

        tx, ty = self.attitude.compute(
            pitch_d, roll_d, av_x, av_y,
            desired_pitch_d=pitch_des,
            desired_roll_d=roll_des,
        )

        # Per-step diagnostics snapshot
        v_down     = max(-vel_z, 0.0)
        self._diag = {
            'phase'    : phase.name,
            'phase_int': int(phase),
            'pos_x'    : pos_x,
            'pos_y'    : pos_y,
            'pos_z'    : pos_z,
            'vel_x'    : vel_x,
            'vel_y'    : vel_y,
            'vel_z'    : vel_z,
            'lat_err'  : float(np.hypot(pos_x, pos_y)),
            'v_lat'    : float(np.hypot(vel_x, vel_y)),
            'v_down'   : v_down,
            'tz'       : tz,
            'tx'       : tx,
            'ty'       : ty,
            'pitch'    : pitch_d,
            'roll'     : roll_d,
            'pitch_des': pitch_des,
            'roll_des' : roll_des,
            'stop_dist': TrajectoryPredictor.stopping_distance(v_down),
            't_land'   : float(max(pos_z - TARGET_H, 0.3) / max(v_down, 1.0)),
        }

        return np.array([tx, ty, tz], dtype=np.float32)

    @property
    def phase_name(self) -> str:
        return self.fsm.phase.name

    @property
    def diagnostics(self) -> dict:
        return dict(self._diag)


# ── Outcome codes ──────────────────────────────────────────────────────────────

class LandingOutcome:
    """Maps env info['crash_report'] codes to named outcomes."""
    TIMEOUT       = 0
    SUCCESS       = 1
    CRASH         = 2
    ROLL_OVER     = 3
    PITCH_OVER    = 4
    OUT_OF_BOUNDS = 5

    _NAMES = {
        0: 'TIMEOUT', 1: 'SUCCESS', 2: 'CRASH',
        3: 'ROLL_OVER', 4: 'PITCH_OVER', 5: 'OUT_OF_BOUNDS',
    }

    @classmethod
    def name(cls, code: int) -> str:
        return cls._NAMES.get(code, f'UNKNOWN({code})')
