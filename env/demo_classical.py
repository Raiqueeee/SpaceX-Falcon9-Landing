"""
Render demo videos using the deterministic Falcon 9-style classical controller.

No training checkpoint required — runs immediately.

Usage:
    python env/demo_classical.py
    python env/demo_classical.py --height 15 --attempts 5
    python env/demo_classical.py --scenario ENGINE_RELIGHT
    python env/demo_classical.py --scenario ROUGH_SEAS --attempts 3
    python env/demo_classical.py --output-dir videos/classical --resolution 1080
    python env/demo_classical.py --no-overlay

Scenarios (--scenario flag):
    HYDRAULIC_EXHAUSTION   Grid fins degrade after 60% of descent
    ENGINE_RELIGHT         Landing burn ignites 0.4 s late
    HYPERSONIC_TUMBLE      Starts at 30 deg tilt with angular velocity
    ROUGH_SEAS             Pad moves +/-2 m at 0.3 Hz wave frequency
    CROSSWIND              Constant 8 m/s side wind throughout descent
    SENSOR_DRIFT           GPS drifts 1.5 m over descent
    LATE_SEPARATION        Enters with 15 m/s horizontal velocity
    PARTIAL_THRUSTER       Lateral thruster X stuck at +30% output
"""

import argparse
import os
import sys

import imageio
import mujoco
import numpy as np

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
os.chdir(ROOT_DIR)

from env.rocket_landing import RocketLander
from env.classical_controller import FalconController, FlightPhase, TARGET_H, DT, LandingOutcome

DEMO_XML = os.path.join(os.path.dirname(__file__), "xml_files", "demo_v0.xml")

try:
    from PIL import Image, ImageDraw, ImageFont
    _PIL = True
except ImportError:
    _PIL = False


# ── TrajectoryVisualizer ───────────────────────────────────────────────────────

class TrajectoryVisualizer:
    """HUD overlay on rendered video frames, with optional scenario warning strip."""

    _PHASE_COLORS = {
        'FREE_FALL' : (80,  160, 255),
        'GUIDANCE'  : (60,  210, 80),
        'ALIGNMENT' : (255, 200, 50),
        'BURN'      : (255, 80,  20),
        'TOUCHDOWN' : (200, 220, 255),
    }
    _BAR_BG   = (40,  40,  40)
    _TEXT     = (230, 230, 230)
    _DIM_TEXT = (140, 140, 140)

    def __init__(self) -> None:
        self._font_lg = None
        self._font_sm = None
        self._fonts_loaded = False

    def _ensure_fonts(self, h: int) -> None:
        if self._fonts_loaded:
            return
        sz_lg = max(14, h // 40)
        sz_sm = max(10, h // 56)
        candidates = [
            "C:/Windows/Fonts/arial.ttf",
            "C:/Windows/Fonts/segoeui.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/System/Library/Fonts/Helvetica.ttc",
        ]
        font_lg = font_sm = None
        for path in candidates:
            if os.path.exists(path):
                try:
                    from PIL import ImageFont as _IF
                    font_lg = _IF.truetype(path, sz_lg)
                    font_sm = _IF.truetype(path, sz_sm)
                    break
                except Exception:
                    pass
        if font_lg is None:
            from PIL import ImageFont as _IF
            font_lg = _IF.load_default()
            font_sm = _IF.load_default()
        self._font_lg = font_lg
        self._font_sm = font_sm
        self._fonts_loaded = True

    def overlay(self, frame: np.ndarray, diag: dict) -> np.ndarray:
        if not _PIL or not diag:
            return frame

        img  = Image.fromarray(frame)
        draw = ImageDraw.Draw(img, 'RGBA')
        h, w = frame.shape[:2]
        self._ensure_fonts(h)
        pad = max(6, h // 120)

        phase  = diag.get('phase', 'UNKNOWN')
        color  = self._PHASE_COLORS.get(phase, (200, 200, 200))
        pos_z  = diag.get('pos_z',  0.0)
        vel_z  = diag.get('vel_z',  0.0)
        v_down = diag.get('v_down', 0.0)
        tz     = diag.get('tz',     0.0)
        lat_e  = diag.get('lat_err', 0.0)
        v_lat  = diag.get('v_lat',   0.0)
        pitch  = diag.get('pitch',   0.0)
        roll   = diag.get('roll',    0.0)
        pdest  = diag.get('pitch_des', 0.0)
        rdest  = diag.get('roll_des',  0.0)
        sdist  = diag.get('stop_dist', 0.0)
        tland  = diag.get('t_land',    0.0)

        bar_h = h // 60

        # ── Top banner ────────────────────────────────────────────────────────
        banner_h = h // 18
        draw.rectangle([(0, 0), (w, banner_h)], fill=(0, 0, 0, 190))

        draw.rectangle(
            [(w // 2 - w // 10, 0), (w // 2 + w // 10, banner_h)],
            fill=(*color, 210),
        )
        draw.text((w // 2, banner_h // 2), f"  {phase}  ",
                  fill=(0, 0, 0), font=self._font_lg, anchor='mm')

        draw.text(
            (pad * 2, banner_h // 2),
            f"ALT {pos_z:5.1f} m    VZ {vel_z:+5.1f} m/s",
            fill=self._TEXT, font=self._font_lg, anchor='lm',
        )
        draw.text(
            (w - pad * 2, banner_h // 2),
            f"T-LAND {tland:.1f} s",
            fill=self._DIM_TEXT, font=self._font_lg, anchor='rm',
        )

        # ── Scenario warning strip (below banner, only when active) ───────────
        scenario = diag.get('scenario')
        strip_h  = 0
        if scenario:
            strip_h = h // 28
            sev     = float(diag.get('disturbance_severity', 0.0))
            status  = diag.get('disturbance_status', '')
            slabel  = diag.get('scenario_label', scenario)
            # Severity: green(0) → amber(0.5) → red(1)
            sr = int(min(255, 80  + 175 * sev))
            sg = int(max(30,  200 - 170 * sev))
            sb = 30
            draw.rectangle([(0, banner_h), (w, banner_h + strip_h)],
                           fill=(15, 15, 15, 215))
            draw.text(
                (pad, banner_h + strip_h // 2),
                f"[SCENARIO] {slabel}   |   {status}",
                fill=(sr, sg, sb), font=self._font_sm, anchor='lm',
            )

        # ── Right-side metrics panel ──────────────────────────────────────────
        panel_w = w // 5
        px = w - panel_w - pad
        py = banner_h + strip_h + pad * 3

        def metric(label, value, unit='', y_off=0):
            draw.text((px, py + y_off), label,
                      fill=self._DIM_TEXT, font=self._font_sm, anchor='la')
            draw.text((px + panel_w - pad, py + y_off), f"{value}{unit}",
                      fill=self._TEXT, font=self._font_sm, anchor='ra')

        row = h // 32
        metric("PITCH",      f"{pitch:+6.1f}", "°",    row * 0)
        metric("ROLL",       f"{roll:+6.1f}",  "°",    row * 1)
        metric("PITCH CMD",  f"{pdest:+6.1f}", "°",    row * 2)
        metric("ROLL CMD",   f"{rdest:+6.1f}", "°",    row * 3)
        metric("LAT ERR",    f"{lat_e:6.2f}",  " m",   row * 5)
        metric("V_LAT",      f"{v_lat:6.2f}",  " m/s", row * 6)
        metric("STOP DIST",  f"{sdist:6.2f}",  " m",   row * 8)

        # ── Left-side bar gauges ──────────────────────────────────────────────
        bx     = pad * 2
        bar_w  = w // 6
        by_thr = h - pad * 2 - bar_h * 7
        label_off = bar_h + pad

        def bar(x, y, val, vmax, clr, label):
            frac = float(np.clip(val / max(vmax, 1e-9), 0.0, 1.0))
            draw.rectangle([(x, y), (x + bar_w, y + bar_h)],
                           fill=(*self._BAR_BG, 200))
            if frac > 0:
                draw.rectangle([(x, y), (x + int(bar_w * frac), y + bar_h)],
                               fill=(*clr, 220))
            draw.text((x, y - pad), label, fill=self._DIM_TEXT, font=self._font_sm)
            draw.text((x + bar_w + pad, y + bar_h // 2),
                      f"{val:.2f}", fill=self._TEXT, font=self._font_sm, anchor='lm')

        bar(bx, by_thr,                  tz,     1.0, (255, 100,  30), "THROTTLE")
        bar(bx, by_thr + label_off * 2,  lat_e,  5.0, (80,  180, 255), "LAT ERR (m)")
        bar(bx, by_thr + label_off * 4,  v_lat,  3.0, (200, 100, 255), "V_LAT (m/s)")
        bar(bx, by_thr + label_off * 6,  v_down, 5.0, (100, 255, 200), "V_DOWN (m/s)")

        # ── Burn ring ─────────────────────────────────────────────────────────
        if phase == 'BURN':
            r = h // 20
            cx, cy = w - r - pad * 4, h - r - pad * 4
            draw.ellipse([(cx - r, cy - r), (cx + r, cy + r)],
                         outline=(255, 100, 0, 240), width=max(2, r // 4))
            draw.text((cx, cy), "BURN", fill=(255, 120, 0),
                      font=self._font_sm, anchor='mm')

        return np.array(img)


# ── ScenarioDisturbance ────────────────────────────────────────────────────────

class ScenarioDisturbance:
    """
    Injects realistic disturbances inspired by actual Falcon 9 failure modes.

    All disturbances are applied externally (obs/action/force injection) so the
    classical controller code remains unchanged — the controller doesn't know
    it's being challenged.
    """

    SCENARIO_META: dict = {
        'HYDRAULIC_EXHAUSTION': {
            'label': 'Hydraulic Fluid Exhaustion',
            'desc' : 'Grid fins lose effectiveness after 60% of descent',
            'height': 15.0,
        },
        'ENGINE_RELIGHT': {
            'label': 'Engine Relight Failure',
            'desc' : 'Landing burn ignites 0.4 s late — arrives too fast',
            'height': 20.0,
        },
        'HYPERSONIC_TUMBLE': {
            'label': 'Hypersonic Reentry Tumble',
            'desc' : 'Starts at 30 deg pitch with 1.5 rad/s spin',
            'height': 20.0,
        },
        'ROUGH_SEAS': {
            'label': 'Rough Sea State',
            'desc' : 'Landing pad drifts +/-2 m at 0.3 Hz wave frequency',
            'height': 15.0,
        },
        'CROSSWIND': {
            'label': 'Strong Crosswind',
            'desc' : 'Constant 8 m/s lateral wind throughout descent',
            'height': 15.0,
        },
        'SENSOR_DRIFT': {
            'label': 'GPS Sensor Drift',
            'desc' : 'Position reading drifts 1.5 m over entire descent',
            'height': 15.0,
        },
        'LATE_SEPARATION': {
            'label': 'Late Stage Separation',
            'desc' : 'Enters guidance with 15 m/s horizontal velocity',
            'height': 20.0,
        },
        'PARTIAL_THRUSTER': {
            'label': 'Partial Thruster Failure',
            'desc' : 'Lateral thruster X stuck at +30% — cannot push left',
            'height': 15.0,
        },
    }

    # 0.5 s delay at DT=0.025 s/step
    _RELIGHT_DELAY = 20

    # Crosswind drag force (N) in +X world frame  (8 m/s wind, scaled to model size)
    _WIND_FORCE_N = 3.0

    def __init__(self, scenario: str, start_height: float) -> None:
        if scenario not in self.SCENARIO_META:
            valid = ', '.join(self.SCENARIO_META)
            raise ValueError(f"Unknown scenario '{scenario}'. Valid: {valid}")
        self.scenario     = scenario
        self.start_height = start_height
        self._step        = 0
        self._rocket_bid  = None   # MuJoCo body index, resolved on first use
        # ENGINE_RELIGHT: track when BURN phase was first entered
        self._burn_start  : int | None = None
        # ROUGH_SEAS: current pad position (used by augment_diag)
        self._pad_x = 0.0
        self._pad_y = 0.0

    def reset(self) -> None:
        self._step       = 0
        self._burn_start = None
        self._pad_x      = 0.0
        self._pad_y      = 0.0

    # ── internal helpers ──────────────────────────────────────────────────────

    def _get_rocket_body(self, env: RocketLander) -> int:
        if self._rocket_bid is None:
            try:
                self._rocket_bid = env.model.body('rocket').id
            except Exception:
                self._rocket_bid = 1
        return self._rocket_bid

    def _descent_frac(self, pos_z: float) -> float:
        """0.0 at start_height, 1.0 at TARGET_H."""
        return float(np.clip(
            1.0 - (pos_z - TARGET_H) / max(self.start_height - TARGET_H, 1.0),
            0.0, 1.0,
        ))

    # ── public API ────────────────────────────────────────────────────────────

    def apply_initial_conditions(self, env: RocketLander) -> None:
        """Called once, immediately after env.reset()."""
        if self.scenario == 'HYPERSONIC_TUMBLE':
            # 30 deg pitch tilt: rotation about Y, q = [cos(15°), 0, sin(15°), 0]
            a = np.radians(30.0)
            env.data.qpos[3] = np.cos(a / 2)   # qw
            env.data.qpos[4] = 0.0               # qx
            env.data.qpos[5] = np.sin(a / 2)    # qy  (pitch)
            env.data.qpos[6] = 0.0               # qz
            # Angular velocity: pitch rate + slight roll to create tumble
            env.data.qvel[3] =  0.8   # roll rate  (rad/s, world frame)
            env.data.qvel[4] =  1.5   # pitch rate
            mujoco.mj_forward(env.model, env.data)

        elif self.scenario == 'LATE_SEPARATION':
            env.data.qvel[0] = 15.0   # vx = 15 m/s horizontal
            mujoco.mj_forward(env.model, env.data)

    def apply_forces(self, env: RocketLander) -> None:
        """Apply external forces BEFORE each env.step()."""
        bid = self._get_rocket_body(env)
        # Clear all applied forces first
        env.data.xfrc_applied[:] = 0.0
        if self.scenario == 'CROSSWIND':
            env.data.xfrc_applied[bid, 0] = self._WIND_FORCE_N

    def disturb_obs(self, obs: np.ndarray, step: int) -> np.ndarray:
        """Return (possibly modified) observation for the controller to see."""
        obs = obs.copy()
        t   = step * DT

        if self.scenario == 'ROUGH_SEAS':
            self._pad_x = 2.0 * np.sin(2.0 * np.pi * 0.3 * t)
            self._pad_y = 2.0 * np.cos(2.0 * np.pi * 0.3 * t)
            # Controller sees position relative to the moving pad
            obs[0] -= self._pad_x
            obs[1] -= self._pad_y

        elif self.scenario == 'SENSOR_DRIFT':
            frac   = self._descent_frac(float(obs[2]))
            obs[0] += 1.5 * frac          # drifting X reading
            obs[1] += 0.6 * frac          # smaller Y drift

        return obs

    def disturb_action(
        self,
        action    : np.ndarray,
        real_obs  : np.ndarray,
        phase_name: str,
        step      : int,
    ) -> np.ndarray:
        """Return (possibly modified) action to execute in env."""
        action = action.copy()
        frac   = self._descent_frac(float(real_obs[2]))

        if self.scenario == 'HYDRAULIC_EXHAUSTION':
            if frac > 0.6:
                remaining    = (frac - 0.6) / 0.4   # 0→1 from 60% to 100%
                effectiveness = max(0.0, 1.0 - remaining)
                noise = np.random.normal(0.0, 0.08 * remaining)
                action[0] = action[0] * effectiveness + noise
                action[1] = action[1] * effectiveness + noise

        elif self.scenario == 'ENGINE_RELIGHT':
            if phase_name == 'BURN':
                if self._burn_start is None:
                    self._burn_start = step
                elapsed = step - self._burn_start
                if elapsed < self._RELIGHT_DELAY:
                    action[2] = 0.18   # engine not yet lit — near-idle thrust

        elif self.scenario == 'PARTIAL_THRUSTER':
            # Thruster X can't produce net leftward force — jammed at +30%
            if action[0] < 0.30:
                action[0] = 0.30

        return np.clip(action, [-1.0, -1.0, 0.0], [1.0, 1.0, 1.0])

    def augment_diag(self, diag: dict, step: int) -> dict:
        """Inject scenario metadata into the diagnostics dict for HUD rendering."""
        diag  = dict(diag)
        meta  = self.SCENARIO_META[self.scenario]
        frac  = self._descent_frac(float(diag.get('pos_z', self.start_height)))

        diag['scenario']       = self.scenario
        diag['scenario_label'] = meta['label']

        sev    = 0.0
        status = 'NOMINAL'

        if self.scenario == 'HYDRAULIC_EXHAUSTION':
            if frac > 0.6:
                remaining = (frac - 0.6) / 0.4
                eff_pct   = max(0.0, 1.0 - remaining) * 100.0
                sev       = remaining
                status    = f"CTRL EFF: {eff_pct:.0f}%  {'!' * min(5, int(remaining * 5 + 1))}"
            else:
                status = f"CTRL EFF: 100%  (degrades at {60:.0f}%)"

        elif self.scenario == 'ENGINE_RELIGHT':
            if self._burn_start is not None:
                elapsed = step - self._burn_start
                if elapsed < self._RELIGHT_DELAY:
                    rem_s  = (self._RELIGHT_DELAY - elapsed) * DT
                    sev    = 1.0
                    status = f"ENGINE SUPPRESSED — RELIGHT IN {rem_s:.2f}s"
                else:
                    sev    = 0.2
                    status = "ENGINE LIT (late)"
            else:
                status = "AWAITING BURN TRIGGER"

        elif self.scenario == 'ROUGH_SEAS':
            sev    = float(np.hypot(self._pad_x, self._pad_y) / 2.83)  # max=2.83m
            status = f"PAD ({self._pad_x:+.1f}, {self._pad_y:+.1f}) m"

        elif self.scenario == 'CROSSWIND':
            sev    = 0.55
            status = f"WIND {self._WIND_FORCE_N:.1f}N -> +X  (8 m/s)"

        elif self.scenario == 'SENSOR_DRIFT':
            drift  = 1.5 * frac
            sev    = frac
            status = f"GPS DRIFT: {drift:.2f} m"

        elif self.scenario == 'LATE_SEPARATION':
            vx     = float(diag.get('vel_x', 0.0))
            sev    = min(1.0, abs(vx) / 15.0)
            status = f"HORIZ VX: {vx:+.1f} m/s"

        elif self.scenario == 'PARTIAL_THRUSTER':
            sev    = 0.4
            status = "TX MIN LOCKED +30%"

        elif self.scenario == 'HYPERSONIC_TUMBLE':
            tilt   = float(np.hypot(diag.get('pitch', 0.0), diag.get('roll', 0.0)))
            sev    = min(1.0, tilt / 30.0)
            status = f"TILT: {tilt:.1f} deg"

        diag['disturbance_status']   = status
        diag['disturbance_severity'] = float(sev)
        return diag

    def tick(self) -> None:
        self._step += 1

    @property
    def label(self) -> str:
        return self.SCENARIO_META[self.scenario]['label']

    @property
    def description(self) -> str:
        return self.SCENARIO_META[self.scenario]['desc']

    @property
    def recommended_height(self) -> float:
        return float(self.SCENARIO_META[self.scenario]['height'])


# ── Outcome classification ─────────────────────────────────────────────────────

def classify_outcome(crash_report: int, approach_vel: float) -> str:
    """Map env crash_report code to a named outcome.

    The env's crash_report is authoritative for SUCCESS/CRASH.
    approach_vel (velocity at first entry into the pad zone) is displayed
    as context but does not override the code — energy braking continues
    well past the measurement point.
    """
    if crash_report == LandingOutcome.SUCCESS:
        return 'SUCCESS'
    elif crash_report == LandingOutcome.CRASH:
        return 'HARD_LANDING'
    elif crash_report in (LandingOutcome.ROLL_OVER, LandingOutcome.PITCH_OVER):
        return 'TIPOVER'
    elif crash_report == LandingOutcome.OUT_OF_BOUNDS:
        return 'MISSED_PAD'
    else:   # TIMEOUT or unknown
        return 'FUEL_EXHAUSTED'


# ── Environment helpers ────────────────────────────────────────────────────────

def make_env(rocket_design: str = "demo", height: float = 15.0) -> RocketLander:
    env = RocketLander(
        rocket_design=rocket_design,
        render_mode="rgb_array",
        width=64,
        height=64,
    )
    env.set_curriculum_height(height)
    return env


# ── Trajectory collection — normal (no disturbance) ───────────────────────────

def run_episode(
    env: RocketLander,
    ctrl: FalconController,
    max_steps: int = 600,
    linger_steps: int = 240,
    verbose: bool = False,
) -> tuple[list, float, bool, str]:
    ctrl.reset()
    obs, _ = env.reset()
    trajectory: list = []
    landed    = False
    final_alt = float(obs[2])
    prev_phase: str  = ""
    last_diag : dict = {}

    for step in range(max_steps):
        action    = ctrl.step(obs)
        last_diag = ctrl.diagnostics

        obs, _reward, done, trunc, info = env.step(action)

        qpos = env.data.qpos.copy()
        qvel = env.data.qvel.copy()
        trajectory.append((qpos, qvel, last_diag))
        final_alt = float(qpos[2])

        phase = ctrl.phase_name
        if phase != prev_phase:
            if verbose:
                print(f"    step {step:4d}  z={final_alt:5.2f}m  vz={obs[8]:+.2f}  -> {phase}")
            prev_phase = phase

        if done or trunc:
            crash_report = info.get("crash_report", 0)
            landed       = crash_report == 1
            reason_map   = {0: "truncated", 1: "LANDED", 2: "crash",
                            3: "roll over", 4: "pitch over", 5: "out of bounds"}
            reason       = reason_map.get(crash_report, f"code {crash_report}")
            for _ in range(linger_steps):
                mujoco.mj_step(env.model, env.data)
                trajectory.append((env.data.qpos.copy(), env.data.qvel.copy(), last_diag))
            break
    else:
        reason = "timeout"

    return trajectory, final_alt, landed, reason


# ── Trajectory collection — scenario (with disturbance) ───────────────────────

def run_scenario_episode(
    env         : RocketLander,
    ctrl        : FalconController,
    dist        : ScenarioDisturbance,
    max_steps   : int = 800,
    linger_steps: int = 300,
    verbose     : bool = False,
) -> tuple[list, int, float]:
    """Run one disturbed episode.

    Returns:
        trajectory   — list of (qpos, qvel, augmented_diag)
        crash_report — raw env code (0-5)
        impact_vel   — velocity magnitude at first near-pad contact (m/s)
    """
    ctrl.reset()
    dist.reset()
    obs, _ = env.reset()

    # Apply scenario-specific initial conditions (tilt, velocity, etc.)
    dist.apply_initial_conditions(env)
    obs = env._get_obs()

    trajectory  : list      = []
    crash_report: int       = 0
    impact_vel  : float | None = None
    last_diag   : dict      = {}
    prev_phase  : str       = ""

    for step in range(max_steps):
        # External forces (wind) must be set before env.step() calls mj_step
        dist.apply_forces(env)

        # Build observation the controller will see (may be distorted)
        obs_ctrl = dist.disturb_obs(obs, step)

        # Controller computes its ideal action
        action = ctrl.step(obs_ctrl)

        # Merge scenario metadata into diagnostics for HUD
        last_diag = dist.augment_diag(ctrl.diagnostics, step)

        # Apply scenario-level action disturbances
        action = dist.disturb_action(action, obs, ctrl.phase_name, step)

        # Record velocity at first near-pad contact
        if obs[2] < TARGET_H + 0.5 and impact_vel is None:
            impact_vel = float(np.linalg.norm(obs[6:9]))

        if verbose and ctrl.phase_name != prev_phase:
            d = ctrl.diagnostics
            print(f"    step {step:4d}  z={d['pos_z']:5.2f}m  vz={d['vel_z']:+.2f}  -> {ctrl.phase_name}")
            prev_phase = ctrl.phase_name

        obs, _reward, done, trunc, info = env.step(action)
        trajectory.append((env.data.qpos.copy(), env.data.qvel.copy(), last_diag))

        dist.tick()

        if done or trunc:
            crash_report = info.get("crash_report", 0)
            for _ in range(linger_steps):
                mujoco.mj_step(env.model, env.data)
                trajectory.append((env.data.qpos.copy(), env.data.qvel.copy(), last_diag))
            break

    if impact_vel is None:
        impact_vel = float(np.linalg.norm(obs[6:9]))

    return trajectory, crash_report, impact_vel


# ── Multi-attempt best-of (normal mode) ───────────────────────────────────────

def best_of(
    env: RocketLander,
    ctrl: FalconController,
    attempts: int = 3,
    max_steps: int = 600,
    linger_steps: int = 240,
    verbose: bool = False,
) -> list:
    best_traj, best_alt, best_landed = None, float("inf"), False

    for i in range(attempts):
        traj, alt, landed, reason = run_episode(env, ctrl, max_steps, linger_steps, verbose)
        tag = "LANDED" if landed else f"alt={alt:.2f}m ({reason})"
        print(f"  Attempt {i + 1}/{attempts}: {tag}")

        if best_traj is None:
            best_traj, best_alt, best_landed = traj, alt, landed
        elif landed and not best_landed:
            best_traj, best_alt, best_landed = traj, alt, landed
        elif not landed and not best_landed and alt < best_alt:
            best_traj, best_alt, best_landed = traj, alt, landed

    print(f"  Best: {'LANDED' if best_landed else f'alt={best_alt:.2f}m'}")
    return best_traj


# ── Rendering ─────────────────────────────────────────────────────────────────

def make_camera(lookat, distance, azimuth, elevation):
    cam           = mujoco.MjvCamera()
    cam.type      = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = lookat
    cam.distance  = distance
    cam.azimuth   = azimuth
    cam.elevation = elevation
    return cam


def render_trajectory(
    trajectory     : list,
    resolution     : int,
    fps            : int,
    output_dir     : str,
    overlay        : bool  = True,
    camera_distance: float = 80.0,
    prefix         : str   = "classical",
) -> None:
    model    = mujoco.MjModel.from_xml_path(DEMO_XML)
    data     = mujoco.MjData(model)
    width    = int(resolution * 16 / 9)
    renderer = mujoco.Renderer(model, height=resolution, width=width)
    viz      = TrajectoryVisualizer() if (overlay and _PIL) else None

    if overlay and not _PIL:
        print("  [note] Pillow not installed — rendering without HUD overlay")

    aerial_frames   = []
    tracking_frames = []

    for item in trajectory:
        qpos, qvel = item[0], item[1]
        diag       = item[2] if len(item) > 2 else {}

        data.qpos[:] = qpos
        data.qvel[:] = qvel
        mujoco.mj_forward(model, data)

        rocket_pos = qpos[:3]
        lookat_z   = rocket_pos[2] * 0.4

        cam = make_camera([0.0, 0.0, lookat_z], camera_distance, 135, -25)
        renderer.update_scene(data, camera=cam)
        raw = renderer.render().copy()
        aerial_frames.append(viz.overlay(raw, diag) if viz else raw)

        cam = make_camera(rocket_pos.copy(), 25, 135, -30)
        renderer.update_scene(data, camera=cam)
        raw = renderer.render().copy()
        tracking_frames.append(viz.overlay(raw, diag) if viz else raw)

    renderer.close()
    os.makedirs(output_dir, exist_ok=True)
    _save(aerial_frames,   os.path.join(output_dir, f"{prefix}_aerial.mp4"),   fps)
    _save(tracking_frames, os.path.join(output_dir, f"{prefix}_tracking.mp4"), fps)


def _save(frames: list, path: str, fps: int) -> None:
    if not frames:
        print(f"  No frames for {path}")
        return
    writer = imageio.get_writer(path, fps=fps, quality=8, codec="libx264",
                                macro_block_size=1)
    for frame in frames:
        writer.append_data(frame)
    writer.close()
    mb = os.path.getsize(path) / (1024 * 1024)
    print(f"  Saved: {path}  ({len(frames)} frames, {mb:.1f} MB)")


# ── Scenario report ────────────────────────────────────────────────────────────

_OUTCOME_SYMBOL = {
    'SUCCESS'       : ('OK', '\033[92m'),
    'HARD_LANDING'  : ('!!', '\033[93m'),
    'TIPOVER'       : ('XX', '\033[91m'),
    'MISSED_PAD'    : ('XX', '\033[91m'),
    'FUEL_EXHAUSTED': ('--', '\033[95m'),
}


def print_scenario_report(
    scenario    : str,
    outcome     : str,
    crash_report: int,
    impact_vel  : float,
    n_steps     : int,
) -> None:
    meta   = ScenarioDisturbance.SCENARIO_META[scenario]
    sym, clr = _OUTCOME_SYMBOL.get(outcome, ('??', ''))
    RESET  = '\033[0m'

    print(f"\n{'=' * 58}")
    print(f"  FALCON 9 FAILURE SCENARIO REPORT")
    print(f"{'=' * 58}")
    print(f"  Scenario  : {meta['label']}")
    print(f"  Condition : {meta['desc']}")
    print(f"  Duration  : {n_steps * DT:.1f} s  ({n_steps} steps)")
    print(f"  Approach v: {impact_vel:.2f} m/s  (at pad zone entry, z < 2.4 m)")
    print(f"  Env code  : {LandingOutcome.name(crash_report)}")
    print(f"\n  OUTCOME   : {clr}[{sym}] {outcome}{RESET}")
    print(f"{'=' * 58}\n")


# ── Diagnostics ───────────────────────────────────────────────────────────────

def print_episode_summary(env: RocketLander, ctrl: FalconController) -> None:
    print("\n-- Single diagnostic run -------------------------------------------")
    ctrl.reset()
    obs, _    = env.reset()
    prev_phase = None

    for step in range(600):
        action = ctrl.step(obs)
        phase  = ctrl.phase_name

        if phase != prev_phase:
            d = ctrl.diagnostics
            print(f"  step {step:4d}  z={d['pos_z']:6.2f}m  vz={d['vel_z']:+.2f}m/s"
                  f"  lat={d['lat_err']:.2f}m  t_land={d['t_land']:.2f}s  -> {phase}")
            prev_phase = phase

        obs, _, done, trunc, info = env.step(action)
        if done or trunc:
            code = info.get("crash_report", 0)
            vz   = float(obs[8])
            vxy  = float(np.linalg.norm(obs[6:8]))
            z    = float(obs[2])
            print(f"\n  Result: {'SUCCESS' if code == 1 else 'FAIL'}  "
                  f"code={code}  z={z:.3f}m  vz={vz:+.2f}  v_lat={vxy:.2f}")
            break

    print("--------------------------------------------------------------------\n")


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Classical Falcon 9 controller demo")
    parser.add_argument("--height",        type=float, default=None,
                        help="Starting height (m). Defaults to scenario minimum or 15 m.")
    parser.add_argument("--attempts",      type=int,   default=5,
                        help="Rollouts; best rendered (ignored in scenario mode)")
    parser.add_argument("--scenario",      type=str,   default=None,
                        choices=list(ScenarioDisturbance.SCENARIO_META),
                        help="Failure scenario to simulate")
    parser.add_argument("--max-steps",     type=int,   default=800)
    parser.add_argument("--linger",        type=int,   default=300,
                        help="Extra frames after episode ends")
    parser.add_argument("--resolution",    type=int,   default=1080)
    parser.add_argument("--fps",           type=int,   default=24)
    parser.add_argument("--output-dir",    type=str,   default=None)
    parser.add_argument("--rocket-design", type=str,   default="demo")
    parser.add_argument("--diagnose",      action="store_true",
                        help="Verbose phase diagnostics")
    parser.add_argument("--no-overlay",    action="store_true",
                        help="Disable HUD overlay")
    parser.add_argument("--seed",          type=int,   default=None)
    args = parser.parse_args()

    if args.seed is not None:
        np.random.seed(args.seed)

    # Resolve height: explicit arg > scenario default > 15 m
    if args.scenario:
        scenario_default_h = ScenarioDisturbance.SCENARIO_META[args.scenario]['height']
        height = args.height if args.height is not None else scenario_default_h
    else:
        height = args.height if args.height is not None else 15.0

    print(f"Classical Falcon 9 controller  (v2 — 5-phase predictive)")
    print(f"  Design  : {args.rocket_design}")
    print(f"  Height  : {height} m")
    print(f"  Overlay : {'disabled' if args.no_overlay else 'enabled (PIL HUD)'}")

    output_dir = args.output_dir or os.path.join(ROOT_DIR, "videos", "classical")

    env  = make_env(args.rocket_design, height)
    ctrl = FalconController()

    # ── Scenario mode ─────────────────────────────────────────────────────────
    if args.scenario:
        dist = ScenarioDisturbance(args.scenario, start_height=height)

        print(f"\n  [SCENARIO]  {dist.label}")
        print(f"  Condition : {dist.description}")
        print(f"\nCollecting scenario trajectory...")

        traj, crash_report, impact_vel = run_scenario_episode(
            env, ctrl, dist,
            max_steps   = args.max_steps,
            linger_steps= args.linger,
            verbose     = args.diagnose,
        )
        print(f"  Recorded {len(traj)} frames")

        outcome = classify_outcome(crash_report, impact_vel)
        print_scenario_report(
            args.scenario, outcome, crash_report, impact_vel,
            n_steps=len(traj) - args.linger,
        )

        prefix = f"scenario_{args.scenario.lower()}"
        print(f"Rendering ({int(args.resolution * 16 / 9)}x{args.resolution})"
              f" -> {output_dir}/...")
        render_trajectory(
            traj,
            resolution=args.resolution,
            fps=args.fps,
            output_dir=output_dir,
            overlay=not args.no_overlay,
            prefix=prefix,
        )

    # ── Normal mode ───────────────────────────────────────────────────────────
    else:
        print(f"  Attempts: {args.attempts}")

        if args.diagnose:
            print_episode_summary(env, ctrl)

        print(f"\nCollecting trajectory ({args.attempts} attempt(s))...")
        trajectory = best_of(
            env, ctrl,
            attempts    = args.attempts,
            max_steps   = args.max_steps,
            linger_steps= args.linger,
            verbose     = args.diagnose,
        )
        print(f"  Recorded {len(trajectory)} frames")

        print(f"\nRendering ({int(args.resolution * 16 / 9)}x{args.resolution})"
              f" -> {output_dir}/...")
        render_trajectory(
            trajectory,
            resolution=args.resolution,
            fps=args.fps,
            output_dir=output_dir,
            overlay=not args.no_overlay,
        )

    env.close()
    print("Done.")


if __name__ == "__main__":
    main()
