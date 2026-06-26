"""Batched rocket landing environment using MuJoCo Warp.

Runs parallel rocket simulations via NVIDIA Warp (GPU when available, CPU fallback),
with observation/reward/done logic in pure PyTorch.

Usage with TorchRL:
    env = RocketLanderWarp(num_envs=64, device="cpu")
    td = env.reset()
    td = env.step(td)  # steps all environments at once
"""

import os
import sys

import mujoco
import mujoco_warp as mjw
import numpy as np
import torch
import warp as wp
from tensordict import TensorDict, TensorDictBase
from torchrl.data import Bounded, Composite, Unbounded
from torchrl.envs import EnvBase

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from env.config import ROCKET_DESIGNS


def _quat_to_euler_batch(quat):
    """Convert batched quaternions [N, 4] (w,x,y,z) to euler angles [N, 3] in degrees."""
    w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]

    # Roll
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = torch.atan2(sinr_cosp, cosr_cosp)

    # Pitch
    sinp = 2 * (w * y - z * x)
    pitch = torch.where(
        sinp.abs() >= 1,
        torch.copysign(torch.tensor(torch.pi / 2, device=quat.device), sinp),
        torch.asin(sinp),
    )

    # Yaw
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = torch.atan2(siny_cosp, cosy_cosp)

    return torch.rad2deg(torch.stack([roll, pitch, yaw], dim=-1))


class RocketLanderWarp(EnvBase):
    """Batched rocket landing environment using MuJoCo Warp.

    All N environments step simultaneously. Observation, reward,
    and termination logic are computed in batched PyTorch ops.
    """

    def __init__(
        self,
        num_envs: int = 4096,
        rocket_design: str = "v0",
        device: str = "cpu",
        max_episode_steps: int = 1000,
        frame_skip: int = 5,
        starting_height: float = 50.0,
        # Reset noise (domain randomization for initial conditions)
        reset_pos_noise: float = 3.0,  # xy position noise in meters
        reset_vel_noise: float = 3.0,  # linear velocity noise in m/s
        reset_ang_noise: float = 0.15,  # angular noise in radians (~8 degrees)
        reset_angvel_noise: float = 0.3,  # angular velocity noise in rad/s
        # Reward weights (exponential shaping, per-step components in [0,1])
        w_distance: float = 0.7,
        w_velocity: float = 0.15,
        w_upright: float = 0.1,
        w_angular: float = 0.05,
        w_success: float = 100.0,
        w_crash: float = -10.0,
        w_tipover: float = -10.0,
        # Velocity penalty shaping (exponential gate near ground)
        vel_gate_scale: float = 10.0,  # height scale for danger zone (meters)
        vel_penalty_coeff: float = 1.0,  # steepness of velocity penalty
        # Time penalty
        w_time_penalty: float = 0.3,  # per-step cost to incentivize landing
        # Termination
        max_angle: float = 70.0,
        max_distance: float = 20.0,
        crash_velocity: float = 5.0,  # m/s — ground contact above this = crash
    ):
        self._num_envs = num_envs
        self._device = torch.device(device)
        self._max_episode_steps = max_episode_steps
        self._frame_skip = frame_skip
        self._starting_height = starting_height
        self._reset_pos_noise = reset_pos_noise
        self._reset_vel_noise = reset_vel_noise
        self._reset_ang_noise = reset_ang_noise
        self._reset_angvel_noise = reset_angvel_noise
        self._max_angle = max_angle
        self._max_distance = max_distance
        self._crash_velocity = crash_velocity
        self._vel_gate_scale = vel_gate_scale
        self._vel_penalty_coeff = vel_penalty_coeff
        self._time_penalty = w_time_penalty

        # Reward weights as tensors on device
        self._w = {
            "distance": w_distance,
            "velocity": w_velocity,
            "upright": w_upright,
            "angular": w_angular,
            "success": w_success,
            "crash": w_crash,
            "tipover": w_tipover,
        }

        # Load rocket design config
        design_cfg = ROCKET_DESIGNS[rocket_design]
        self._target_height = design_cfg.target_height
        xml_path = os.path.join(ROOT_DIR, "env", "xml_files", design_cfg.xml_file)

        # Load MuJoCo model and create Warp data
        os.chdir(ROOT_DIR)
        self._mjm = mujoco.MjModel.from_xml_path(xml_path)
        self._mjw_model = mjw.put_model(self._mjm)
        self._mjw_data = mjw.make_data(self._mjm, nworld=num_envs, nconmax=20, njmax=50)

        self._nq = self._mjm.nq  # 7 for free joint (3 pos + 4 quat)
        self._nv = self._mjm.nv  # 6 for free joint (3 vel + 3 ang_vel)
        self._nu = self._mjm.nu  # 3 actuators

        # Store initial qpos for resets
        mjd = mujoco.MjData(self._mjm)
        self._init_qpos = torch.tensor(mjd.qpos.copy(), dtype=torch.float32, device=self._device)
        self._init_qvel = torch.tensor(mjd.qvel.copy(), dtype=torch.float32, device=self._device)

        # Step counter per environment
        self._step_count = torch.zeros(num_envs, dtype=torch.int32, device=self._device)

        # Velocity history for approach speed tracking (max over last N steps)
        self._vel_history_len = 10
        self._vel_history = torch.zeros(num_envs, self._vel_history_len, device=self._device)
        self._vel_history_idx = 0

        # Capture CUDA graph for fast stepping (GPU only; falls back to direct stepping on CPU)
        self._graph = None
        if torch.cuda.is_available():
            self._capture_cuda_graph()

        super().__init__(device=self._device, batch_size=torch.Size([num_envs]))
        self._make_spec()

    def _capture_cuda_graph(self):
        """Capture a CUDA graph for the step function for maximum throughput (GPU only)."""
        mjw.step(self._mjw_model, self._mjw_data)
        wp.synchronize()
        with wp.ScopedCapture() as capture:
            for _ in range(self._frame_skip):
                mjw.step(self._mjw_model, self._mjw_data)
        self._graph = capture.graph

    def _make_spec(self, td_params=None):
        """Define observation, action, reward, and done specs."""
        self.observation_spec = Composite(
            observation=Unbounded(shape=(self._num_envs, 12), dtype=torch.float32, device=self._device),
            shape=(self._num_envs,),
        )
        self.action_spec = Composite(
            action=Bounded(
                low=torch.tensor([-1.0, -1.0, 0.0], device=self._device).expand(self._num_envs, -1),
                high=torch.tensor([1.0, 1.0, 1.0], device=self._device).expand(self._num_envs, -1),
                dtype=torch.float32,
                device=self._device,
            ),
            shape=(self._num_envs,),
        )
        self.reward_spec = Unbounded(shape=(self._num_envs, 1), dtype=torch.float32, device=self._device)
        self.done_spec = Composite(
            done=Unbounded(shape=(self._num_envs, 1), dtype=torch.bool, device=self._device),
            terminated=Unbounded(shape=(self._num_envs, 1), dtype=torch.bool, device=self._device),
            truncated=Unbounded(shape=(self._num_envs, 1), dtype=torch.bool, device=self._device),
            shape=(self._num_envs,),
        )

    def _quat_mul(self, q1, q2):
        """Batched quaternion multiplication q1 * q2.

        Args:
            q1, q2: [N, 4] quaternions (w, x, y, z)

        Returns:
            [N, 4] quaternion product
        """
        w1, x1, y1, z1 = q1.unbind(-1)
        w2, x2, y2, z2 = q2.unbind(-1)
        return torch.stack([
            w1*w2 - x1*x2 - y1*y2 - z1*z2,
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2,
        ], dim=-1)

    def _build_obs(self, qpos, qvel):
        """Build batched 12-dim observations from qpos/qvel tensors.

        Args:
            qpos: [N, nq] joint positions
            qvel: [N, nv] joint velocities

        Returns:
            [N, 12] observation tensor: [pos(3), euler(3), vel(3), ang_vel(3)]
        """
        pos = qpos[:, :3]  # [N, 3]
        quat = qpos[:, 3:7]  # [N, 4]
        vel = qvel[:, :3]  # [N, 3]
        angular_vel = qvel[:, 3:6]  # [N, 3]

        # Quaternion to euler (degrees)
        euler = _quat_to_euler_batch(quat)  # [N, 3] (roll, pitch, yaw)

        return torch.cat([pos, euler, vel, angular_vel], dim=-1)  # [N, 12]

    def _compute_reward(self, obs, done_mask, crash_type, pre_step_vel=None, max_recent_vel=None):
        """Compute batched rewards using exponential shaping.

        All per-step components are in [0, 1] via exp(-k * x), so longer
        episodes accumulate more reward and the agent is incentivised to survive.

        Args:
            obs: [N, 12] observations
            done_mask: [N] bool tensor
            crash_type: [N] int tensor (0=ongoing, 1=success, 2=crash, 3=roll, 4=pitch, 5=oob)
            pre_step_vel: [N, 3] velocity before physics step (for impact penalty)

        Returns:
            [N, 1] reward tensor
        """
        pos_x, pos_y, pos_z = obs[:, 0], obs[:, 1], obs[:, 2]
        roll_deg, pitch_deg = obs[:, 3], obs[:, 4]
        vel_x, vel_y, vel_z = obs[:, 6], obs[:, 7], obs[:, 8]
        ang_x, ang_y, ang_z = obs[:, 9], obs[:, 10], obs[:, 11]

        # Distance to target (3D)
        dist_3d = torch.sqrt(pos_x**2 + pos_y**2 + (pos_z - self._target_height)**2)
        r_distance = torch.exp(-0.05 * dist_3d)

        # Velocity penalty: safe speed depends on altitude
        # At high altitude: allow fast descent. Near ground: must be slow.
        vel_mag = torch.sqrt(vel_x**2 + vel_y**2 + vel_z**2)
        safe_vel = 1.0 + pos_z * 0.4  # At z=50m: 21 m/s ok. At z=2m: 1.8 m/s. At z=0: 1 m/s
        excess_vel = torch.clamp(vel_mag - safe_vel, min=0.0)
        r_velocity = torch.exp(-self._vel_penalty_coeff * excess_vel)

        # Upright reward (tilt in radians)
        tilt_rad = (roll_deg.abs() + pitch_deg.abs()) * (torch.pi / 180.0)
        r_upright = torch.exp(-2.0 * tilt_rad)

        # Angular velocity reward
        ang_mag = torch.sqrt(ang_x**2 + ang_y**2 + ang_z**2)
        r_angular = torch.exp(-0.5 * ang_mag)

        # Weighted sum (weights should sum to 1.0)
        reward = (
            self._w["distance"] * r_distance
            + self._w["velocity"] * r_velocity
            + self._w["upright"] * r_upright
            + self._w["angular"] * r_angular
            - self._time_penalty
        )

        # Terminal bonuses / penalties
        # Success: bonus scaled by max velocity over last N steps (approach speed)
        # This prevents exploiting MuJoCo contact absorption — the policy must
        # genuinely approach slowly, not just slam and let the legs absorb impact.
        success_mask = (crash_type == 1).float()
        if max_recent_vel is not None:
            approach_vel = max_recent_vel
        elif pre_step_vel is not None:
            approach_vel = torch.sqrt((pre_step_vel**2).sum(dim=-1))
        else:
            approach_vel = vel_mag
        r_soft = torch.exp(-1.0 * approach_vel)  # 0.5 m/s → 0.61, 1.0 m/s → 0.37
        reward = reward + success_mask * self._w["success"] * r_soft

        reward = reward + (crash_type == 2).float() * self._w["crash"]
        reward = reward + ((crash_type == 3) | (crash_type == 4)).float() * self._w["tipover"]

        return reward.unsqueeze(-1)

    def _compute_done(self, obs, pre_step_vel=None):
        """Compute termination conditions.

        Returns:
            terminated: [N] bool
            truncated: [N] bool
            crash_type: [N] int (0=ongoing, 1=success, 2=crash, 3=roll, 4=pitch, 5=oob)
        """
        pos_x, pos_y, pos_z = obs[:, 0], obs[:, 1], obs[:, 2]
        roll = obs[:, 3]
        pitch = obs[:, 4]
        vel_x, vel_y, vel_z = obs[:, 6], obs[:, 7], obs[:, 8]
        h_dist = torch.sqrt(pos_x**2 + pos_y**2)

        crash_type = torch.zeros(self._num_envs, dtype=torch.int32, device=self._device)

        # Failure conditions (lower priority assigned first, higher overrides)
        oob = h_dist > self._max_distance
        pitch_over = pitch.abs() > self._max_angle
        roll_over = roll.abs() > self._max_angle
        crashed = pos_z < 0.5

        crash_type = torch.where(oob, torch.tensor(5, device=self._device), crash_type)
        crash_type = torch.where(pitch_over, torch.tensor(4, device=self._device), crash_type)
        crash_type = torch.where(roll_over, torch.tensor(3, device=self._device), crash_type)
        crash_type = torch.where(crashed, torch.tensor(2, device=self._device), crash_type)

        # Hard-landing crash: near surface + pre-step velocity too high
        if pre_step_vel is not None:
            pre_vel_mag = torch.sqrt((pre_step_vel**2).sum(dim=-1))
        else:
            pre_vel_mag = torch.sqrt(vel_x**2 + vel_y**2 + vel_z**2)
        near_surface = pos_z < (self._target_height + 0.5)
        hard_landing = near_surface & (pre_vel_mag > self._crash_velocity)
        crash_type = torch.where(hard_landing, torch.tensor(2, device=self._device), crash_type)

        # Success condition (highest priority — overrides crash when rocket touches down softly)
        near_ground = pos_z < (self._target_height + 0.1)
        above_crash = pos_z >= 0.5
        near_pad = h_dist < 2.0
        slow = pre_vel_mag < 1.0
        upright = (roll.abs() < 15.0) & (pitch.abs() < 15.0)
        success = near_ground & above_crash & near_pad & slow & upright
        crash_type = torch.where(success, torch.tensor(1, device=self._device), crash_type)

        terminated = crash_type > 0
        truncated = self._step_count >= self._max_episode_steps

        return terminated, truncated, crash_type

    def _warp_to_torch(self, warp_array):
        """Zero-copy convert a Warp array to a PyTorch tensor."""
        return wp.to_torch(warp_array)

    def _torch_to_warp(self, tensor):
        """Zero-copy convert a PyTorch tensor to a Warp array."""
        return wp.from_torch(tensor)

    def _get_state_tensors(self):
        """Get qpos and qvel as PyTorch tensors (zero-copy from Warp)."""
        qpos = self._warp_to_torch(self._mjw_data.qpos)  # [N, nq]
        qvel = self._warp_to_torch(self._mjw_data.qvel)  # [N, nv]
        return qpos, qvel

    def _reset_envs(self, env_mask):
        """Reset specific environments by mask.

        Args:
            env_mask: [N] bool tensor, True for envs to reset
        """
        if not env_mask.any():
            return

        qpos = self._warp_to_torch(self._mjw_data.qpos)
        qvel = self._warp_to_torch(self._mjw_data.qvel)

        # Reset qpos to initial + noise
        init_qpos = self._init_qpos.unsqueeze(0).expand(self._num_envs, -1)
        new_qpos = init_qpos.clone()

        # Add xy position noise (start offset from target)
        new_qpos[:, 0] += torch.randn(self._num_envs, device=self._device) * self._reset_pos_noise
        new_qpos[:, 1] += torch.randn(self._num_envs, device=self._device) * self._reset_pos_noise
        # Set height to starting height (no noise on z)
        new_qpos[:, 2] = self._starting_height

        # Add angular noise via small euler perturbation to quaternion
        # Generate small random euler angles and convert to quaternion perturbation
        ang_noise = torch.randn(self._num_envs, 3, device=self._device) * self._reset_ang_noise
        # Simple axis-angle to quaternion for small angles: q ≈ [1, θ/2]
        half_ang = ang_noise * 0.5
        noise_quat = torch.cat([
            torch.ones(self._num_envs, 1, device=self._device),
            half_ang
        ], dim=-1)
        noise_quat = torch.nn.functional.normalize(noise_quat, dim=-1)
        # Quaternion multiply: q_new = q_noise * q_init (apply perturbation)
        q0 = new_qpos[:, 3:7]
        new_qpos[:, 3:7] = self._quat_mul(noise_quat, q0)
        new_qpos[:, 3:7] = torch.nn.functional.normalize(new_qpos[:, 3:7], dim=-1)

        # Reset qvel with separate linear and angular noise
        new_qvel = torch.zeros_like(qvel)
        new_qvel[:, :3] = torch.randn(self._num_envs, 3, device=self._device) * self._reset_vel_noise
        new_qvel[:, 3:6] = torch.randn(self._num_envs, 3, device=self._device) * self._reset_angvel_noise

        # Apply only to masked environments
        qpos[env_mask] = new_qpos[env_mask]
        qvel[env_mask] = new_qvel[env_mask]

        # Reset step counter and velocity history
        self._step_count[env_mask] = 0
        self._vel_history[env_mask] = 0.0

    def _step(self, td: TensorDictBase) -> TensorDictBase:
        """Step all environments with given actions."""
        # Store pre-step velocity for impact force penalty
        _, qvel_pre = self._get_state_tensors()
        pre_step_vel = qvel_pre[:, :3].clone()  # [N, 3] linear velocity before physics

        # Record velocity magnitude in rolling history buffer
        pre_vel_mag = torch.sqrt((pre_step_vel**2).sum(dim=-1))
        self._vel_history[:, self._vel_history_idx] = pre_vel_mag
        self._vel_history_idx = (self._vel_history_idx + 1) % self._vel_history_len
        # Max velocity over recent steps — captures true approach speed before contact absorption
        max_recent_vel = self._vel_history.max(dim=-1).values

        # Write actions to Warp ctrl
        actions = td["action"]  # [N, 3]
        ctrl = self._warp_to_torch(self._mjw_data.ctrl)
        ctrl.copy_(actions)

        # Step physics (frame_skip steps; uses CUDA graph on GPU, direct stepping on CPU)
        if self._graph is not None:
            wp.capture_launch(self._graph)
        else:
            for _ in range(self._frame_skip):
                mjw.step(self._mjw_model, self._mjw_data)
        wp.synchronize()

        self._step_count += 1

        # Read state
        qpos, qvel = self._get_state_tensors()
        obs = self._build_obs(qpos, qvel)

        # Compute termination
        terminated, truncated, crash_type = self._compute_done(obs, pre_step_vel=pre_step_vel)
        done = terminated | truncated

        # Compute reward (pass max recent velocity for landing quality bonus)
        reward = self._compute_reward(obs, done, crash_type, pre_step_vel=pre_step_vel, max_recent_vel=max_recent_vel)

        # Auto-reset done environments
        self._reset_envs(done)

        # Get observation after reset for done envs
        if done.any():
            qpos_after, qvel_after = self._get_state_tensors()
            obs_after = self._build_obs(qpos_after, qvel_after)
        else:
            obs_after = obs

        return TensorDict(
            {
                "observation": obs_after,
                "reward": reward,
                "done": done.unsqueeze(-1),
                "terminated": terminated.unsqueeze(-1),
                "truncated": truncated.unsqueeze(-1),
                "crash_report": crash_type,
            },
            batch_size=self.batch_size,
            device=self._device,
        )

    def _reset(self, td: TensorDictBase = None, **kwargs) -> TensorDictBase:
        """Reset all or specific environments."""
        if td is not None and "_reset" in td.keys():
            # Selective reset
            env_mask = td["_reset"].squeeze(-1)
        else:
            # Full reset
            env_mask = torch.ones(self._num_envs, dtype=torch.bool, device=self._device)

        self._reset_envs(env_mask)

        # Forward dynamics to update state
        mjw.step(self._mjw_model, self._mjw_data)
        wp.synchronize()

        qpos, qvel = self._get_state_tensors()
        obs = self._build_obs(qpos, qvel)

        return TensorDict(
            {
                "observation": obs,
                "done": torch.zeros(self._num_envs, 1, dtype=torch.bool, device=self._device),
                "terminated": torch.zeros(self._num_envs, 1, dtype=torch.bool, device=self._device),
                "truncated": torch.zeros(self._num_envs, 1, dtype=torch.bool, device=self._device),
            },
            batch_size=self.batch_size,
            device=self._device,
        )

    def _set_seed(self, seed):
        torch.manual_seed(seed)

    def set_curriculum_height(self, height: float):
        """Update starting height for curriculum learning."""
        self._starting_height = height


if __name__ == "__main__":
    import time

    num_envs = 64
    device = "cpu"
    print("Testing RocketLanderWarp...")
    env = RocketLanderWarp(num_envs=num_envs, device=device, rocket_design="v0")
    print(f"  Created {env._num_envs} environments on {env._device}")
    print(f"  observation_spec: {env.observation_spec}")
    print(f"  action_spec: {env.action_spec}")

    td = env.reset()
    print(f"  Reset obs shape: {td['observation'].shape}")

    # Benchmark
    n_steps = 50
    actions = torch.rand(num_envs, 3, device=device) * 2 - 1
    actions[:, 2] = actions[:, 2].abs()  # thrust_z >= 0

    # Warmup
    for _ in range(5):
        td["action"] = actions
        td = env.step(td)

    t0 = time.time()
    for _ in range(n_steps):
        td["action"] = actions
        td = env.step(td)
    elapsed = time.time() - t0

    total_steps = n_steps * num_envs
    sps = total_steps / elapsed
    print(f"\n  Benchmark: {n_steps} steps x {num_envs} envs = {total_steps} total steps")
    print(f"  Time: {elapsed:.2f}s")
    print(f"  Throughput: {sps:,.0f} steps/sec")
