"""PPO training with batched MuJoCo environments.

On-policy training: collect large batches from parallel envs,
compute GAE advantages, then do mini-batch PPO updates. Runs on CPU.
"""
import glob
import os
import time
from collections import defaultdict

import hydra

# Save original working directory before Hydra changes it
_ORIGINAL_CWD = os.getcwd()

import numpy as np
import torch
import tqdm
import wandb
from tensordict import TensorDict
from torchrl._utils import logger as torchrl_logger
from torchrl.data import LazyTensorStorage, TensorDictReplayBuffer
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
from torchrl.envs import ExplorationType, set_exploration_type
from torchrl.objectives import ClipPPOLoss, group_optimizers
from torchrl.objectives.value.advantages import GAE
from torchrl.record.loggers import generate_exp_name, get_logger

from utils_ppo import (
    log_metrics,
    make_environment,
    make_ppo_models,
    make_render_env,
)


# Crash report codes to names
CRASH_REPORT_NAMES = {
    0: "ongoing",
    1: "success",
    2: "crash",
    3: "roll_over",
    4: "pitch_over",
    5: "out_of_bounds",
}


def _find_latest_checkpoint(checkpoint_dir: str):
    """Return the path of the checkpoint with the most collected frames, or None."""
    best_path, best_frames = None, -1
    for p in glob.glob(os.path.join(checkpoint_dir, "ppo_eval_*.pt")):
        try:
            frames = int(os.path.splitext(os.path.basename(p))[0].rsplit("_", 1)[-1])
            if frames > best_frames:
                best_frames, best_path = frames, p
        except (ValueError, IndexError):
            pass
    # Prefer ppo_final.pt if it is newer than the best eval checkpoint
    final = os.path.join(checkpoint_dir, "ppo_final.pt")
    if os.path.exists(final):
        if best_path is None or os.path.getmtime(final) >= os.path.getmtime(best_path):
            return final
    return best_path


def aggregate_crash_stats(crash_reports):
    """Aggregate crash report statistics from a batch of episodes."""
    stats = defaultdict(int)
    for code in crash_reports:
        name = CRASH_REPORT_NAMES.get(int(code), "unknown")
        stats[name] += 1
    return dict(stats)


@hydra.main(version_base="1.1", config_path="", config_name="config_ppo")
def main(cfg: "DictConfig"):  # noqa: F821
    # Reset working directory to project root
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, ".."))
    os.chdir(project_root)

    device = cfg.network.device or "cpu"
    device = torch.device(device)

    # Create logger
    exp_name = generate_exp_name("PPO", cfg.logger.exp_name)
    logger = None
    if cfg.logger.backend:
        logger = get_logger(
            logger_type=cfg.logger.backend,
            logger_name="ppo_logging",
            experiment_name=exp_name,
            wandb_kwargs={
                "mode": cfg.logger.mode,
                "config": dict(cfg),
                "project": cfg.logger.project_name,
                "group": cfg.logger.group_name,
            },
        )

    torch.manual_seed(cfg.env.seed)
    np.random.seed(cfg.env.seed)

    log_crash_breakdown = getattr(cfg.logger, "log_crash_breakdown", True)

    # Create environments
    train_env, eval_env = make_environment(cfg, logger=logger)

    # Create CPU render env for video logging
    render_env = None
    if cfg.logger.video and cfg.logger.backend:
        render_env = make_render_env(cfg)

    # Create PPO actor and critic
    actor, critic = make_ppo_models(cfg, train_env, device)

    # Create GAE advantage module
    adv_module = GAE(
        gamma=cfg.loss.gamma,
        lmbda=cfg.loss.gae_lambda,
        value_network=critic,
        average_gae=False,
        device=device,
    )

    # Create PPO loss module
    loss_module = ClipPPOLoss(
        actor_network=actor,
        critic_network=critic,
        clip_epsilon=cfg.loss.clip_epsilon,
        loss_critic_type=cfg.loss.loss_critic_type,
        entropy_coeff=cfg.loss.entropy_coeff,
        critic_coeff=cfg.loss.critic_coeff,
        normalize_advantage=True,
    )

    # Create optimizers
    actor_optim = torch.optim.Adam(
        actor.parameters(), lr=cfg.optim.lr, eps=1e-5
    )
    critic_optim = torch.optim.Adam(
        critic.parameters(), lr=cfg.optim.lr, eps=1e-5
    )
    optim = group_optimizers(actor_optim, critic_optim)
    del actor_optim, critic_optim

    # Mini-batch data buffer (SamplerWithoutReplacement = shuffle without replacement each epoch)
    sampler = SamplerWithoutReplacement()
    data_buffer = TensorDictReplayBuffer(
        storage=LazyTensorStorage(
            cfg.collector.frames_per_batch,
            device=device,
        ),
        sampler=sampler,
        batch_size=cfg.loss.mini_batch_size,
    )

    # Create checkpoint directory
    checkpoint_dir = os.path.join(script_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    # --- Auto-resume from latest checkpoint ---
    collected_frames = 0
    num_network_updates = 0
    resume_path = _find_latest_checkpoint(checkpoint_dir)
    if resume_path:
        torchrl_logger.info(f"Auto-resuming from: {resume_path}")
        ckpt = torch.load(resume_path, map_location=device, weights_only=False)
        actor.load_state_dict(ckpt["actor_state_dict"])
        critic.load_state_dict(ckpt["critic_state_dict"])
        if "optim_state_dict" in ckpt:
            optim.load_state_dict(ckpt["optim_state_dict"])
        collected_frames = ckpt.get("collected_frames", 0)
        num_network_updates = ckpt.get("num_network_updates", 0)
        torchrl_logger.info(f"Resumed at {collected_frames:,} frames ({num_network_updates:,} updates)")
    else:
        torchrl_logger.info("No checkpoint found — starting fresh")

    # If the previous run already finished, train for that many more frames
    total_frames = cfg.collector.total_frames
    if collected_frames >= total_frames:
        total_frames = collected_frames + cfg.collector.total_frames
        torchrl_logger.info(
            f"Previous target reached. Training {cfg.collector.total_frames:,} more frames "
            f"(target → {total_frames:,})."
        )

    # Precompute loop constants (use total_frames so annealing is correct on resume)
    frames_per_batch = cfg.collector.frames_per_batch
    num_mini_batches = frames_per_batch // cfg.loss.mini_batch_size
    total_network_updates = (
        (total_frames // frames_per_batch) * cfg.loss.ppo_epochs * num_mini_batches
    )
    ppo_epochs = cfg.loss.ppo_epochs
    max_grad_norm = cfg.loss.max_grad_norm
    eval_iter = cfg.logger.eval_iter
    eval_rollout_steps = cfg.env.max_episode_steps
    anneal_lr = cfg.loss.anneal_lr
    anneal_clip_epsilon = cfg.loss.anneal_clip_epsilon
    cfg_lr = cfg.optim.lr
    cfg_clip_epsilon = cfg.loss.clip_epsilon
    steps_per_env = frames_per_batch // cfg.env.num_envs

    # Main loop
    start_time = time.time()
    pbar = tqdm.tqdm(total=total_frames, initial=collected_frames)
    batch_crash_reports = []

    losses = TensorDict(batch_size=[ppo_epochs, num_mini_batches])

    # Reset env once; auto-reset handles episode boundaries internally
    td = train_env.reset()

    while collected_frames < total_frames:
        # --- Collect rollout by stepping env directly ---
        collect_start = time.time()
        rollout_tds = []
        for _ in range(steps_per_env):
            with torch.no_grad():
                td = actor(td)
            # step_and_maybe_reset returns:
            #   transition_td: the full step td (with "next") for training
            #   td: the reset root td for the next iteration
            transition_td, td = train_env.step_and_maybe_reset(td)
            rollout_tds.append(transition_td.clone())

        data = torch.stack(rollout_tds, dim=1)  # (num_envs, steps_per_env, ...)
        collect_time = time.time() - collect_start

        frames_in_batch = data.numel()
        collected_frames += frames_in_batch
        pbar.update(frames_in_batch)

        metrics_to_log = {}
        metrics_to_log["train/collect_time"] = collect_time
        metrics_to_log["train/fps"] = frames_in_batch / collect_time

        # --- Training rewards ---
        episode_end = data["next", "done"]
        episode_rewards = data["next", "episode_reward"][episode_end]
        if len(episode_rewards) > 0:
            episode_length = data["next", "step_count"][episode_end]
            metrics_to_log["train/reward"] = episode_rewards.mean().item()
            metrics_to_log["train/episode_length"] = (
                episode_length.sum().item() / len(episode_length)
            )

        # --- Crash statistics ---
        if ("next", "crash_report") in data.keys(True):
            done_mask = data["next", "done"].squeeze(-1)
            if done_mask.any():
                crash_codes = data["next", "crash_report"][done_mask]
                batch_crash_reports.extend(crash_codes.tolist())

        if log_crash_breakdown and batch_crash_reports:
            crash_stats = aggregate_crash_stats(batch_crash_reports)
            total_episodes = sum(crash_stats.values())
            for crash_type, count in crash_stats.items():
                if crash_type == "ongoing":
                    continue
                if crash_type == "success":
                    metrics_to_log["train/landing_success_rate"] = (
                        count / total_episodes if total_episodes > 0 else 0
                    )
                else:
                    metrics_to_log[f"train/crash_{crash_type}"] = count
                    metrics_to_log[f"train/crash_{crash_type}_rate"] = (
                        count / total_episodes if total_episodes > 0 else 0
                    )
            batch_crash_reports = []

        # --- PPO training ---
        training_start = time.time()
        for j in range(ppo_epochs):
            # Compute GAE advantages (no grad)
            with torch.no_grad():
                data = adv_module(data)

            data_reshape = data.reshape(-1)
            data_buffer.extend(data_reshape)

            for k, batch in enumerate(data_buffer):
                # Learning rate annealing
                alpha = 1.0
                if anneal_lr:
                    alpha = 1 - (num_network_updates / total_network_updates)
                    for group in optim.param_groups:
                        group["lr"] = cfg_lr * alpha
                if anneal_clip_epsilon:
                    loss_module.clip_epsilon.copy_(cfg_clip_epsilon * alpha)

                num_network_updates += 1

                optim.zero_grad(set_to_none=True)
                loss = loss_module(batch)
                total_loss = (
                    loss["loss_objective"] + loss["loss_entropy"] + loss["loss_critic"]
                )
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(loss_module.parameters(), max_grad_norm)
                optim.step()

                losses[j, k] = loss.detach().select(
                    "loss_critic", "loss_entropy", "loss_objective"
                )

        training_time = time.time() - training_start

        # --- Log losses ---
        losses_mean = losses.apply(lambda x: x.float().mean(), batch_size=[])
        for key, value in losses_mean.items():
            metrics_to_log[f"train/{key}"] = value.item()
        metrics_to_log["train/training_time"] = training_time

        # --- Evaluation ---
        if abs(collected_frames % eval_iter) < frames_per_batch:
            with set_exploration_type(ExplorationType.DETERMINISTIC), torch.no_grad():
                eval_start = time.time()
                actor.eval()
                eval_rollout = eval_env.rollout(
                    eval_rollout_steps,
                    actor,
                    auto_cast_to_device=True,
                    break_when_any_done=True,
                )
                actor.train()
                eval_time = time.time() - eval_start
                eval_reward = eval_rollout["next", "reward"].sum(-2).mean().item()
                eval_episode_length = eval_rollout.batch_size[-1]
                metrics_to_log["eval/reward"] = eval_reward
                metrics_to_log["eval/time"] = eval_time
                metrics_to_log["eval/episode_length"] = eval_episode_length

                # Detailed eval metrics (single eval env: shape [1, T, ...])
                # Use "observation" not "next/observation" to get state before reset
                final_obs = eval_rollout["observation"][0, -1]
                pos = final_obs[:3].cpu().numpy()
                vel = final_obs[6:9].cpu().numpy()
                metrics_to_log["eval/final_horizontal_error"] = float(
                    np.sqrt(pos[0] ** 2 + pos[1] ** 2)
                )
                metrics_to_log["eval/final_altitude"] = float(pos[2])
                metrics_to_log["eval/final_velocity"] = float(np.linalg.norm(vel))

                if "action" in eval_rollout.keys():
                    actions = eval_rollout["action"].cpu().numpy()
                    metrics_to_log["eval/total_thrust"] = float(np.sum(np.abs(actions)))

            # Record video from CPU render env
            if render_env is not None:
                with set_exploration_type(ExplorationType.DETERMINISTIC), torch.no_grad():
                    actor.eval()
                    vid_rollout = render_env.rollout(
                        eval_rollout_steps,
                        actor,
                        auto_cast_to_device=True,
                        break_when_any_done=True,
                    )
                    actor.train()
                    if "pixels" in vid_rollout.keys():
                        pixels = vid_rollout["pixels"]
                        if pixels.ndim == 5:
                            pixels = pixels[0]  # [T, H, W, C]
                        vid = pixels.cpu().numpy()
                        vid = np.transpose(vid, (0, 3, 1, 2))  # [T,C,H,W]
                        vid = vid.astype(np.uint8)
                        wandb.log({"eval/video": wandb.Video(vid, fps=20, format="mp4")})

            # Save checkpoint at eval
            ckpt_path = os.path.join(checkpoint_dir, f"ppo_eval_{collected_frames}.pt")
            torch.save({
                "actor_state_dict": actor.state_dict(),
                "critic_state_dict": critic.state_dict(),
                "optim_state_dict": optim.state_dict(),
                "collected_frames": collected_frames,
                "num_network_updates": num_network_updates,
                "config": dict(cfg),
            }, ckpt_path)
            torchrl_logger.info(f"Saved checkpoint: {ckpt_path}")

        if logger is not None:
            log_metrics(logger, metrics_to_log, collected_frames)

    # Save final checkpoint
    final_ckpt_path = os.path.join(checkpoint_dir, "ppo_final.pt")
    torch.save({
        "actor_state_dict": actor.state_dict(),
        "critic_state_dict": critic.state_dict(),
        "optim_state_dict": optim.state_dict(),
        "collected_frames": collected_frames,
        "num_network_updates": num_network_updates,
        "config": dict(cfg),
    }, final_ckpt_path)
    torchrl_logger.info(f"Saved final checkpoint: {final_ckpt_path}")

    # Cleanup
    if render_env is not None and not render_env.is_closed:
        render_env.close()
    if not eval_env.is_closed:
        eval_env.close()
    if not train_env.is_closed:
        train_env.close()
    end_time = time.time()
    torchrl_logger.info(f"Training took {end_time - start_time:.2f} seconds to finish")

    if logger is not None:
        wandb.finish()


if __name__ == "__main__":
    main()
