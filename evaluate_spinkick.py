"""Evaluate a single Spinkick checkpoint and write tracking metrics as JSON.

Adapted from mjlab.tasks.tracking.scripts.evaluate to:
  - register the Spinkick task (by importing spinkick_example),
  - skip the "Tracking" name filter,
  - load a local --checkpoint-file directly (no W&B dependency).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import cast

import torch
import tyro
from rsl_rl.runners import OnPolicyRunner

import spinkick_example  # noqa: F401  -- registers Mjlab-Spinkick-Unitree-G1
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.tasks.tracking.mdp import MotionCommandCfg
from mjlab.tasks.tracking.mdp.commands import MotionCommand
from mjlab.tasks.tracking.mdp.metrics import (
  compute_ee_orientation_error,
  compute_ee_position_error,
  compute_joint_velocity_error,
  compute_mpkpe,
  compute_root_relative_mpkpe,
)
from mjlab.utils.torch import configure_torch_backends

TASK_ID = "Mjlab-Spinkick-Unitree-G1"


@dataclass(frozen=True)
class Config:
  checkpoint_file: str
  """Local checkpoint .pt path."""
  motion_file: str
  """Local motion .npz path used during training."""
  num_envs: int = 1024
  device: str | None = None
  output_file: str | None = None


def run(cfg: Config) -> dict[str, float]:
  configure_torch_backends()
  device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")

  ckpt = Path(cfg.checkpoint_file)
  motion = Path(cfg.motion_file)
  if not ckpt.is_file():
    raise FileNotFoundError(f"Checkpoint not found: {ckpt}")
  if not motion.is_file():
    raise FileNotFoundError(f"Motion file not found: {motion}")

  env_cfg = load_env_cfg(TASK_ID, play=False)
  agent_cfg = load_rl_cfg(TASK_ID)

  motion_cmd = env_cfg.commands.get("motion")
  if not isinstance(motion_cmd, MotionCommandCfg):
    raise ValueError(f"Task {TASK_ID} is not a tracking task.")
  motion_cmd.motion_file = str(motion)
  motion_cmd.sampling_mode = "start"
  env_cfg.observations["actor"].enable_corruption = True
  env_cfg.events.pop("push_robot", None)
  env_cfg.scene.num_envs = cfg.num_envs

  env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
  env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

  runner_cls = load_runner_cls(TASK_ID) or OnPolicyRunner
  runner = runner_cls(env, asdict(agent_cfg), device=device)
  runner.load(str(ckpt.resolve()), load_cfg={"actor": True}, map_location=device)
  policy = runner.get_inference_policy(device=device)

  command = cast(MotionCommand, env.unwrapped.command_manager.get_term("motion"))
  ee_body_names = env_cfg.terminations["ee_body_pos"].params["body_names"]
  print(f"[INFO] Loaded checkpoint: {ckpt}")
  print(f"[INFO] Motion: {motion}")
  print(f"[INFO] End effector bodies: {ee_body_names}")

  all_mpkpe, all_r_mpkpe, all_jv, all_eep, all_eeo = [], [], [], [], []
  done_envs = torch.zeros(cfg.num_envs, dtype=torch.bool, device=device)
  success = torch.zeros(cfg.num_envs, dtype=torch.bool, device=device)

  obs = env.get_observations()
  env.unwrapped.command_manager.compute(dt=env.unwrapped.step_dt)

  print(f"[INFO] Running {cfg.num_envs} evaluation episodes...")
  step = 0
  while not done_envs.all():
    with torch.no_grad():
      actions = policy(obs)
    obs, _, dones, _ = env.step(actions)

    active = ~done_envs
    if active.any():
      all_mpkpe.append(torch.where(active, compute_mpkpe(command), 0.0))
      all_r_mpkpe.append(torch.where(active, compute_root_relative_mpkpe(command), 0.0))
      all_jv.append(torch.where(active, compute_joint_velocity_error(command), 0.0))
      all_eep.append(
        torch.where(active, compute_ee_position_error(command, ee_body_names), 0.0)
      )
      all_eeo.append(
        torch.where(active, compute_ee_orientation_error(command, ee_body_names), 0.0)
      )

    terminated = env.unwrapped.termination_manager.terminated
    truncated = env.unwrapped.termination_manager.time_outs
    newly_done = dones.bool() & ~done_envs
    if newly_done.any():
      success = success | (newly_done & truncated & ~terminated)
      done_envs = done_envs | newly_done
    step += 1

  stacks = [torch.stack(s, dim=0) for s in (all_mpkpe, all_r_mpkpe, all_jv, all_eep, all_eeo)]
  active_steps = (stacks[0] != 0).sum(dim=0).float().clamp(min=1)
  means = [s.sum(dim=0) / active_steps for s in stacks]

  metrics = {
    "success_rate": success.float().mean().item(),
    "mpkpe": means[0].mean().item(),
    "r_mpkpe": means[1].mean().item(),
    "joint_vel_error": means[2].mean().item(),
    "ee_pos_error": means[3].mean().item(),
    "ee_ori_error": means[4].mean().item(),
  }

  print("\n" + "=" * 50)
  print("Evaluation Results")
  print("=" * 50)
  for name, value in metrics.items():
    print(f"  {name}: {value:.4f}")
  print("=" * 50)

  if cfg.output_file:
    out = Path(cfg.output_file)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
      json.dump(metrics, f, indent=2)
    print(f"[INFO] Metrics saved to {out}")

  env.close()
  return metrics


if __name__ == "__main__":
  run(tyro.cli(Config))
