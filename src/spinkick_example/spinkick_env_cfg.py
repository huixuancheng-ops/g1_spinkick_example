"""Spinkick environment configuration for Unitree G1."""

import math
from dataclasses import dataclass, field
from typing import Tuple

import torch
from mjlab.entity import Entity
from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg
from mjlab.managers import TerminationTermCfg
from mjlab.rl.config import RslRlOnPolicyRunnerCfg
from mjlab.tasks.tracking.config.g1.env_cfgs import unitree_g1_flat_tracking_env_cfg

_MAX_ANG_VEL = 500 * math.pi / 180.0  # [rad/s]


def base_ang_vel_exceed(
  env: ManagerBasedRlEnv,
  threshold: float,
) -> torch.Tensor:
  asset: Entity = env.scene["robot"]
  ang_vel = asset.data.root_link_ang_vel_b
  return torch.any(ang_vel.abs() > threshold, dim=-1)


def unitree_g1_spinkick_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create Unitree G1 spinkick environment configuration."""
  # Start with the base tracking environment without state estimation.
  cfg = unitree_g1_flat_tracking_env_cfg(has_state_estimation=False, play=play)

  # Add custom spinkick termination.
  cfg.terminations["base_ang_vel_exceed"] = TerminationTermCfg(
    func=base_ang_vel_exceed, params={"threshold": _MAX_ANG_VEL}
  )

  return cfg


@dataclass
class SpinkickRunnerCfg(RslRlOnPolicyRunnerCfg):
  dense_save_iterations: Tuple[int, ...] = ()
  """Extra iterations at which to save checkpoints in addition to save_interval."""


def unitree_g1_spinkick_runner_cfg():
  """Create RL runner configuration for Unitree G1 spinkick task."""
  from mjlab.tasks.tracking.config.g1.rl_cfg import unitree_g1_tracking_ppo_runner_cfg

  base = unitree_g1_tracking_ppo_runner_cfg()
  cfg = SpinkickRunnerCfg(**{f.name: getattr(base, f.name) for f in base.__dataclass_fields__.values()})
  cfg.experiment_name = "g1_spinkick"
  cfg.actor.hidden_dims = [128, 128]
  cfg.actor.obs_normalization = False
  cfg.critic.obs_normalization = False
  return cfg
