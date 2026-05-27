#!/bin/bash

# Actor/Critic network hidden layer sizes (default: 512,256,128)
# ACTOR_DIMS="512,256,128"
ACTOR_DIMS="128,128"
CRITIC_DIMS="512,256,128"

# Training settings
NUM_ENVS=4096
MAX_ITER=4500
GPU_ID=0
EXP_NAME="g1_spinkick_sweep_norm"

for SEED in $(seq 574 599); do
  echo "=== Training with seed $SEED ==="
  MUJOCO_GL=egl CUDA_VISIBLE_DEVICES=$GPU_ID uv run train \
      Mjlab-Spinkick-Unitree-G1 \
      --registry-name huixuan_cheng-uc-berkeley-org/wandb-registry-Motions/mimickit_spinkick_safe \
      --env.scene.num-envs $NUM_ENVS \
      --agent.max-iterations $MAX_ITER \
      --agent.seed $SEED \
      --agent.experiment-name $EXP_NAME \
      --agent.run-name "seed_${SEED}" \
      --agent.actor.hidden-dims $ACTOR_DIMS \
      --agent.critic.hidden-dims $CRITIC_DIMS \
      --agent.wandb-project spinkick_norm \
      --agent.actor.obs-normalization True \
      --agent.critic.obs-normalization True \
      --env.commands.motion.sampling-mode uniform \
      --agent.dense-save-iterations 4499,
done
