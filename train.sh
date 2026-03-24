#!/bin/bash

# Actor/Critic network hidden layer sizes (default: 512,256,128)
# ACTOR_DIMS="512,256,128"
ACTOR_DIMS="128,128"
CRITIC_DIMS="512,256,128"

# Training settings
NUM_ENVS=4096
MAX_ITER=20000
GPU_ID=3

MUJOCO_GL=egl CUDA_VISIBLE_DEVICES=$GPU_ID uv run train \
    Mjlab-Spinkick-Unitree-G1 \
    --registry-name huixuan_cheng-uc-berkeley-org/wandb-registry-Motions/mimickit_spinkick_safe \
    --env.scene.num-envs $NUM_ENVS \
    --agent.max-iterations $MAX_ITER \
    --agent.actor.hidden-dims $ACTOR_DIMS \
    --agent.critic.hidden-dims $CRITIC_DIMS \
    --agent.wandb-project spinkick \
    --agent.actor.obs-normalization False \
    --agent.critic.obs-normalization False \
    --env.commands.motion.sampling-mode uniform
