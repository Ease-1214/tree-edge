#!/usr/bin/env bash
# Verified Pure GRPO Baseline (G=8, fixed_num_minibatches=6 -> exact 6 updates/step)
# Strictly aligned with T75 main run: Qwen2.5-3B/7B LoRA r=64 alpha=128, lr=2e-5,
# bsz=256, max_steps=4, invalid_action_penalty=0.5, exact 6 PPO updates/step.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

export ALLOW_NON_T75_BRANCH=${ALLOW_NON_T75_BRANCH:-1}
export GPU_PROFILE=${GPU_PROFILE:-4x3090}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
export MODEL_PATH=${MODEL_PATH:-/home/zehao/Tree-GRPO/model/Qwen2.5-3B-Instruct}
export T75_RUN_DIR=${T75_RUN_DIR:-/home/zehao/echo-collab/T075/t75-ablation-grpo-n8-3b-${GPU_PROFILE}-$(date +%Y%m%d-%H%M%S)}
export WANDB_NAME=${WANDB_NAME:-t75-3b-baseline-grpo-n8-fix6-${GPU_PROFILE}}
export WANDB_TAGS=${WANDB_TAGS:-t75,ablation,grpo-baseline,n8,fixed6mini,qwen25-3b,r64,a128,${GPU_PROFILE}}
export SEARCH_URL=${SEARCH_URL:-http://127.0.0.1:29661/retrieve}
export EDGE_EMBED_URL=${EDGE_EMBED_URL:-http://127.0.0.1:29700}
export ENABLE_EXPERIENCE_EVOLUTION=false
export ENABLE_DYNAMIC_MANAGEMENT=false
export WANDB_MODE=${WANDB_MODE:-online}
export WANDB_BASE_URL=${WANDB_BASE_URL:-https://api.wandb.ai}
export T062_HTTP_PROXY=${T062_HTTP_PROXY:-http://127.0.0.1:29662}

exec bash examples/grpo_trainer/run_t75.sh \
  env.tree_experience.enable=False \
  env.use_experience_memory=False \
  env.experience_memory.enable_exp_rollout=False \
  env.rollout.n=8 \
  +actor_rollout_ref.actor.fixed_num_minibatches=6 \
  actor_rollout_ref.actor.success_sft.enable=False \
  actor_rollout_ref.actor.success_sft.coef=0.0 \
  actor_rollout_ref.actor.edge_distillation.enable=False \
  actor_rollout_ref.actor.edge_distillation.coef=0.0 \
  "$@"
