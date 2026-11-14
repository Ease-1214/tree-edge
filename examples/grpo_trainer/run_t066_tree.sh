#!/usr/bin/env bash
set -euo pipefail
set +x
source "$HOME/.edge_reflector.env"
export T062_ROOT=/home/zehao/echo-collab/T066/tree
export PERF_TAG=formal PERF_QUESTIONS=256 PERF_STEPS=200 PERF_GPUS=4
export CUDA_VISIBLE_DEVICES=2,3,4,5
export EDGE_EMBED_URL=http://127.0.0.1:29700 EDGE_EMBED_CLIENT=t066-tree
export EDGE_REFLECTOR_PROMPT=search EDGE_TOKEN_SELECTION=teacher_topk EDGE_TOP_K=20
export ENABLE_EXPERIENCE_EVOLUTION=true EXP_ROLLOUT_WARMUP_STEPS=null
export T063_REFLECTOR_AUDIT=1
export WANDB_MODE=${WANDB_MODE:-online} T062_HTTP_PROXY=http://127.0.0.1:29662
export WANDB_RUN_ID=${WANDB_RUN_ID:-t066-tree-tier2-s0} WANDB_RESUME=must WANDB_RUN_GROUP=t066
export WANDB_NAME=t066-tree-tier2-s0 WANDB_TAGS=tree,tier2,beta0.3,full200
exec bash examples/grpo_trainer/run_t062_perf.sh \
 env.rollout.n=1 env.tree_experience.enable=True \
 env.tree_experience.utility_ema_beta=0.3 env.tree_experience.utility_update_interval=5 \
 env.tree_experience.min_delete_questions=20 env.tree_experience.protect_recent_steps=20 \
 env.experience_memory.update_frequency=5 \
 actor_rollout_ref.actor.edge_distillation.path_tier=2 \
 actor_rollout_ref.actor.edge_distillation.normalization=grpo_microbatch_tokens \
 actor_rollout_ref.actor.checkpoint.contents=[lora_adapter] \
 trainer.save_freq=10 trainer.project_name=tree-edge-search trainer.experiment_name=t066-tree-tier2-s0 \
 "$@"
