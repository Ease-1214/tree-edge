#!/usr/bin/env bash
set -euo pipefail
set +x
source "$HOME/.edge_reflector.env"
export T062_ROOT=/home/zehao/echo-collab/T065
export PERF_TAG=${PERF_TAG:-merged-smoke}
export PERF_QUESTIONS=64 PERF_STEPS=3 PERF_GPUS=4
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
export EDGE_EMBED_URL=http://127.0.0.1:29700
export EDGE_EMBED_CLIENT=t065-smoke
export EDGE_EMBED_REQUEST_LOG=/home/zehao/echo-collab/T065/embedding-requests-private.jsonl
export EDGE_REFLECTOR_PROMPT=search EDGE_TOKEN_SELECTION=teacher_topk EDGE_TOP_K=20
export ENABLE_EXPERIENCE_EVOLUTION=true
export EXP_ROLLOUT_WARMUP_STEPS=2
export WANDB_MODE=offline T062_HTTP_PROXY=http://127.0.0.1:29662
export WANDB_NAME=t065-merged-smoke
exec bash examples/grpo_trainer/run_t062_perf.sh \
 env.rollout.n=1 \
 env.tree_experience.enable=True \
 env.tree_experience.utility_update_interval=1 \
 env.tree_experience.min_delete_questions=2 \
 env.tree_experience.protect_recent_steps=0 \
 env.experience_memory.update_frequency=1 \
 actor_rollout_ref.actor.edge_distillation.path_tier=2 \
 actor_rollout_ref.actor.edge_distillation.normalization=grpo_microbatch_tokens \
 trainer.save_freq=2 actor_rollout_ref.actor.checkpoint.contents=[lora_adapter] \
 trainer.project_name=tree-edge-t065 trainer.experiment_name=t065-merged-smoke \
 "$@"
