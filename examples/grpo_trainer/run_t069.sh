#!/usr/bin/env bash
set -euo pipefail
set +x
MODE=${1:?tree or edge}
shift
case "$MODE" in tree|edge) ;; *) exit 2 ;; esac
ROOT=/home/zehao/echo-collab/T069
SOURCE=/home/zehao/echo-collab/T066/EDGE
source "$HOME/.edge_reflector.env"
unset WANDB_SERVICE __WANDB_SERVICE WANDB_SERVICE_TRANSPORT EXPERIENCES_JSON
export EXPERIENCE_UPDATE_BASE_URL=http://127.0.0.1:29663/v1
export EXPERIENCE_UPDATE_MODEL=gpt-6.1-sol
export T062_ROOT=$ROOT/$MODE PERF_RUN_DIR=$ROOT/$MODE
export T069_RUN_DIR=$PERF_RUN_DIR
export T069_TRAIN_DATA=$ROOT/data/train.parquet T069_VAL_DATA=$ROOT/data/dev1000.parquet
export PERF_TAG=formal PERF_QUESTIONS=256 PERF_STEPS=200 PERF_GPUS=4
export CUDA_VISIBLE_DEVICES=2,3,4,5
export EDGE_EMBED_URL=http://127.0.0.1:29700 EDGE_EMBED_CLIENT=t069-$MODE
export EDGE_REFLECTOR_PROMPT=search EDGE_TOKEN_SELECTION=teacher_topk EDGE_TOP_K=20
export ENABLE_EXPERIENCE_EVOLUTION=true ENABLE_DYNAMIC_MANAGEMENT=true EXP_ROLLOUT_WARMUP_STEPS=null
export T063_REFLECTOR_AUDIT=1
export WANDB_MODE=online T062_HTTP_PROXY=http://127.0.0.1:29662
export WANDB_RUN_ID=t069-$MODE-s0-20261111 WANDB_RESUME=${WANDB_RESUME:-allow}
export WANDB_RUN_GROUP=t069 WANDB_NAME=t069-$MODE-s0 WANDB_TAGS=t069,$MODE,s0,full200
cd "$SOURCE"
OPTIONS=()
if [[ "$MODE" == tree ]]; then
  OPTIONS+=(env.rollout.n=1 env.tree_experience.enable=True
    env.tree_experience.utility_ema_beta=0.3 env.tree_experience.utility_update_interval=5
    env.tree_experience.min_delete_questions=50 env.tree_experience.protect_recent_steps=20
    env.tree_experience.dedup_shared_prefix=True +env.tree_experience.dedup_batching=node_mixed
    env.experience_memory.update_frequency=5
    actor_rollout_ref.actor.edge_distillation.path_tier=2
    actor_rollout_ref.actor.edge_distillation.normalization=grpo_microbatch_tokens)
fi
exec bash examples/grpo_trainer/run_t062_perf.sh \
  "${OPTIONS[@]}" \
  actor_rollout_ref.actor.checkpoint.contents=[lora_adapter] \
  trainer.save_freq=10 trainer.resume_mode=auto \
  trainer.project_name=tree-edge-search trainer.experiment_name=t069-$MODE-s0 \
  "$@"
