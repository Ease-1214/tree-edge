#!/usr/bin/env bash
# T70 dedicated experiment: four no-experience trajectories + conditional full tree.
# Fresh T70 run only. Never touches T069 historical training/checkpoints.
set -euo pipefail
set +x

ROOT=/home/zehao/echo-collab/T070
DATA_ROOT=/home/zehao/echo-collab/T069
SOURCE=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
RUN_DIR=${T70_RUN_DIR:-$ROOT/conditional-natural4-sft/v2-opsd-active-005}
case "$RUN_DIR" in
  "$DATA_ROOT/"*|"$ROOT/"tv2*|"$ROOT/"ev2*|"$ROOT/"ev3*)
    echo "Refusing to use an existing T069 run directory: $RUN_DIR" >&2; exit 2 ;;
esac
if [[ ! -f "$SOURCE/.git" && ! -d "$SOURCE/.git" ]]; then
  echo "Missing isolated worktree: $SOURCE" >&2; exit 2
fi
if [[ "$(git -C "$SOURCE" branch --show-current)" != "t70" ]]; then
  echo "Wrong branch; refusing to start" >&2; exit 2
fi
# Runtime-only credentials remain in their existing service environment.
source "$HOME/.edge_reflector.env"
unset WANDB_SERVICE __WANDB_SERVICE WANDB_SERVICE_TRANSPORT EXPERIENCES_JSON
export EXPERIENCE_UPDATE_BASE_URL=http://127.0.0.1:29663/v1
export EXPERIENCE_UPDATE_MODEL=gpt-6.1-sol
if [[ -e "$RUN_DIR" ]] && [[ -n "$(ls -A "$RUN_DIR")" ]]; then
  echo "Refusing to overwrite existing run directory: $RUN_DIR" >&2; exit 2
fi
export PERF_RUN_DIR="$RUN_DIR" T062_ROOT="$ROOT"
# Keep Ray AF_UNIX socket paths short while checkpoints remain in RUN_DIR.
# T70 validates in-process on the training GPUs. No independent eval queue.
export T069_TRAIN_DATA="$DATA_ROOT/data/train.parquet"
export T069_VAL_DATA="$DATA_ROOT/data/dev1000.parquet"
export PERF_TAG=conditional-natural4-sft PERF_QUESTIONS=256 PERF_STEPS=200 PERF_GPUS=4
export CUDA_VISIBLE_DEVICES=2,3,4,5
export EDGE_EMBED_URL=http://127.0.0.1:29700 EDGE_EMBED_CLIENT=t70-conditional-natural4
export EDGE_REFLECTOR_PROMPT=search EDGE_TOKEN_SELECTION=teacher_topk EDGE_TOP_K=20
export ENABLE_EXPERIENCE_EVOLUTION=true ENABLE_DYNAMIC_MANAGEMENT=true EXP_ROLLOUT_WARMUP_STEPS=null
export T063_REFLECTOR_AUDIT=1
export WANDB_MODE=online T062_HTTP_PROXY=http://127.0.0.1:29662
export WANDB_RUN_ID=${WANDB_RUN_ID:-t70-conditional-natural4-opsdactive005-v2-20261112}
export WANDB_RESUME=${WANDB_RESUME:-never}
export WANDB_RUN_GROUP=t70 WANDB_NAME=${WANDB_NAME:-t70-conditional-natural4-opsdactive005-v2}
export WANDB_TAGS=t70,conditional,natural4,opsd-active-token,sft005,opsd005,v2
cd "$SOURCE"
exec bash examples/grpo_trainer/run_t062_perf.sh \
  actor_rollout_ref.actor.optim.lr=2e-5 \
  actor_rollout_ref.actor.optim.lr_warmup_steps=5 \
  actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.0 \
  env.rollout.n=1 \
  env.tree_experience.enable=True \
  env.tree_experience.conditional_natural.enable=True \
  env.tree_experience.conditional_natural.natural_rollouts=4 \
  env.tree_experience.utility_ema_beta=0.1 \
  env.tree_experience.utility_update_interval=5 \
  env.experience_memory.management.eviction_utility_threshold=-0.1 \
  env.tree_experience.protect_recent_steps=20 \
  env.tree_experience.dedup_shared_prefix=True \
  +env.tree_experience.dedup_batching=node_mixed \
  env.experience_memory.update_frequency=5 \
  actor_rollout_ref.actor.success_sft.enable=True \
  actor_rollout_ref.actor.success_sft.coef=0.05 \
  actor_rollout_ref.actor.edge_distillation.coef=0.05 \
  actor_rollout_ref.actor.edge_distillation.path_tier=2 \
  actor_rollout_ref.actor.edge_distillation.normalization=active_kd_tokens \
  actor_rollout_ref.actor.checkpoint.contents='[lora_adapter]' \
  trainer.save_freq=10 trainer.resume_mode=auto \
  trainer.val_before_train=True trainer.test_freq=10 \
  data.val_batch_size=128 \
  +trainer.t70_test6125_file="$DATA_ROOT/data/test6125.parquet" \
  +trainer.t70_test6125_freq=50 \
  trainer.project_name=tree-edge-search \
  trainer.experiment_name="$WANDB_NAME" \
  "$@"
