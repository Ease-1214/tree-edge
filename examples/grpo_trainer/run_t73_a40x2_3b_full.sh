#!/usr/bin/env bash
# T73 A40x2 Qwen2.5-3B-Instruct full run, T71-V2 runtime config + rank64/alpha128 + 2 A40 GPUs.
# T73 latest Fork + EDGE extraction code is retained; T71 actor/rollout settings restored.
# Fresh isolated run; never overwrites historical training/checkpoints.
set -euo pipefail
set +x

ROOT=/home/zehao/echo-collab/T073
export TMPDIR="$ROOT/tmp" XDG_CACHE_HOME="$ROOT/cache" TORCH_EXTENSIONS_DIR="$ROOT/torch_extensions"
export TRITON_LIBCUDA_PATH="$ROOT/cuda-driver-link"
export RAY_TMPDIR=/dev/shm/t73-31-ray
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME" "$TORCH_EXTENSIONS_DIR" "$RAY_TMPDIR"
chmod 700 "$RAY_TMPDIR"
T70_ROOT=/home/zehao/echo-collab/T070
DATA_ROOT=/home/zehao/echo-collab/T069
SOURCE=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
RUN_DIR=${T73_RUN_DIR:-$ROOT/qwen25-3b-r64a128-a40x2-full11015-20261115}
case "$RUN_DIR" in
  "$DATA_ROOT/"*|"$T70_ROOT/"*|"$ROOT/"tv2*|"$ROOT/"ev2*|"$ROOT/"ev3*)
    echo "Refusing to use a historical run directory: $RUN_DIR" >&2; exit 2 ;;
esac
if [[ ! -f "$SOURCE/.git" && ! -d "$SOURCE/.git" ]]; then
  echo "Missing isolated worktree: $SOURCE" >&2; exit 2
fi
CURRENT_BRANCH="$(git -C "$SOURCE" branch --show-current)"
if [[ "$CURRENT_BRANCH" != "t75" && "$CURRENT_BRANCH" != "fix/t73-reflector-type-validation" ]]; then
  echo "Wrong branch ($CURRENT_BRANCH); refusing to start" >&2; exit 2
fi
# Runtime-only credentials remain in their existing service environment.
source "$HOME/.edge_reflector.env"
unset WANDB_SERVICE __WANDB_SERVICE WANDB_SERVICE_TRANSPORT EXPERIENCES_JSON
# T71 reflector uses a local HTTP endpoint, while the inherited ALL_PROXY may
# be socks5h:// and httpx2 has no socksio. Explicit HTTP(S)_PROXY remains
# available for external services; remove only the global SOCKS fallback.
unset ALL_PROXY all_proxy
export EXPERIENCE_UPDATE_BASE_URL=http://127.0.0.1:29663/v1
export EXPERIENCE_UPDATE_MODEL=gpt-6.1-sol
# Fail fast BEFORE allocating 8 GPUs: the previous run crashed at Step 5
# because OpenAI -> httpx2 tried to initialize an unavailable SOCKS transport.
PYTHONPATH="$SOURCE${PYTHONPATH:+:$PYTHONPATH}" \
  /home/zehao/.conda/envs/tree-edge/bin/python - <<'PY'
from agent_system.memory.experience_updater import ExperienceUpdater
client = ExperienceUpdater(max_new_experiences_per_update=5)
client.client.close()
print("T73_REFLECTOR_CLIENT_PREFLIGHT_OK", flush=True)
PY
# A new, isolated run directory is mandatory. Never overwrite live T73.
ORIGINAL_RUN_DIR="$ROOT/qwen25-3b-r64a128-a40x2-full11015-20261115"
if [[ "$RUN_DIR" == "$ORIGINAL_RUN_DIR" ]]; then
  echo "Original run directory protected; specify isolated T73_RUN_DIR" >&2; exit 2
fi
if [[ "$T73_SAFE_RESUME" != "1" ||
      ! -f "$RUN_DIR/checkpoints/global_step_20/tree_state.json" ||
      ! -f "$RUN_DIR/checkpoints/global_step_20/actor/training_state_rank_0.pt" ||
      ! -f "$RUN_DIR/checkpoints/global_step_20/actor/training_state_rank_1.pt" ||
      "$(cat "$RUN_DIR/checkpoints/latest_checkpointed_iteration.txt" 2>/dev/null)" != "20" ]]; then
  echo "Missing verified isolated Step20 checkpoint; refusing to start" >&2; exit 2
fi
export PERF_RUN_DIR="$RUN_DIR" T062_ROOT="$ROOT"
# Keep Ray AF_UNIX socket paths short while checkpoints remain in RUN_DIR.
# T70 validates in-process on the training GPUs. No independent eval queue.
export T069_TRAIN_DATA="$DATA_ROOT/data/train.parquet"
export T069_VAL_DATA="$ROOT/data/nq_hotpotqa_full11015.parquet"
export PERF_TAG=t73-a40x2-r64-full11015 PERF_QUESTIONS=256 PERF_STEPS=200 PERF_GPUS=2
export CUDA_VISIBLE_DEVICES=2,3
# T71-V2 replica FSDP size 1 on each A40 (CUDA_VISIBLE_DEVICES=2,3); vLLM TP=1.
# Preserve BF16 base / FP32 LoRA without any 4/8-bit quantization.
export PERF_FSDP_SIZE=1 TENSOR_MODEL_PARALLEL_SIZE=1 PERF_MODEL_DTYPE=bf16 PERF_DYNAMIC_BSZ=1
# Proposed high-throughput A40 profile; verify with one full Step after approval.
export PERF_VLLM_UTIL=0.48 PERF_EAGER=0
export PERF_PPO_TOKENS=24576 PERF_LOGPROB_TOKENS=49152 PERF_VLLM_BATCHED_TOKENS=16384
export EDGE_EMBED_URL=http://127.0.0.1:29700 EDGE_EMBED_CLIENT=t73-qwen25-3b-a40x2-full
export EDGE_REFLECTOR_PROMPT=search EDGE_TOKEN_SELECTION=teacher_topk EDGE_TOP_K=20
export ENABLE_EXPERIENCE_EVOLUTION=true ENABLE_DYNAMIC_MANAGEMENT=true EXP_ROLLOUT_WARMUP_STEPS=null
export T063_REFLECTOR_AUDIT=1
export WANDB_MODE=online T062_HTTP_PROXY=http://127.0.0.1:29662
export T72_WANDB_STARTUP_HEARTBEAT=1
# W&B state belongs to this specific experiment, not the shared source worktree.
export WANDB_DIR="$RUN_DIR"
mkdir -p "$WANDB_DIR"
export WANDB_RUN_ID=${WANDB_RUN_ID:-t73-qwen25-3b-r64a128-a40x2-full11015-20261115}
export WANDB_RESUME=${WANDB_RESUME:-must}
export WANDB_RUN_GROUP=t73 WANDB_NAME=${WANDB_NAME:-t73-qwen25-3b-r64a128-a40x2-full11015}
export WANDB_TAGS=t73,qwen25-3b,r64,a128,two-a40,t71replica,eval50,edge-single-stage-extraction,conditional,natural4,opsd005,sft005
cd "$SOURCE"
exec bash examples/grpo_trainer/run_t062_perf.sh \
  actor_rollout_ref.actor.optim.lr=2e-5 \
  actor_rollout_ref.actor.optim.lr_warmup_steps=5 \
  actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.0 \
  actor_rollout_ref.actor.invalid_action_penalty_coef=0.5 \
  env.rollout.n=1 \
  env.tree_experience.enable=True \
  env.tree_experience.conditional_natural.enable=True \
  env.tree_experience.conditional_natural.natural_rollouts=4 \
  env.tree_experience.task_top_k=12 \
  env.tree_experience.step_top_k=12 \
  +env.tree_experience.semantic_candidate_k=12 \
  env.experience_memory.top_k=12 \
  env.experience_memory.task_specific_top_k=12 \
  env.experience_memory.similarity_threshold=0.4 \
  env.experience_memory.max_new_experiences=5 \
  env.experience_memory.management.retrieval_alpha=0.5 \
  env.experience_memory.management.retrieval_ucb_c=0.5 \
  env.tree_experience.utility_ema_beta=0.1 \
  env.tree_experience.utility_update_interval=5 \
  env.experience_memory.management.eviction_utility_threshold=-0.05 \
  env.experience_memory.management.eviction_min_exposures=20 \
  env.experience_memory.management.eviction_negative_windows=2 \
  env.experience_memory.management.eviction_protect_recent_steps=0 \
  env.experience_memory.management.eviction_max_task_experiences=200 \
  env.experience_memory.management.eviction_max_step_experiences=200 \
  env.tree_experience.protect_recent_steps=0 \
  env.tree_experience.dedup_shared_prefix=True \
  +env.tree_experience.dedup_batching=node_mixed \
  env.experience_memory.update_frequency=5 \
  actor_rollout_ref.actor.success_sft.enable=True \
  actor_rollout_ref.actor.success_sft.coef=0.05 \
  actor_rollout_ref.actor.edge_distillation.coef=0.05 \
  actor_rollout_ref.actor.edge_distillation.path_tier=2 \
  actor_rollout_ref.actor.edge_distillation.normalization=active_kd_tokens \
  actor_rollout_ref.actor.checkpoint.contents='[lora_adapter]' \
  trainer.save_freq=10 trainer.resume_mode=resume_path trainer.resume_from_path="$RUN_DIR/checkpoints/global_step_20" \
  trainer.val_before_train=False trainer.test_freq=50 \
  data.val_batch_size=128 \
  trainer.project_name=tree-edge-search \
  trainer.experiment_name="$WANDB_NAME" \
  actor_rollout_ref.model.path=/home/zehao/Tree-GRPO/model/Qwen2.5-3B-Instruct \
  actor_rollout_ref.model.lora_rank=64 \
  actor_rollout_ref.model.lora_alpha=128 \
  actor_rollout_ref.actor.ppo_mini_batch_size=256 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
  actor_rollout_ref.rollout.max_num_seqs=512 \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=8 \
  "$@"