#!/usr/bin/env bash
# ==============================================================================
# T75 Unified Multi-Hardware Training Entrypoint (Qwen2.5-3B-Instruct + Tree-EDGE)
# ==============================================================================
# Supports 3 hardware profiles out of the box via GPU_PROFILE:
#   - 2xa40  : 2x NVIDIA A40 (48GB VRAM each)
#   - 4x3090 : 4x NVIDIA RTX 3090 (24GB VRAM each)
#   - 8x3090 : 8x NVIDIA RTX 3090 (24GB VRAM each)
#   - auto   : Auto-detects GPU model (A40 vs 3090) and visible GPU count
#
# Quick Start Examples:
#   1) 2x A40 (GPUs 0,1):
#      GPU_PROFILE=2xa40 CUDA_VISIBLE_DEVICES=0,1 bash examples/grpo_trainer/run_t75.sh
#
#   2) 4x RTX 3090 (GPUs 0,1,2,3):
#      GPU_PROFILE=4x3090 CUDA_VISIBLE_DEVICES=0,1,2,3 bash examples/grpo_trainer/run_t75.sh
#
#   3) 8x RTX 3090 (GPUs 0..7):
#      GPU_PROFILE=8x3090 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash examples/grpo_trainer/run_t75.sh
#
#   4) Resume from a specific checkpoint:
#      GPU_PROFILE=4x3090 RESUME_PATH=/path/to/checkpoints/global_step_20 bash examples/grpo_trainer/run_t75.sh
# ==============================================================================
set -euo pipefail
set +x

SOURCE=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
CURRENT_BRANCH="$(git -C "$SOURCE" branch --show-current 2>/dev/null || echo unknown)"
if [[ "$CURRENT_BRANCH" != "t75" && "${ALLOW_NON_T75_BRANCH:-0}" != "1" ]]; then
  echo "Expected branch 't75' (got '$CURRENT_BRANCH'). Set ALLOW_NON_T75_BRANCH=1 to override." >&2
  exit 2
fi

# ------------------------------------------------------------------------------
# 1. Hardware Profile Detection & Per-GPU Memory / Token Budget Tuning
# ------------------------------------------------------------------------------
GPU_PROFILE=${GPU_PROFILE:-auto}

if [[ "$GPU_PROFILE" == "auto" ]]; then
  if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    IFS=',' read -ra _VIS_DEV <<< "$CUDA_VISIBLE_DEVICES"
    DETECTED_GPUS=${#_VIS_DEV[@]}
  elif command -v nvidia-smi >/dev/null 2>&1; then
    DETECTED_GPUS=$(nvidia-smi -L | wc -l)
  else
    DETECTED_GPUS=2
  fi
  GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -n 1 || echo "A40")
  if [[ "$GPU_NAME" == *"3090"* ]]; then
    if (( DETECTED_GPUS >= 8 )); then
      GPU_PROFILE="8x3090"
    else
      GPU_PROFILE="4x3090"
    fi
  else
    GPU_PROFILE="2xa40"
  fi
  echo "[T75] Auto-detected GPU_PROFILE=$GPU_PROFILE (GPU='$GPU_NAME', visible=$DETECTED_GPUS)"
fi

case "$GPU_PROFILE" in
  2xa40)
    # 2x A40 (48GB VRAM each): ample VRAM allows larger token budgets and vLLM KV cache
    export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1}
    export PERF_GPUS=${PERF_GPUS:-2}
    export PERF_FSDP_SIZE=${PERF_FSDP_SIZE:-1}
    export TENSOR_MODEL_PARALLEL_SIZE=1
    export PERF_MODEL_DTYPE=bf16
    export PERF_DYNAMIC_BSZ=1
    export PERF_VLLM_UTIL=${PERF_VLLM_UTIL:-0.48}
    export PERF_EAGER=${PERF_EAGER:-0}
    export PERF_PPO_TOKENS=${PERF_PPO_TOKENS:-24576}
    export PERF_LOGPROB_TOKENS=${PERF_LOGPROB_TOKENS:-49152}
    export PERF_VLLM_BATCHED_TOKENS=${PERF_VLLM_BATCHED_TOKENS:-16384}
    PPO_MICRO_BSZ=${PPO_MICRO_BSZ:-4}
    LOGPROB_MICRO_BSZ=${LOGPROB_MICRO_BSZ:-8}
    MAX_NUM_SEQS=${MAX_NUM_SEQS:-512}
    ;;
  4x3090)
    # 4x RTX 3090 (24GB VRAM each):
    # - Base Qwen2.5-3B (bf16, fsdp_size=1) + FP32 LoRA r64 + Adam states = ~8.0 GB
    # - Cap vLLM at 0.38 (~9.1 GB during rollout; sleeps during actor update)
    # - Cap actor dynamic token budget at 12288 tokens/microbatch (~2 rows of 5700 tokens)
    #   so Teacher + Student + SFT forward/backward never exceeds ~18.5 GB on 24GB cards.
    export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
    export PERF_GPUS=4
    export PERF_FSDP_SIZE=${PERF_FSDP_SIZE:-1}
    export TENSOR_MODEL_PARALLEL_SIZE=1
    export PERF_MODEL_DTYPE=bf16
    export PERF_DYNAMIC_BSZ=1
    export PERF_VLLM_UTIL=${PERF_VLLM_UTIL:-0.38}
    export PERF_EAGER=${PERF_EAGER:-0}
    export PERF_PPO_TOKENS=${PERF_PPO_TOKENS:-12288}
    export PERF_LOGPROB_TOKENS=${PERF_LOGPROB_TOKENS:-24576}
    export PERF_VLLM_BATCHED_TOKENS=${PERF_VLLM_BATCHED_TOKENS:-12288}
    PPO_MICRO_BSZ=${PPO_MICRO_BSZ:-2}
    LOGPROB_MICRO_BSZ=${LOGPROB_MICRO_BSZ:-4}
    MAX_NUM_SEQS=${MAX_NUM_SEQS:-256}
    export EDGE_EMBED_MEM_GB=${EDGE_EMBED_MEM_GB:-3}
    ;;
  8x3090)
    # 8x RTX 3090 (24GB VRAM each):
    # Same safe 24GB per-card memory envelope as 4x3090, with 2x data-parallel throughput.
    export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
    export PERF_GPUS=8
    export PERF_FSDP_SIZE=${PERF_FSDP_SIZE:-1}
    export TENSOR_MODEL_PARALLEL_SIZE=1
    export PERF_MODEL_DTYPE=bf16
    export PERF_DYNAMIC_BSZ=1
    export PERF_VLLM_UTIL=${PERF_VLLM_UTIL:-0.38}
    export PERF_EAGER=${PERF_EAGER:-0}
    export PERF_PPO_TOKENS=${PERF_PPO_TOKENS:-12288}
    export PERF_LOGPROB_TOKENS=${PERF_LOGPROB_TOKENS:-24576}
    export PERF_VLLM_BATCHED_TOKENS=${PERF_VLLM_BATCHED_TOKENS:-12288}
    PPO_MICRO_BSZ=${PPO_MICRO_BSZ:-2}
    LOGPROB_MICRO_BSZ=${LOGPROB_MICRO_BSZ:-4}
    MAX_NUM_SEQS=${MAX_NUM_SEQS:-256}
    export EDGE_EMBED_MEM_GB=${EDGE_EMBED_MEM_GB:-3}
    ;;
  *)
    echo "Unsupported GPU_PROFILE='$GPU_PROFILE'. Choose from: 2xa40, 4x3090, 8x3090, auto" >&2
    exit 2
    ;;
esac

# ------------------------------------------------------------------------------
# 2. Paths, Environment & Reflector API Preflight
# ------------------------------------------------------------------------------
ROOT=${T75_ROOT:-/home/zehao/echo-collab/T075}
DATA_ROOT=${DATA_ROOT:-/home/zehao/echo-collab/T069}
VAL_ROOT=${VAL_ROOT:-/home/zehao/echo-collab/T073}

export PERF_TAG=${PERF_TAG:-t75-${GPU_PROFILE}-r64}
RUN_DIR=${T75_RUN_DIR:-$ROOT/${PERF_TAG}-$(date +%Y%m%d-%H%M%S)}
export PERF_RUN_DIR="$RUN_DIR"
export T062_ROOT="$ROOT"

export TMPDIR="$ROOT/tmp" XDG_CACHE_HOME="$ROOT/cache" TORCH_EXTENSIONS_DIR="$ROOT/torch_extensions"
if [[ -z "${TRITON_LIBCUDA_PATH:-}" && -d "/home/zehao/echo-collab/T073/cuda-driver-link" ]]; then
  export TRITON_LIBCUDA_PATH="/home/zehao/echo-collab/T073/cuda-driver-link"
fi
export RAY_TMPDIR=${RAY_TMPDIR:-/dev/shm/t75-${GPU_PROFILE}-ray}
mkdir -p "$RUN_DIR" "$TMPDIR" "$XDG_CACHE_HOME" "$TORCH_EXTENSIONS_DIR" "$RAY_TMPDIR"
chmod 700 "$RAY_TMPDIR"

if [[ -f "$HOME/.edge_reflector.env" ]]; then
  source "$HOME/.edge_reflector.env"
fi
unset WANDB_SERVICE __WANDB_SERVICE WANDB_SERVICE_TRANSPORT EXPERIENCES_JSON
unset ALL_PROXY all_proxy

export EXPERIENCE_UPDATE_BASE_URL=${EXPERIENCE_UPDATE_BASE_URL:-http://127.0.0.1:29663/v1}
export EXPERIENCE_UPDATE_MODEL=${EXPERIENCE_UPDATE_MODEL:-gpt-6.1-sol}
export ENABLE_EXPERIENCE_EVOLUTION=${ENABLE_EXPERIENCE_EVOLUTION:-true}
export ENABLE_DYNAMIC_MANAGEMENT=${ENABLE_DYNAMIC_MANAGEMENT:-true}
export EXP_ROLLOUT_WARMUP_STEPS=null
# Default to 1 single LLM API call per reflection window (set to 1 to enable 2nd-pass type validator)
export ENABLE_EXPERIENCE_TYPE_VALIDATION=${ENABLE_EXPERIENCE_TYPE_VALIDATION:-0}
export T063_REFLECTOR_AUDIT=1

# Fail fast BEFORE allocating GPUs if Reflector client cannot initialize
if [[ "$ENABLE_EXPERIENCE_EVOLUTION" == "true" ]]; then
  PYTHON_BIN=${PYTHON_BIN:-/home/zehao/.conda/envs/tree-edge/bin/python}
  if [[ -x "$PYTHON_BIN" ]]; then
    PYTHONPATH="$SOURCE${PYTHONPATH:+:$PYTHONPATH}" "$PYTHON_BIN" - <<'PY'
from agent_system.memory.experience_updater import ExperienceUpdater
client = ExperienceUpdater(max_new_experiences_per_update=5)
client.client.close()
print("T75_REFLECTOR_CLIENT_PREFLIGHT_OK", flush=True)
PY
  fi
fi

# ------------------------------------------------------------------------------
# 3. Dataset, Model, Embedding & W&B Settings
# ------------------------------------------------------------------------------
export MODEL_PATH=${MODEL_PATH:-/home/zehao/Tree-GRPO/model/Qwen2.5-3B-Instruct}
export T069_TRAIN_DATA=${TRAIN_DATA:-$DATA_ROOT/data/train.parquet}
export T069_VAL_DATA=${VAL_DATA:-$VAL_ROOT/data/nq_hotpotqa_full11015.parquet}
export EXPERIENCES_JSON=${EXPERIENCES_JSON:-$SOURCE/memory_data/search/state_aware_experiences_v2.json}

# If a shared embedding service is running on 29700, use it; otherwise fall back to local GPU embedding
export EDGE_EMBED_URL=${EDGE_EMBED_URL:-http://127.0.0.1:29700}
export EDGE_EMBED_CLIENT=${EDGE_EMBED_CLIENT:-t75-${GPU_PROFILE}}
export EDGE_REFLECTOR_PROMPT=search
export EDGE_TOKEN_SELECTION=teacher_topk
export EDGE_TOP_K=20

export PERF_QUESTIONS=${PERF_QUESTIONS:-256}
export PERF_STEPS=${PERF_STEPS:-200}

export WANDB_MODE=${WANDB_MODE:-online}
if [[ "$WANDB_MODE" != "offline" && -z "${T062_HTTP_PROXY:-}" ]]; then
  _wandb_proxy=""
  for _cand in "http://127.0.0.1:7890" "http://127.0.0.1:29662"; do
    _code=$(curl -sS -o /dev/null -w '%{http_code}' --max-time 3 --proxy "$_cand" https://api.wandb.ai/ 2>/dev/null || echo "000")
    if [[ "$_code" == "200" || "$_code" == "301" || "$_code" == "302" || "$_code" == "404" ]]; then
      _wandb_proxy="$_cand"
      break
    fi
  done
  if [[ -n "$_wandb_proxy" ]]; then
    export T062_HTTP_PROXY="$_wandb_proxy"
  else
    _direct_code=$(curl -sS -o /dev/null -w '%{http_code}' --max-time 3 https://api.wandb.ai/ 2>/dev/null || echo "000")
    if [[ "$_direct_code" != "200" && "$_direct_code" != "404" ]]; then
      export WANDB_MODE=offline
      echo "[T75] No working W&B proxy found; falling back to WANDB_MODE=offline" >&2
    fi
  fi
fi
export T062_HTTP_PROXY=${T062_HTTP_PROXY:-http://127.0.0.1:29662}
if [[ -z "${WANDB_BASE_URL:-}" && -f "$HOME/.netrc" ]]; then
  if ! grep -q "api\.forge\.coreweave\.com" "$HOME/.netrc" && grep -q "api\.wandb\.ai" "$HOME/.netrc"; then
    export WANDB_BASE_URL="https://api.wandb.ai"
  fi
fi
export T72_WANDB_STARTUP_HEARTBEAT=1
export WANDB_DIR="$RUN_DIR"
export WANDB_RUN_GROUP=${WANDB_RUN_GROUP:-t75}
export WANDB_NAME=${WANDB_NAME:-t75-qwen25-3b-r64a128-${GPU_PROFILE}}
export WANDB_TAGS=${WANDB_TAGS:-t75,qwen25-3b,r64,a128,${GPU_PROFILE},tree-edge}

# Optional checkpoint resume support
RESUME_ARGS=()
if [[ -n "${RESUME_PATH:-}" ]]; then
  RESUME_ARGS+=(
    "trainer.resume_mode=resume_path"
    "trainer.resume_from_path=$RESUME_PATH"
  )
else
  RESUME_ARGS+=("trainer.resume_mode=auto")
fi

echo "=============================================================================="
echo "[T75 Launch Summary]"
echo "  GPU_PROFILE          : $GPU_PROFILE (GPUs=$PERF_GPUS, CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES)"
echo "  vLLM Util / Batched  : $PERF_VLLM_UTIL / $PERF_VLLM_BATCHED_TOKENS tokens (max_seqs=$MAX_NUM_SEQS)"
echo "  Actor / LogProb Toks : $PERF_PPO_TOKENS / $PERF_LOGPROB_TOKENS tokens per GPU"
echo "  Run Directory        : $RUN_DIR"
echo "  Experience Bank      : $EXPERIENCES_JSON"
echo "  Type Validator Call  : $ENABLE_EXPERIENCE_TYPE_VALIDATION (0 = single-call, 1 = two-call)"
echo "=============================================================================="

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
  env.experience_memory.experiences_json_path="$EXPERIENCES_JSON" \
  env.experience_memory.top_k=12 \
  env.experience_memory.task_specific_top_k=12 \
  env.experience_memory.similarity_threshold=0.3 \
  env.experience_memory.step_similarity_threshold=0.3 \
  env.experience_memory.max_new_experiences=5 \
  env.experience_memory.management.retrieval_alpha=0.5 \
  env.experience_memory.management.retrieval_ucb_c=0.2 \
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
  trainer.save_freq=10 \
  "${RESUME_ARGS[@]}" \
  trainer.val_before_train=False \
  trainer.test_freq=50 \
  data.val_batch_size=128 \
  trainer.project_name=tree-edge-search \
  trainer.experiment_name="$WANDB_NAME" \
  actor_rollout_ref.model.path="$MODEL_PATH" \
  actor_rollout_ref.model.lora_rank=64 \
  actor_rollout_ref.model.lora_alpha=128 \
  actor_rollout_ref.actor.ppo_mini_batch_size=256 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="$PPO_MICRO_BSZ" \
  actor_rollout_ref.rollout.max_num_seqs="$MAX_NUM_SEQS" \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu="$LOGPROB_MICRO_BSZ" \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu="$LOGPROB_MICRO_BSZ" \
  "$@"
