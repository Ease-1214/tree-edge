#!/usr/bin/env bash
# T73 eight-RTX3090 Qwen2.5-3B-Instruct memory-fit smoke, based on T72 latest config.
# Same latest Fork/EDGE/GRPO/OPSD/SFT settings and LoRA rank/alpha; change 7B to 3B and six to eight ranks.
# Fresh isolated run; never overwrites historical training/checkpoints.
set -euo pipefail
set +x

ROOT=/home/zehao/echo-collab/T073
T70_ROOT=/home/zehao/echo-collab/T070
DATA_ROOT=/home/zehao/echo-collab/T069
SOURCE=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
RUN_DIR=${T73_RUN_DIR:-$ROOT/qwen25-3b-r64a128-eightgpu-fixedmicro-smoke-20261114}
case "$RUN_DIR" in
  "$DATA_ROOT/"*|"$T70_ROOT/"*|"$ROOT/"tv2*|"$ROOT/"ev2*|"$ROOT/"ev3*)
    echo "Refusing to use a historical run directory: $RUN_DIR" >&2; exit 2 ;;
esac
if [[ ! -f "$SOURCE/.git" && ! -d "$SOURCE/.git" ]]; then
  echo "Missing isolated worktree: $SOURCE" >&2; exit 2
fi
if [[ "$(git -C "$SOURCE" branch --show-current)" != "t73-qwen25-3b-r64a128-8gpu" ]]; then
  echo "Wrong branch; refusing to start" >&2; exit 2
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
client = ExperienceUpdater(max_new_experiences_per_update=3)
client.client.close()
print("T73_REFLECTOR_CLIENT_PREFLIGHT_OK", flush=True)
PY
if [[ -e "$RUN_DIR" ]] && [[ -n "$(ls -A "$RUN_DIR")" ]]; then
  echo "Refusing to overwrite existing run directory: $RUN_DIR" >&2; exit 2
fi
export PERF_RUN_DIR="$RUN_DIR" T062_ROOT="$ROOT"
# Keep Ray AF_UNIX socket paths short while checkpoints remain in RUN_DIR.
# T70 validates in-process on the training GPUs. No independent eval queue.
export T069_TRAIN_DATA="$DATA_ROOT/data/train.parquet"
export T069_VAL_DATA="$DATA_ROOT/data/dev1000.parquet"
export PERF_TAG=t73-qwen25-3b-eightgpu-fixedmicro-smoke PERF_QUESTIONS=16 PERF_STEPS=1 PERF_GPUS=8
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
# 24GB RTX3090: FSDP FULL_SHARD across eight ranks and vLLM TP=2.
# Preserve BF16 base / FP32 LoRA without any 4/8-bit quantization.
export PERF_FSDP_SIZE=-1 TENSOR_MODEL_PARALLEL_SIZE=2 PERF_MODEL_DTYPE=bf16 PERF_DYNAMIC_BSZ=0
export PERF_VLLM_UTIL=0.65 PERF_EAGER=1
export PERF_PPO_TOKENS=6144 PERF_LOGPROB_TOKENS=8192 PERF_VLLM_BATCHED_TOKENS=8192
export EDGE_EMBED_URL=http://127.0.0.1:29700 EDGE_EMBED_CLIENT=t73-qwen25-3b-eightgpu-smoke
export EDGE_REFLECTOR_PROMPT=search EDGE_TOKEN_SELECTION=teacher_topk EDGE_TOP_K=20
export ENABLE_EXPERIENCE_EVOLUTION=true ENABLE_DYNAMIC_MANAGEMENT=true EXP_ROLLOUT_WARMUP_STEPS=null
export T063_REFLECTOR_AUDIT=1
export WANDB_MODE=online T062_HTTP_PROXY=http://127.0.0.1:29662
export T72_WANDB_STARTUP_HEARTBEAT=1
# W&B state belongs to this specific experiment, not the shared source worktree.
export WANDB_DIR="$RUN_DIR"
mkdir -p "$WANDB_DIR"
export WANDB_RUN_ID=${WANDB_RUN_ID:-t73-qwen25-3b-r64a128-eightgpu-fixedmicro-smoke-20261114}
export WANDB_RESUME=${WANDB_RESUME:-never}
export WANDB_RUN_GROUP=t73 WANDB_NAME=${WANDB_NAME:-t73-qwen25-3b-r64a128-eightgpu-fixedmicro-smoke}
export WANDB_TAGS=t73,qwen25-3b,r64,a128,eight-rtx3090,smoke,edge-single-stage-extraction,conditional,natural4,opsd005,sft005
cd "$SOURCE"
exec bash examples/grpo_trainer/run_t062_perf.sh \
  actor_rollout_ref.actor.optim.lr=2e-5 \
  actor_rollout_ref.actor.optim.lr_warmup_steps=5 \
  actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.0 \
  env.rollout.n=1 \
  env.tree_experience.enable=True \
  env.tree_experience.conditional_natural.enable=True \
  env.tree_experience.conditional_natural.natural_rollouts=4 \
  env.tree_experience.task_top_k=4 \
  env.tree_experience.step_top_k=6 \
  env.experience_memory.top_k=6 \
  env.experience_memory.task_specific_top_k=4 \
  env.experience_memory.management.retrieval_alpha=0.5 \
  env.experience_memory.management.retrieval_ucb_c=0.5 \
  env.tree_experience.utility_ema_beta=0.1 \
  env.tree_experience.utility_update_interval=5 \
  env.experience_memory.management.eviction_utility_threshold=-0.01 \
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
  trainer.save_freq=-1 trainer.resume_mode=disable \
  trainer.val_before_train=False trainer.test_freq=-1 \
  data.val_batch_size=24 \
  +trainer.t70_test6125_file="$DATA_ROOT/data/test6125.parquet" \
  +trainer.t70_test6125_freq=50 \
  trainer.project_name=tree-edge-search \
  trainer.experiment_name="$WANDB_NAME" \
  actor_rollout_ref.model.path=/home/zehao/Tree-GRPO/model/Qwen2.5-3B-Instruct \
  actor_rollout_ref.model.lora_rank=64 \
  actor_rollout_ref.model.lora_alpha=128 \
  actor_rollout_ref.actor.ppo_mini_batch_size=16 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
  "$@"
