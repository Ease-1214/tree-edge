#!/usr/bin/env bash
# T062 throughput profile: EDGE original method + T061 LoRA setup, only performance knobs vary.
# Every knob below is infrastructure-only; none of them changes the EDGE method (gating,
# distillation, experience management, prompts, rewards). Validation and checkpoints are off.
#
# Usage (example):
#   PERF_TAG=f1 PERF_QUESTIONS=128 PERF_STEPS=2 bash examples/grpo_trainer/run_t062_perf.sh   # defaults = final config
set -euo pipefail

ROOT=${T062_ROOT:-/home/zehao/echo-collab/T062}
T061=${T061_ROOT:-/home/zehao/echo-collab/T061}

PERF_TAG=${PERF_TAG:?Set PERF_TAG to a short run label}
PERF_MODEL_DTYPE=${PERF_MODEL_DTYPE:-bf16}     # bf16 = frozen base stored in bf16 (final), fp32 = T061 behaviour
PERF_FSDP_SIZE=${PERF_FSDP_SIZE:-1}            # 1 = every GPU keeps a full replica (final), -1 = FULL_SHARD (T061)
PERF_DYNAMIC_BSZ=${PERF_DYNAMIC_BSZ:-1}         # 1 = token-budget micro-batches (final), 0 = fixed 2/4 rows
PERF_PPO_TOKENS=${PERF_PPO_TOKENS:-12288}       # per-GPU token budget for actor update (>= 5700 max row)
PERF_LOGPROB_TOKENS=${PERF_LOGPROB_TOKENS:-24576}
PERF_EAGER=${PERF_EAGER:-0}                     # 0 = CUDA graph (final), 1 = eager (T061)
PERF_VLLM_UTIL=${PERF_VLLM_UTIL:-0.55}
PERF_VLLM_BATCHED_TOKENS=${PERF_VLLM_BATCHED_TOKENS:-16384}  # vLLM scheduler token budget per step (prefill of 5k-token prompts)
PERF_CHUNKED_PREFILL=${PERF_CHUNKED_PREFILL:-True}
PERF_QUESTIONS=${PERF_QUESTIONS:-32}
PERF_STEPS=${PERF_STEPS:-2}
PERF_GPUS=${PERF_GPUS:-4}
# PERF_RESPONSE_ONLY_LOGITS (default True): lm_head only at response predictor positions
# PERF_TEACHER_ACTIVE_ONLY (default True): EDGE teacher forward only on gate-active rows
# Embedding (aux GPU): dynamic token-budget batches, see EDGE_EMBED_* below

if [[ "$PERF_MODEL_DTYPE" == "fp32" ]]; then MODEL_DTYPE_ARG=null; else MODEL_DTYPE_ARG=$PERF_MODEL_DTYPE; fi
if [[ "$PERF_DYNAMIC_BSZ" == "1" ]]; then DYN=True; else DYN=False; fi
if [[ "$PERF_EAGER" == "1" ]]; then EAGER=True; FREE_CACHE=True; else EAGER=False; FREE_CACHE=False; fi

RUN_DIR=$ROOT/runs/$PERF_TAG
mkdir -p "$RUN_DIR"

export PATH=/home/zehao/.conda/envs/tree-edge/bin:/home/zehao/miniforge3/bin:$PATH
export EDGE_CONTROLLER_GPUS=1
export EDGE_EMBED_TOKEN_BUDGET=${EDGE_EMBED_TOKEN_BUDGET:-32768}  # padded tokens per embedding batch; halves itself on OOM
export EDGE_EMBED_MAX_ROWS=${EDGE_EMBED_MAX_ROWS:-256}
export EDGE_EMBED_DTYPE=${EDGE_EMBED_DTYPE:-bf16}
export EDGE_EMBED_MEM_GB=${EDGE_EMBED_MEM_GB:-0}                 # >0 caps the embedding process (e.g. 4 when sharing a GPU)
export CC=/home/zehao/.conda/envs/tree-edge/bin/x86_64-conda-linux-gnu-cc
export RAY_ADDRESS=local RAY_TMPDIR=$ROOT/r
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export ENABLE_EXPERIENCE_EVOLUTION=${ENABLE_EXPERIENCE_EVOLUTION:-false} ENABLE_DYNAMIC_MANAGEMENT=true ENABLE_EXP_ROLLOUT=true
export MODEL_PATH=/home/zehao/Tree-GRPO/model/Qwen2.5-3B-Instruct
export TRAIN_DATA=$T061/data/train.parquet VAL_DATA=$T061/data/validation-smoke280.parquet
export EMBEDDING_MODEL=$T061/assets/Qwen3-Embedding-0.6B
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export SEARCH_URL=${SEARCH_URL:-http://127.0.0.1:29661/retrieve}
export NO_PROXY=127.0.0.1,localhost no_proxy=127.0.0.1,localhost
export N_GPUS_PER_NODE=$PERF_GPUS TRAIN_DATA_SIZE=$PERF_QUESTIONS VAL_DATA_SIZE=128 GROUP_SIZE=8
# Optional server-local network settings contain no credentials.
if [[ -f "$ROOT/wandb-online.env" ]]; then
  source "$ROOT/wandb-online.env"
fi
export WANDB_MODE=${WANDB_MODE:-online}
export HTTP_PROXY=${T062_HTTP_PROXY:-http://127.0.0.1:7890}
export HTTPS_PROXY=$HTTP_PROXY
export http_proxy=$HTTP_PROXY https_proxy=$HTTPS_PROXY
export WANDB_BASE_URL=https://api.forge.coreweave.com WANDB_ENTITY=wuzehao1214-sun-yat-sen-university
unset EDGE_SMOKE_AUDIT   # no per-step hashing / probes in throughput runs

cd "$(dirname "${BASH_SOURCE[0]}")/../.."
bash examples/grpo_trainer/run_search_edge.sh vllm \
 actor_rollout_ref.model.lora_rank=16 actor_rollout_ref.model.lora_alpha=32 actor_rollout_ref.model.init_seed=0 \
 'actor_rollout_ref.model.target_modules=[q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj]' \
 actor_rollout_ref.model.lora_adapter_dtype=fp32 \
 actor_rollout_ref.actor.fsdp_config.model_dtype=$MODEL_DTYPE_ARG \
 actor_rollout_ref.actor.fsdp_config.fsdp_size=$PERF_FSDP_SIZE \
 actor_rollout_ref.actor.optim.lr=5e-6 \
 actor_rollout_ref.model.use_fused_kernels=False \
 actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
 actor_rollout_ref.actor.use_dynamic_bsz=$DYN \
 actor_rollout_ref.actor.response_only_logits=${PERF_RESPONSE_ONLY_LOGITS:-True} \
 actor_rollout_ref.actor.edge_teacher_active_only=${PERF_TEACHER_ACTIVE_ONLY:-True} \
 actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$PERF_PPO_TOKENS \
 actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=$PERF_LOGPROB_TOKENS \
 actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=$PERF_LOGPROB_TOKENS \
 actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
 actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
 actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
 actor_rollout_ref.rollout.gpu_memory_utilization=$PERF_VLLM_UTIL \
 actor_rollout_ref.rollout.max_num_batched_tokens=$PERF_VLLM_BATCHED_TOKENS \
 actor_rollout_ref.rollout.enable_chunked_prefill=$PERF_CHUNKED_PREFILL \
 actor_rollout_ref.rollout.free_cache_engine=$FREE_CACHE \
 actor_rollout_ref.rollout.enforce_eager=$EAGER \
 actor_rollout_ref.rollout.load_format=safetensors \
 actor_rollout_ref.rollout.layered_summon=False \
 actor_rollout_ref.rollout.n=1 \
 env.search.api_format=search_r1_batch env.search.request_batch_size=128 \
 env.experience_memory.device=cuda:0 \
 ray_init.num_cpus=24 trainer.total_training_steps=$PERF_STEPS trainer.total_epochs=1 \
 trainer.val_before_train=False trainer.test_freq=-1 trainer.save_freq=-1 trainer.save_best_checkpoint=False \
 trainer.project_name=tree-edge-t062-perf trainer.experiment_name=t062-$PERF_TAG \
 trainer.default_local_dir=$RUN_DIR/checkpoints \
 "$@" 2>&1 | tee "$RUN_DIR/train.log"
