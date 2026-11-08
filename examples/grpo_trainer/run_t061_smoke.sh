#!/usr/bin/env bash
set -euo pipefail
ROOT=/home/zehao/echo-collab/T061
export PATH=/home/zehao/.conda/envs/tree-edge/bin:/home/zehao/miniforge3/bin:$PATH
export CUDA_VISIBLE_DEVICES=0,1,2,3
export CC=/home/zehao/.conda/envs/tree-edge/bin/x86_64-conda-linux-gnu-cc
export RAY_ADDRESS=local RAY_TMPDIR=$ROOT/ray-smoke
export ENABLE_EXPERIENCE_EVOLUTION=false ENABLE_DYNAMIC_MANAGEMENT=true ENABLE_EXP_ROLLOUT=true
export MODEL_PATH=/home/zehao/Tree-GRPO/model/Qwen2.5-3B-Instruct
export TRAIN_DATA=$ROOT/data/train.parquet VAL_DATA=$ROOT/data/validation.parquet
export SEARCH_URL=http://127.0.0.1:29661/retrieve
export NO_PROXY=127.0.0.1,localhost no_proxy=127.0.0.1,localhost
export N_GPUS_PER_NODE=4 TRAIN_DATA_SIZE=32 VAL_DATA_SIZE=128 GROUP_SIZE=8
export WANDB_BASE_URL=https://api.forge.coreweave.com WANDB_ENTITY=wuzehao1214-sun-yat-sen-university
cd "$ROOT/EDGE"
bash examples/grpo_trainer/run_search_edge.sh vllm \
 actor_rollout_ref.model.lora_rank=16 actor_rollout_ref.model.lora_alpha=32 \
 'actor_rollout_ref.model.target_modules=[q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj]' \
 actor_rollout_ref.actor.optim.lr=5e-6 \
 actor_rollout_ref.model.use_fused_kernels=False \
 actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
 actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
 actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
 actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
 actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
 actor_rollout_ref.rollout.free_cache_engine=True \
 actor_rollout_ref.rollout.enforce_eager=True \
 actor_rollout_ref.rollout.load_format=safetensors \
 actor_rollout_ref.rollout.layered_summon=True \
 actor_rollout_ref.rollout.n=1 \
 env.search.api_format=search_r1_batch \
 env.experience_memory.device=cpu \
 ray_init.num_cpus=24 trainer.total_training_steps=1 trainer.total_epochs=1 trainer.test_freq=20 \
 trainer.project_name=tree-edge-stage0 trainer.experiment_name=t061-edge-lora-smoke \
 trainer.rollout_data_dir=$ROOT/smoke/rollouts \
 trainer.validation_data_dir=$ROOT/smoke/validation \
 trainer.default_local_dir=$ROOT/smoke/checkpoints \
 "$@"
