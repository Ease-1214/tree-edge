#!/usr/bin/env bash
set -euo pipefail
set +x
if [[ ! -f "$HOME/.edge_reflector.env" ]]; then echo 'Missing reflector environment file' >&2; exit 1; fi
source "$HOME/.edge_reflector.env"
export T062_ROOT=/home/zehao/echo-collab/T063
export PERF_TAG=${PERF_TAG:-smoke} PERF_QUESTIONS=64 PERF_STEPS=3
export ENABLE_EXPERIENCE_EVOLUTION=true T063_REFLECTOR_AUDIT=1
export WANDB_MODE=offline T062_HTTP_PROXY=http://127.0.0.1:29662
export EXPERIENCES_JSON=$T062_ROOT/initial-experiences.json
export EDGE_SEARCH_METRICS_PATH=$T062_ROOT/runs/$PERF_TAG/search-metrics.jsonl
bash "$(dirname "${BASH_SOURCE[0]}")/run_t062_perf.sh" \
 env.experience_memory.enable_experience_evolution=True \
 env.experience_memory.update_frequency=1 \
 env.experience_memory.management.eviction_min_exposures=2 \
 env.experience_memory.save_to_canonical_path=False \
 trainer.project_name=tree-edge-t063-smoke trainer.experiment_name=t063-$PERF_TAG \
 trainer.save_freq=2 actor_rollout_ref.actor.checkpoint.contents='[lora_adapter]' \
 "$@"
