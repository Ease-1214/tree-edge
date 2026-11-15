"""Collect tree diagnostics from every Actor rank before DataProto keeps rank zero."""

from collections import defaultdict
import math

try:
    import torch.distributed as distributed
except ImportError:  # pragma: no cover - CPU-only / test environments without PyTorch
    distributed = None


def merge_tree_metrics(per_rank):
    combined = defaultdict(list)
    for metrics in per_rank:
        for name, value in metrics.items():
            combined[name].extend(value if isinstance(value, list) else [value])
    global_actor_statistics = all(float(m.get("actor/statistics_global", 0)) == 1 for m in per_rank)
    teacher_rows = sum(combined.get("perf/edge_teacher_rows", []))
    student_rows = sum(combined.get("perf/edge_student_rows", []))
    if global_actor_statistics:
        teacher_rows = per_rank[0].get("perf/edge_teacher_rows", 0)
        student_rows = per_rank[0].get("perf/edge_student_rows", 0)
    combined["perf/edge_teacher_rows"] = teacher_rows
    combined["perf/edge_student_rows"] = student_rows
    combined["perf/edge_teacher_row_fraction"] = teacher_rows / max(1, student_rows)
    for key in ("perf/max_memory_allocated_gb", "perf/max_memory_reserved_gb"):
        if key in combined:
            combined[key] = max(combined[key])
    return dict(combined)


def gather_tree_metrics(metrics):
    if distributed is None:
        raise ImportError("torch.distributed is required for gather_tree_metrics")
    per_rank = [None] * distributed.get_world_size()
    distributed.all_gather_object(per_rank, metrics)
    return merge_tree_metrics(per_rank)


def _scalar(value):
    if isinstance(value, (list, tuple)):
        vals = []
        for item in value:
            try:
                num = float(item)
            except (TypeError, ValueError):
                continue
            if math.isfinite(num):
                vals.append(num)
        return sum(vals) / len(vals) if vals else None
    if isinstance(value, bool):
        return float(int(value))
    try:
        num = float(value)
    except (TypeError, ValueError):
        return None
    return num if math.isfinite(num) else None


# Clean 6-panel W&B specification: (wandb_key, fallback_raw_keys).
# Full raw metrics are still written to step_metrics.jsonl on disk; W&B logs
# only these curated, non-redundant metrics unless WANDB_LOG_RAW_METRICS=1.
CLEAN_WANDB_PANELS = (
    # 1_eval: validation & core training pass/rescue rates
    ("1_eval/val_strict_em", ("val/strict_em", "val/success_rate")),
    ("1_eval/val_macro_em", ("val/macro_em",)),
    ("1_eval/train_natural_pass_rate", ("natural/pass_rate", "success/pure_noexp")),
    ("1_eval/train_tree_rescue_rate", ("tree/rescue_rate",)),
    ("1_eval/train_leaf_mean_reward", ("success/leaf_mean",)),
    # 2_tree_exp: tree search & A/B experience advantage
    ("2_tree_exp/tree_trigger_rate", ("tree/trigger_rate",)),
    ("2_tree_exp/fork_count", ("gate/fork_count",)),
    ("2_tree_exp/task_delta_mean", ("exp_ab/task_T/delta_mean",)),
    ("2_tree_exp/step_delta_mean", ("exp_ab/step_S/delta_mean",)),
    ("2_tree_exp/helped_rate", ("exp_ab/all/helped_rate",)),
    ("2_tree_exp/harmed_rate", ("exp_ab/all/harmed_rate",)),
    ("2_tree_exp/same_action_rate", ("gate/same_action_rate",)),
    # 3_memory: experience bank size, utility & churn
    ("3_memory/bank_size", ("bank/size",)),
    ("3_memory/utility_mean", ("bank/utility_mean",)),
    ("3_memory/utility_positive_rate", ("bank/utility_positive_rate",)),
    ("3_memory/exposed_experiences", ("retrieval/exposed_experiences",)),
    ("3_memory/reflector_submitted_forks", ("reflector/submitted",)),
    ("3_memory/added_task", ("reflector/added_task",)),
    ("3_memory/added_step", ("reflector/added_step",)),
    ("3_memory/evicted_negative", ("bank/deleted_negative",)),
    ("3_memory/evicted_capacity", ("bank/deleted_capacity",)),
    # 4_loss: PG / OPSD / SFT optimization health
    ("4_loss/pg_loss", ("actor/pg_loss_expanded", "actor/pg_loss")),
    ("4_loss/opsd_loss", ("kd/weighted_loss_active_tokens", "edge/distill_loss")),
    ("4_loss/opsd_active_rows", ("kd/canonical_rows", "kd/active_rows")),
    ("4_loss/sft_loss", ("sft/loss_token_mean",)),
    ("4_loss/sft_active_rows", ("sft/active_rows",)),
    ("4_loss/grad_norm", ("actor/grad_norm",)),
    ("4_loss/ppo_kl", ("actor/ppo_kl",)),
    ("4_loss/clipfrac", ("actor/pg_clipfrac",)),
    ("4_loss/lr", ("actor/lr",)),
    # 5_behavior: model formatting & search calls
    ("5_behavior/valid_action_ratio", ("episode/valid_action_ratio",)),
    ("5_behavior/search_calls_per_question", ("tree/search_calls_per_question",)),
    ("5_behavior/response_length_mean", ("actor/response_length_expanded", "response_length/mean")),
    # 6_perf: step timing & GPU throughput
    ("6_perf/step_time_s", ("timing_s/step",)),
    ("6_perf/rollout_time_s", ("timing_s/gen",)),
    ("6_perf/actor_update_time_s", ("timing_s/update_actor",)),
    ("6_perf/reflector_time_s", ("timing_s/experience_management",)),
    ("6_perf/max_gpu_mem_gb", ("perf/max_memory_allocated_gb",)),
    ("6_perf/tokens_per_sec_per_gpu", ("perf/throughput",)),
)


def build_clean_wandb_metrics(metrics: dict) -> dict:
    """Select the first finite scalar for each curated 6-panel W&B metric."""
    out = {}
    for dest, candidates in CLEAN_WANDB_PANELS:
        for src in candidates:
            if src in metrics:
                value = _scalar(metrics[src])
                if value is not None:
                    out[dest] = value
                    break
    for key, raw_value in metrics.items():
        if key.startswith("val/") and key.endswith("_success_rate") and key != "val/success_rate":
            src = key[len("val/") : -len("_success_rate")]
            if src:
                value = _scalar(raw_value)
                if value is not None:
                    out.setdefault(f"1_eval/val_{src}_em", value)
    return out
