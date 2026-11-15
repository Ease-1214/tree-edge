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


# --------------------------------------------------------------------------- #
# Curated dashboard keys (T066).  W&B groups panels by the prefix before "/"  #
# and sorts sections by name, so the numbered prefixes put the ~20 charts     #
# that matter at the top of the default workspace, in reading order.  Raw     #
# keys are still logged for debugging; they appear in the sections below.     #
# --------------------------------------------------------------------------- #

DASHBOARD = [
    # 1 main effect: the no-experience leaf is the deployment condition
    ("1_main/pure_noexp_success", "success/pure_noexp"),
    ("1_main/leaf_success", "success/leaf_mean"),
    # 2 is the experience helping, and is the bank evolving
    ("2_exp/delta_task_T", "exp_ab/task_T/delta_mean"),
    ("2_exp/delta_step_S", "exp_ab/step_S/delta_mean"),
    ("2_exp/helped_rate", "exp_ab/all/helped_rate"),
    ("2_exp/harmed_rate", "exp_ab/all/harmed_rate"),
    ("2_exp/bank_size", "bank/size"),
    ("2_exp/added_task", "reflector/added_task"),
    ("2_exp/added_step", "reflector/added_step"),
    ("2_exp/utility_std", "bank/utility_std"),
    ("2_exp/retrieval_top5_share", "retrieval/top5_share"),
    # 3 distillation volume and the with-T risk
    ("3_kd/canonical_rows", "kd/canonical_rows"),
    ("3_kd/kd_to_pg_loss_ratio", "kd/to_pg_loss_abs_ratio"),
    ("3_kd/student_mass_noT", "kd/no_T/student_mass"),
    ("3_kd/student_mass_withT", "kd/with_T/student_mass"),
    ("3_kd/kl_noT", "kd/no_T/topk_kl"),
    ("3_kd/kl_withT", "kd/with_T/topk_kl"),
    # 4 training health
    ("4_health/grad_norm", "actor/grad_norm"),
    ("4_health/pg_loss", "actor/pg_loss_expanded"),
    ("4_health/entropy", "actor/entropy_expanded"),
    ("4_health/response_length", "actor/response_length_expanded"),
    ("4_health/search_calls_per_q", "tree/search_calls_per_question"),
    # 5 cost
    ("5_cost/step_seconds", "timing_s/step"),
    ("5_cost/generations_per_q", "tree/generations_per_question"),
]


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


def dashboard_metrics(metrics):
    """Copy the curated keys under numbered section prefixes (missing keys skipped)."""
    out = {}
    for dest, src in DASHBOARD:
        if src in metrics:
            value = _scalar(metrics[src])
            if value is not None:
                out[dest] = value
    for key, value in metrics.items():
        if key.startswith("success/pure_noexp/"):
            value = _scalar(value)
            if value is not None:
                out["1_main/pure_noexp_success_" + key.rsplit("/", 1)[-1]] = value
    return out


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
