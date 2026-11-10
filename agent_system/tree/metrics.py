"""Collect tree diagnostics from every Actor rank before DataProto keeps rank zero."""

from collections import defaultdict
import torch.distributed as distributed


def merge_tree_metrics(per_rank):
    combined = defaultdict(list)
    for metrics in per_rank:
        for name, value in metrics.items():
            combined[name].extend(value if isinstance(value, list) else [value])
    teacher_rows = sum(combined.get("perf/edge_teacher_rows", []))
    student_rows = sum(combined.get("perf/edge_student_rows", []))
    combined["perf/edge_teacher_rows"] = teacher_rows
    combined["perf/edge_student_rows"] = student_rows
    combined["perf/edge_teacher_row_fraction"] = teacher_rows / max(1, student_rows)
    for key in ("perf/max_memory_allocated_gb", "perf/max_memory_reserved_gb"):
        if key in combined:
            combined[key] = max(combined[key])
    return dict(combined)


def gather_tree_metrics(metrics):
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
    ("3_kd/kd_to_pg_ratio", "kd/to_pg_abs_ratio"),
    ("3_kd/student_mass_noT", "kd/no_T/student_mass"),
    ("3_kd/student_mass_withT", "kd/with_T/student_mass"),
    ("3_kd/kl_noT", "kd/no_T/topk_kl"),
    ("3_kd/kl_withT", "kd/with_T/topk_kl"),
    # 4 training health
    ("4_health/grad_norm", "actor/grad_norm"),
    ("4_health/pg_loss", "actor/pg_loss_expanded"),
    ("4_health/entropy", "actor/entropy_expanded"),
    ("4_health/response_length", "response_length/mean"),
    ("4_health/search_calls_per_q", "tree/search_calls_per_question"),
    # 5 cost
    ("5_cost/step_seconds", "timing_s/step"),
    ("5_cost/generations_per_q", "tree/generations_per_question"),
]


def _scalar(value):
    if isinstance(value, (list, tuple)):
        value = [float(v) for v in value]
        return sum(value) / len(value) if value else None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


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
