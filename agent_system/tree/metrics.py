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
