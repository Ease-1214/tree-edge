import json
import torch
import torch.distributed as dist
from verl.utils.actor_metric_totals import ActorMetricTotals
dist.init_process_group("gloo")
rank=dist.get_rank()
totals=ActorMetricTotals("cpu")
n=rank+1
totals.ratio("kd/no_T/topk_kl",3*n,n)
totals.add("kd/no_T/tokens",n)
totals.add("perf/edge_teacher_rows",rank)
totals.add("perf/edge_student_rows",10)
m=totals.finalize(1)
assert m["kd/no_T/tokens"]==10
assert m["kd/no_T/topk_kl"]==3
assert m["perf/edge_teacher_rows"]==6
assert m["perf/edge_teacher_row_fraction"]==.15
if rank==0:print("METRIC_TOTALS_FOUR_RANK_PASS",json.dumps(m),flush=True)
dist.destroy_process_group()
