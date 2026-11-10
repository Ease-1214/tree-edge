import json
import torch.distributed as dist
from verl.utils.checkpoint.adapter_state import raise_checkpoint_errors
dist.init_process_group("gloo")
rank=dist.get_rank()
failed=False
try:raise_checkpoint_errors("injected rank-1 failure" if rank==1 else None)
except RuntimeError as error:
    failed="rank-1" in str(error)
assert failed
raise_checkpoint_errors(None)
if rank==0:print("CHECKPOINT_FAILURE_COLLECTIVE_PASSED",json.dumps({"world_size":dist.get_world_size(),"all_ranks_rejected_failure":True,"success_path_passed":True}),flush=True)
dist.destroy_process_group()
