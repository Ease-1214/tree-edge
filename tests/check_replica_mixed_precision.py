"""Regression: NO_SHARD no_sync leaves bf16 grads on fp32 LoRA masters."""
import os,tempfile
import torch
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP,ShardingStrategy,MixedPrecision
from verl.utils.replica_sync import average_replica_gradients

def main():
    torch.cuda.set_device(0)
    dist.init_process_group("nccl",init_method="file://"+tempfile.mktemp(),rank=0,world_size=1)
    torch.manual_seed(7)
    model=FSDP(torch.nn.Linear(8,4).cuda(),device_id=0,sharding_strategy=ShardingStrategy.NO_SHARD,
        mixed_precision=MixedPrecision(param_dtype=torch.bfloat16,reduce_dtype=torch.float32,buffer_dtype=torch.float32))
    optimizer=torch.optim.AdamW(model.parameters(),lr=1e-3)
    observed=[]
    for step in range(3):
        optimizer.zero_grad(set_to_none=True)
        with model.no_sync():
            for _ in range(2):model(torch.randn(3,8,device="cuda")).float().square().mean().backward()
        observed.append([(str(p.dtype),str(p.grad.dtype)) for p in model.parameters()])
        average_replica_gradients(model)
        assert all(p.grad.dtype==p.dtype for p in model.parameters())
        optimizer.step()
        assert all(torch.isfinite(p).all() for p in model.parameters())
    print("MIXED_PRECISION_ADAM_PASS",observed,flush=True)
    dist.destroy_process_group()
if __name__=="__main__":main()
