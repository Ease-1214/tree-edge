"""Four-rank CPU/GPU regression for the production replica synchronization helpers."""
import argparse,copy,json,os,sys,tempfile
from contextlib import nullcontext
from pathlib import Path
import torch
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP,ShardingStrategy
from torch.distributed.device_mesh import init_device_mesh
from verl.utils.replica_sync import average_replica_gradients,summed_denominators,verify_replica_parameters

class TinyAdapter(torch.nn.Module):
    def __init__(self,device):
        super().__init__()
        self.lora_a=torch.nn.Parameter(torch.randn(4,8,device=device)*.1)
        self.lora_b=torch.nn.Parameter(torch.randn(8,4,device=device)*.1)
    def forward(self,x):
        return x@self.lora_a.T@self.lora_b.T

def batch(rank,step,device):
    g=torch.Generator(device=device).manual_seed(1000+step*13+rank)
    n=[2,3,5,7][rank]
    return torch.randn(n,8,generator=g,device=device),torch.randn(n,8,generator=g,device=device)

def numerators(model,x,y,rank):
    output=model(x)
    pg=(output-y).square().sum()
    kd=(output+.25).square().sum() if rank==0 else output.sum()*0
    return pg,kd

def main(gpu):
    rank=int(os.environ["RANK"]);world=int(os.environ["WORLD_SIZE"])
    assert world==4
    if gpu:torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    device=torch.device("cuda",int(os.environ["LOCAL_RANK"])) if gpu else torch.device("cpu")
    dist.init_process_group("nccl" if gpu else "gloo")
    torch.manual_seed(17)
    reference=TinyAdapter(device)
    local=copy.deepcopy(reference)
    if gpu:
        mesh=init_device_mesh("cuda",(world,1),mesh_dim_names=["ddp","fsdp"])
        local=FSDP(local,device_id=device,device_mesh=mesh,
            sharding_strategy=ShardingStrategy.HYBRID_SHARD,use_orig_params=False,sync_module_states=True)
        assert local.sharding_strategy==ShardingStrategy.NO_SHARD
    optimizer=torch.optim.AdamW(local.parameters(),lr=.001)
    ref_optimizer=torch.optim.AdamW(reference.parameters(),lr=.001)
    scheduler=torch.optim.lr_scheduler.StepLR(optimizer,step_size=2,gamma=.9)
    ref_scheduler=torch.optim.lr_scheduler.StepLR(ref_optimizer,step_size=2,gamma=.9)
    differences=[]
    for step in range(4):
        optimizer.zero_grad(set_to_none=True);ref_optimizer.zero_grad(set_to_none=True)
        x,y=batch(rank,step,device)
        mask=torch.ones(len(x),8,device=device)
        den=summed_denominators(mask,torch.ones(len(x),device=device),
            torch.ones(len(x),device=device) if rank==0 else torch.zeros(len(x),device=device))
        with local.no_sync() if gpu else nullcontext():
            # Unequal local microbatch counts: 1,2,5,7. Three ranks have no KD.
            chunks=torch.tensor_split(torch.arange(len(x),device=device),[1,2,5,7][rank])
            for idx in chunks:
                pg,kd=numerators(local,x[idx],y[idx],rank)
                (world*(pg/den[0]+.2*kd/den[1])).backward()
        average_replica_gradients(local)
        # The serial objective concatenates every rank's examples.
        total_pg=total_kd=None
        for source in range(world):
            bx,by=batch(source,step,device)
            pg,kd=numerators(reference,bx,by,source)
            total_pg=pg if total_pg is None else total_pg+pg
            total_kd=kd if total_kd is None else total_kd+kd
        (total_pg/den[0]+.2*total_kd/den[1]).backward()
        actual=torch.cat([p.grad.reshape(-1) for p in local.parameters()])
        expected=torch.cat([p.grad.reshape(-1) for p in reference.parameters()])
        error=float((actual-expected).abs().max()/expected.abs().max().clamp_min(1e-8))
        differences.append(error);assert error<1e-5,error
        torch.nn.utils.clip_grad_norm_(local.parameters(),1)
        torch.nn.utils.clip_grad_norm_(reference.parameters(),1)
        optimizer.step();ref_optimizer.step();scheduler.step();ref_scheduler.step()
        verify_replica_parameters(local)
        for p in local.parameters():
            for value in optimizer.state.get(p,{}).values():
                if torch.is_tensor(value):
                    collective_value=value.to(device)
                    gathered=[torch.empty_like(collective_value) for _ in range(world)]
                    dist.all_gather(gathered,collective_value)
                    assert all(torch.equal(collective_value,v) for v in gathered)
        if step==2:
            saved=copy.deepcopy(dict(model=local.state_dict(),optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),
                rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state() if gpu else None))
            # Restore through disk, then compare the following update to uninterrupted reference.
            path=Path(tempfile.gettempdir())/f"t069-sync-{os.getpid()}.pt"
            torch.save(saved,path);restored=torch.load(path,map_location=device,weights_only=False);path.unlink()
            local.load_state_dict(restored["model"]);optimizer.load_state_dict(restored["optimizer"])
            scheduler.load_state_dict(restored["scheduler"]);torch.set_rng_state(restored["rng"].cpu())
            if gpu:torch.cuda.set_rng_state(restored["cuda_rng"].cpu())
        actual=torch.cat([p.detach().reshape(-1) for p in local.parameters()])
        expected=torch.cat([p.detach().reshape(-1) for p in reference.parameters()])
        assert torch.allclose(actual,expected,atol=1e-7,rtol=1e-6)
    if rank==0:print("T069_SYNC_REGRESSION",json.dumps(dict(backend="nccl-fsdp" if gpu else "gloo",world_size=world,
        updates=4,max_gradient_relative_error=max(differences),adam_equal=True,recovery_next_update_equal=True)),flush=True)
    dist.destroy_process_group()

if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--gpu",action="store_true")
    main(parser.parse_args().gpu)
