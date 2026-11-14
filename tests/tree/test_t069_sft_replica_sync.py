"""Global success-token SFT normalization under four-way replica synchronization."""
import multiprocessing
import torch
import torch.distributed as dist
from verl.utils.replica_sync import summed_denominators


def _four_rank_worker(rank, path, queue):
    dist.init_process_group("gloo",init_method="file://" + path,rank=rank,world_size=4)
    try:
        mask=torch.tensor([[1.,1.,1.],[1.,1.,0.]])
        row_weights=torch.ones(2)
        kd_weights=torch.tensor([1. if rank==0 else 0.,
                                 1. if rank==2 else 0.])
        active = {0:[2.,0.],1:[0.,0.],2:[0.,3.],3:[1.,0.]}
        sft_weights=torch.tensor(active[rank])
        d=summed_denominators(mask,row_weights,kd_weights,sft_weights)
        assert d.shape==(3,)
        torch.testing.assert_close(d,torch.tensor([20.,5.,15.],dtype=torch.float64))
        # A real parameter with uneven successful-token weights by rank.
        parameter=torch.nn.Parameter(torch.tensor(0.25))
        local_logprob=-(rank+1)*parameter*torch.ones_like(mask)
        numerator=-(local_logprob*mask*sft_weights[:,None]).sum()
        loss=0.05 * dist.get_world_size() * numerator / d[2]
        loss.backward()
        total_grad=parameter.grad.detach().clone()
        dist.all_reduce(total_grad)
        total_grad/=dist.get_world_size() # Replica gradient averaging
        if rank==0:
            queue.put(float(total_grad))
    finally:
        dist.destroy_process_group()


def test_four_rank_global_sft_token_mean_gradient(tmp_path):
    ctx=multiprocessing.get_context("spawn")
    queue=ctx.Queue()
    init=str(tmp_path / "sft-gloo-init")
    children=[ctx.Process(target=_four_rank_worker,args=(rank,init,queue))
              for rank in range(4)]
    for process in children: process.start()
    try:
        grad=queue.get(timeout=65)
        # Success-token contributions: 6*1 + 6*3 + 3*4 = 36; denom 15
        assert abs(grad-0.05*36/15)<1e-6
        for process in children:
            process.join(timeout=20)
            assert process.exitcode==0
    finally:
        for process in children:
            if process.is_alive():
                process.terminate()
                process.join()
