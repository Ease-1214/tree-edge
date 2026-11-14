"""T70 v2: independent active-KD-token normalization and faithful metrics."""
import multiprocessing
from types import SimpleNamespace
import torch
import torch.distributed as dist
import pytest
from verl import DataProto
from agent_system.tree.loss import tree_topk_loss
from verl.utils.actor_metric_totals import ActorMetricTotals
from verl.utils.gradient_diagnostics import select_probe_microbatches
from verl.utils.replica_sync import summed_denominators


def test_active_kd_denominator_and_gain_weights_are_distinct():
    torch.manual_seed(30)
    logits = torch.randn(3, 4, 5, requires_grad=True)
    teacher = torch.randn(3, 4, 5, requires_grad=True)
    mask = torch.tensor([[1.,1.,0.,0.], [1.,1.,1.,1.], [1.,0.,0.,0.]])
    weights = torch.tensor([0.5,0.0,0.25])
    # The GRPO denominator must have no influence whatsoever.
    one, kl, _ = tree_topk_loss(logits,teacher,mask,weights,torch.ones(3,400))
    two, _, _ = tree_topk_loss(logits,teacher,mask,weights,torch.ones(3,1))
    expected = (0.5*kl[0,:2].sum() + 0.25*kl[2,0]) / 3.
    torch.testing.assert_close(one,expected)
    torch.testing.assert_close(one,two)
    loss_all_weight_den=(0.5*kl[0,:2].sum()+0.25*kl[2,0])/(2*0.5+0.25)
    assert not torch.allclose(one,loss_all_weight_den)
    one.backward()
    assert teacher.grad is None
    assert logits.grad[0].abs().sum()>0 and logits.grad[2].abs().sum()>0
    assert logits.grad[1].abs().sum()==0
    zero,_,_=tree_topk_loss(logits.detach(),teacher.detach(),mask,weights*0,mask)
    assert torch.isfinite(zero) and zero == 0.


def test_actor_metric_reports_both_active_and_all_token_kd_objectives():
    totals=ActorMetricTotals('cpu')
    # ratio uses actual active KD denominator, while the old diagnostic key
    # continues to expose GRPO-token-diluted loss for historical comparison.
    expanded=torch.ones(2,6)
    mask=torch.tensor([[1.,1.,1.,1.,1.,1.],[1.,1.,1.,1.,1.,1.]])
    data={'edge_distill_weight':torch.tensor([0.5,0.])}
    totals.ratio('actor/pg_loss_expanded',torch.tensor(0.2*12.),torch.tensor(12.))
    totals.kd(data,mask,expanded,torch.tensor(0.125),tree_batch=True)
    m=totals.finalize(0.05)
    assert m['kd/active_tokens']==6
    assert m['kd/weighted_loss_active_tokens']==pytest.approx(0.125)
    assert m['kd/weighted_loss_all_tokens']==pytest.approx(0.0625)
    assert m['edge/distill_loss']==pytest.approx(0.125)
    assert m['kd/weighted_loss_active_denominator']==pytest.approx(0.00625)
    assert m['kd/weighted_loss_pg_denominator']==pytest.approx(0.003125)
    assert m['kd/to_pg_loss_abs_ratio']==pytest.approx(.00625/(.2+1e-6))


def test_gradient_probes_select_distinct_kd_and_sft_active_microbatches():
    # First mini has KD in row 1 and success-SFT in row 2: selecting only
    # the KD micro would silently report zero SFT even though SFT is active.
    B,R=4,2
    data=DataProto.from_dict(tensors={
        'responses':torch.ones(B,R,dtype=torch.long),
        'attention_mask':torch.ones(B,R+1,dtype=torch.long),
        'edge_distill_weight':torch.tensor([0.,.5,0.,0.]),
        'kd_canonical':torch.tensor([False,True,False,False]),
        'is_padding_copy':torch.zeros(B,dtype=torch.bool),
        'sft_weight':torch.tensor([0.,0.,1.,0.]),
    })
    actor=SimpleNamespace(config=SimpleNamespace(
        use_dynamic_bsz=False,ppo_micro_batch_size_per_gpu=1,
        ppo_mini_batch_size=4,
    ))
    kd,sft=select_probe_microbatches(actor,data)
    assert kd['edge_distill_weight'][0]==.5
    assert kd['sft_weight'][0]==0
    assert sft['edge_distill_weight'][0]==0
    assert sft['sft_weight'][0]==1


def _worker(rank,path,queue):
    dist.init_process_group('gloo',init_method='file://'+path,
                            rank=rank,world_size=4)
    try:
        # Some replicas have zero KD-active tokens; all replicas must
        # contribute to a single common active-token mean.
        masks=[torch.tensor([[1.,1.,1.]]),
               torch.tensor([[1.,1.,0.]]),
               torch.tensor([[1.,1.,1.]]),
               torch.tensor([[1.,1.,1.]])]
        ws=[torch.tensor([.5]),torch.tensor([0.]),torch.tensor([.25]),torch.tensor([0.])]
        sftw=[torch.tensor([0.]),torch.tensor([1.]),torch.tensor([0.]),torch.tensor([2.])]
        mask,weight,sw=masks[rank],ws[rank],sftw[rank]
        denom=summed_denominators(mask,torch.ones_like(weight),weight,sw)
        parameter=torch.nn.Parameter(torch.tensor(.3))
        # Two micro-batches partition the response tokens; each micro uses
        # its own active-token mean, corrected by active_tokens/global_count.
        for window in (slice(0,1),slice(1,3)):
            m=mask[:,window]
            active_count=(m*(weight>0)[:,None]).sum().clamp_min(1)
            local_num=(m*weight[:,None] * (rank+1)*parameter).sum()
            local_loss=local_num/active_count
            local_active=(m*(weight>0)[:,None]).sum()
            local_sft_num=(m*sw[:,None] * (rank+1)*parameter).sum()
            loss=0.05*4*local_loss*local_active/denom[1] + .05*4*local_sft_num/denom[2]
            loss.backward()
        grad=parameter.grad.detach().clone()
        dist.all_reduce(grad)
        grad.div_(4)
        if rank==0:
            # KD: rank0 3*.5*1 + rank2 3*.25*3 = 3.75 / 6
            # SFT: rank1 2*1*2 + rank3 3*2*4 = 28 / (2+6)
            target=.05*(3.75/6 + 28/8)
            queue.put((float(grad),target))
    finally:
        dist.destroy_process_group()


def test_four_replica_kd_and_sft_have_separate_global_denominators(tmp_path):
    ctx=multiprocessing.get_context('spawn')
    q=ctx.Queue()
    init=str(tmp_path/'gloo-init')
    ps=[ctx.Process(target=_worker,args=(r,init,q)) for r in range(4)]
    for p in ps:p.start()
    try:
        found,expected=q.get(timeout=70)
        assert found==pytest.approx(expected,abs=1e-6)
        for p in ps:
            p.join(timeout=25)
            assert p.exitcode==0
    finally:
        for p in ps:
            if p.is_alive():
                p.terminate()
                p.join()
