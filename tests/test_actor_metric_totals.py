"""Monitor totals match explicit leaf expansion and ignore microbatch partitioning."""
import torch
import pytest
from verl.utils.actor_metric_totals import ActorMetricTotals
from verl.trainer.ppo.core_algos import compute_policy_loss

def test_mixed_sign_node_clipping_matches_expanded_leaves():
    # One node represents four leaves, with positive, negative and zero advantages.
    logs=torch.tensor([[.4,-.4,1.4]])
    mask=torch.ones_like(logs)
    data=dict(w_tok=torch.tensor([4.]),positive_adv_count=torch.tensor([2.]),
        negative_adv_count=torch.tensor([1.]))
    totals=ActorMetricTotals("cpu")
    totals.policy(data,mask,torch.zeros_like(logs),logs,torch.ones_like(logs),
        torch.tensor(.25),torch.ones_like(logs),.2,.2,3.)
    metrics=totals.finalize(1.)
    advantages=torch.tensor([1.,2.,-1.,0.])[:,None].expand(4,3)
    _,clip,kl,lower=compute_policy_loss(torch.zeros(4,3),logs.expand(4,3),
        advantages,torch.ones(4,3),cliprange=.2)
    assert metrics["actor/pg_clipfrac"]==pytest.approx(float(clip))
    assert metrics["actor/pg_clipfrac_lower"]==pytest.approx(float(lower))
    assert metrics["actor/ppo_kl"]==pytest.approx(float(kl))
    assert metrics["actor/expanded_tokens"]==12

def summarize(chunks):
    totals=ActorMetricTotals("cpu")
    logs=torch.tensor([[.1,.2],[.2,.1],[.3,.2]])
    mask=torch.tensor([[1.,1.],[1.,0.],[1.,1.]])
    weights=torch.tensor([.5,0.,0.])
    kl=torch.tensor([[2.,4.],[9.,9.],[8.,8.]])
    for indexes in chunks:
        m=mask[indexes];w=weights[indexes];lp=logs[indexes]
        d=dict(responses=torch.ones(len(indexes),2,dtype=torch.long),
            edge_distill_weight=w,kd_path_has_T=torch.zeros(len(indexes),dtype=torch.bool))
        expanded,pos,neg=totals.policy(d,m,torch.zeros_like(lp),lp,torch.ones_like(lp),
            torch.tensor(.4),lp,.2,.2,3.)
        active=m*(w>0)[:,None]
        loss=(kl[indexes]*active*w[:,None]).sum()/m.sum().clamp_min(1)
        totals.kd(d,m,expanded,loss,True,kl[indexes],torch.ones_like(m)*.8,
            torch.zeros_like(m),torch.ones(len(indexes),2,2),
            torch.ones(len(indexes),2,2,dtype=torch.long),pos,neg)
    return totals.finalize(.2)

def test_inactive_microbatches_stay_in_coverage_denominator():
    a=summarize([[0],[1,2]])
    b=summarize([[0,1,2]])
    for key in a:assert a[key]==pytest.approx(b[key])
    assert a["kd/active_token_rate"]==pytest.approx(2/5)
    assert a["kd/effective_weight_sum_over_N"]==pytest.approx(1/5)
    assert a["kd/no_T/topk_kl"]==pytest.approx(3.)
    assert a["kd/no_T/tokens"]==2
    assert "kd/with_T/topk_kl" not in a
    assert a["kd/with_T/tokens"]==0
    assert a["kd/weighted_loss_all_tokens"]==pytest.approx(3/5)
    assert a["kd/to_pg_loss_abs_ratio"]==pytest.approx(.2*(3/5)/(.4+1e-6))
