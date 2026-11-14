import torch
import pytest
from agent_system.tree.dedup import weighted_node_policy_loss
from verl.trainer.ppo.core_algos import compute_policy_loss


def expanded_loss(logprob, old, advantages, mask):
    return compute_policy_loss(old,logprob,advantages[:,None].expand_as(logprob),mask,cliprange=.2,clip_ratio_c=3.)[0]


def test_single_batch_clipping_and_gradient():
    logits=torch.tensor([[.4,-.7,1.5],[.1,-.1,.3]],requires_grad=True)
    indices=torch.tensor([0,0,1]);adv=torch.tensor([1.3,-.7,-.4]);mask=torch.tensor([[1.,1.,1.],[1.,1.,1.],[1.,1.,0.]])
    old=torch.zeros_like(logits)
    expanded=expanded_loss(logits[indices],old[indices],adv,mask)
    compressed=weighted_node_policy_loss(old,logits,torch.tensor([1.3,0.]),torch.tensor([-.7,-.4]),mask[[0,2]],torch.tensor([2.,1.]))
    ge=torch.autograd.grad(expanded,logits,retain_graph=True)[0]
    gc=torch.autograd.grad(compressed,logits)[0]
    torch.testing.assert_close(expanded,compressed,rtol=1e-5,atol=1e-7)
    torch.testing.assert_close(ge,gc,rtol=1e-5,atol=1e-7)


@pytest.mark.xfail(strict=True, reason="B2 fails when token-mean minibatch boundaries change; dedup remains disabled")
def test_multiple_batches_preserve_total_gradient():
    logits=torch.zeros(3,3,requires_grad=True)
    old=torch.zeros_like(logits)
    masks=torch.tensor([[1.,0.,0.],[1.,1.,1.],[1.,0.,0.]])
    # Two original mini-batches, each representing 2 expanded rows.
    original=expanded_loss(logits[[0,1]],old[[0,1]],torch.tensor([1.,1.]),masks[[0,1]])
    original+=expanded_loss(logits[[0,2]],old[[0,2]],torch.tensor([-1.,-1.]),masks[[0,2]])
    # Weighted balancing still makes two mini-batches, each with total weight 2.
    dedup=weighted_node_policy_loss(old[[0]],logits[[0]],torch.tensor([1.]),torch.tensor([-1.]),masks[[0]],torch.tensor([2.]))
    dedup+=weighted_node_policy_loss(old[[1,2]],logits[[1,2]],torch.tensor([1.,0.]),torch.tensor([0.,-1.]),masks[[1,2]],torch.tensor([1.,1.]))
    ge=torch.autograd.grad(original,logits,retain_graph=True)[0]
    gd=torch.autograd.grad(dedup,logits)[0]
    torch.testing.assert_close(ge,gd,rtol=1e-5,atol=1e-7)
