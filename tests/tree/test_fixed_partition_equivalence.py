"""Fixed original minibatch/microbatch coefficients, without changing training."""

import copy
import torch
from verl.trainer.ppo.core_algos import compute_policy_loss
from agent_system.tree.loss import tree_topk_loss


class TinyLoRA(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("base", torch.randn(4, 9) * 0.2)
        self.A = torch.nn.Parameter(torch.randn(4, 2) * 0.1)
        self.B = torch.nn.Parameter(torch.randn(2, 9) * 0.1)

    def forward(self, x):
        return x @ (self.base + self.A @ self.B)


def objective(
    model,
    x,
    ids,
    advantages,
    masks,
    kd_active,
    old,
    teacher_ids,
    teacher_logits,
    micros,
    compress,
):
    total = 0.0
    pg_total = 0.0
    kd_total = 0.0
    # Original microbatch boundaries fix c for every original row.
    B = len(ids)
    if compress:
        coefficients = torch.zeros(B)
        for indices in micros:
            coefficients[indices] = len(indices) / (B * masks[indices].sum())
        unique, inverse = torch.unique(ids, sorted=True, return_inverse=True)
        logits = model(x[unique])
        lp = logits.log_softmax(-1)[..., 1]
        ratio = (lp - old[unique]).exp()
        clipped = ratio.clamp(0.8, 1.2)
        positive = torch.zeros(len(unique))
        negative = torch.zeros(len(unique))
        kd_coeff = torch.zeros(len(unique))
        positive.index_add_(0, inverse, coefficients * advantages.clamp_min(0))
        negative.index_add_(0, inverse, coefficients * advantages.clamp_max(0))
        kd_coeff.index_add_(0, inverse, coefficients * kd_active)
        node_mask = masks[torch.tensor([(ids == n).nonzero()[0, 0] for n in unique])]
        pg = (
            (
                -positive[:, None] * torch.minimum(ratio, clipped)
                - negative[:, None] * torch.maximum(ratio, clipped).clamp_max(3)
            )
            * node_mask
        ).sum()
        sl = logits.gather(-1, teacher_ids[unique])
        ls = sl.log_softmax(-1)
        lt = teacher_logits[unique].detach().log_softmax(-1)
        kd = ((ls.exp() * (ls - lt)).sum(-1) * node_mask * kd_coeff[:, None]).sum()
        return pg + kd, pg, kd
    for indices in micros:
        index = torch.tensor(indices)
        mask = masks[index]
        denom = mask.sum()
        c = len(indices) / (B * denom)
        nodes = ids[index]
        logits = model(x[nodes])
        lp = logits.log_softmax(-1)[..., 1]
        pg = (
            compute_policy_loss(
                old[nodes],
                lp,
                advantages[index, None].expand_as(lp),
                mask,
                cliprange=0.2,
                clip_ratio_c=3,
            )[0]
            * len(indices)
            / B
        )
        sl = logits.gather(-1, teacher_ids[nodes])
        kd = (
            tree_topk_loss(sl, teacher_logits[nodes], mask, kd_active[index], mask)[0]
            * len(indices)
            / B
        )
        pg_total = pg_total + pg
        kd_total = kd_total + kd
    return pg_total + kd_total, pg_total, kd_total


def fixture():
    torch.manual_seed(123)
    model = TinyLoRA()
    x = torch.randn(3, 4, 4)
    with torch.no_grad():
        lp = model(x).log_softmax(-1)[..., 1]
    old = lp - torch.tensor(
        [[0.5, -0.6, 1.5, 0.0], [0.0, 0.7, -0.8, 0.0], [1.5, -0.9, 0.1, 0.0]]
    )
    teacher = torch.randn(3, 4, 9)
    tl, ti = teacher.topk(4, -1)
    return model, x, old, ti, tl


def compare_batch(reference, candidate, x, old, ti, tl, ids, adv, mask, kd, micros):
    values = []
    grads = []
    for model, compress in [(reference, False), (candidate, True)]:
        loss, pg, distill = objective(
            model, x, ids, adv, mask, kd, old, ti, tl, micros, compress
        )
        loss.backward()
        values.append(torch.stack([loss, pg, distill]).detach())
        grads.append(torch.cat([p.grad.flatten() for p in model.parameters()]))
    torch.testing.assert_close(values[0], values[1], rtol=1e-5, atol=1e-7)
    torch.testing.assert_close(grads[0], grads[1], rtol=1e-5, atol=1e-7)


def test_single_minibatch_preserves_micro_coefficients():
    ref, x, old, ti, tl = fixture()
    candidate = copy.deepcopy(ref)
    ids = torch.tensor([0, 1, 0, 2, 1, 0])
    adv = torch.tensor([1.0, -0.5, -1.0, -0.7, 0.2, 0.3])
    mask = torch.tensor(
        [
            [1.0, 1.0, 0.0, 0.0],
            [1.0, 1.0, 1.0, 1.0],
            [1.0, 1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0, 1.0],
            [1.0, 1.0, 0.0, 0.0],
        ]
    )
    compare_batch(
        ref,
        candidate,
        x,
        old,
        ti,
        tl,
        ids,
        adv,
        mask,
        torch.tensor([0.5, 0.0, 0.0, 0.3, 0.0, 0.0]),
        [[0, 1, 2], [3, 4, 5]],
    )


def test_two_minibatches_optimizer_trajectory():
    ref, x, old, ti, tl = fixture()
    candidate = copy.deepcopy(ref)
    optimizers = [torch.optim.Adam(m.parameters(), lr=1e-3) for m in [ref, candidate]]
    for ids, adv in [
        (torch.tensor([0, 1, 0, 2]), torch.tensor([1.0, -0.5, -1.0, -0.7])),
        (torch.tensor([0, 2, 0, 1]), torch.tensor([-0.3, 0.7, 0.2, -0.6])),
    ]:
        lengths = torch.tensor([2, 4, 1])[ids]
        mask = (torch.arange(4)[None, :] < lengths[:, None]).float()
        compare_batch(
            ref,
            candidate,
            x,
            old,
            ti,
            tl,
            ids,
            adv,
            mask,
            torch.tensor([0.5, 0.0, 0.0, 0.0]),
            [[0, 1], [2, 3]],
        )
        for optimizer in optimizers:
            optimizer.step()
            optimizer.zero_grad()
        for a, b in zip(ref.parameters(), candidate.parameters()):
            torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-7)
