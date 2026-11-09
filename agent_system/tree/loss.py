"""Node KD on the current GRPO micro-batch denominator."""

import torch


def tree_topk_loss(student, teacher, response_mask, weights, grpo_mask):
    log_s = torch.log_softmax(student.float(), dim=-1)
    log_t = torch.log_softmax(teacher.detach().float(), dim=-1)
    probability = log_s.exp()
    kl = (probability * (log_s - log_t)).sum(-1)
    active = response_mask * (weights > 0).unsqueeze(-1)
    loss = (kl * active * weights.unsqueeze(-1)).sum() / grpo_mask.sum().clamp_min(1)
    return loss, kl, probability * ((log_s - log_t) - kl.unsqueeze(-1))


def tree_kd_metrics(
    kl,
    mass,
    weights,
    response_mask,
    path_has_T,
    gap,
    topk_gradient,
    topk_ids,
    responses,
    advantages,
):
    metrics = {}
    active = response_mask * (weights > 0).unsqueeze(-1)
    sampled_direction = -(
        topk_gradient.detach() * (topk_ids == responses.unsqueeze(-1))
    ).sum(-1)
    for label, group in [("no_T", ~path_has_T), ("with_T", path_has_T)]:
        mask = active * group.unsqueeze(-1)
        count = mask.sum().clamp_min(1)
        prefix = "kd/" + label + "/"
        metrics[prefix + "tokens"] = float(mask.sum())
        metrics[prefix + "rows"] = float(((weights > 0) & group).sum())
        for name, value in [
            ("topk_kl", kl.detach()),
            ("student_mass", mass.detach()),
            ("logprob_gap", gap.detach()),
            ("logprob_gap_abs", gap.detach().abs()),
        ]:
            metrics[prefix + name] = float((value * mask).sum() / count)
        metrics[prefix + "mean_weight"] = float(
            (weights.unsqueeze(-1) * mask).sum() / count
        )
        for sign, region in [
            ("positive_adv", advantages > 0),
            ("negative_adv", advantages < 0),
        ]:
            selected = mask * region * (sampled_direction != 0)
            metrics[prefix + sign + "_conflict_proxy"] = float(
                (((sampled_direction * advantages) < 0) * selected).sum()
                / selected.sum().clamp_min(1)
            )
    return metrics
