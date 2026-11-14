"""Candidate weighted node objective; disabled pending all equivalence checks."""
import torch


def weighted_node_policy_loss(old_log_prob, log_prob, positive_advantage, negative_advantage,
                              token_mask, leaf_count, clip_low=.2, clip_high=.2, dual_clip=3.):
    ratio = (log_prob-old_log_prob).exp()
    clipped = ratio.clamp(1-clip_low,1+clip_high)
    positive_ratio = torch.minimum(ratio,clipped)
    negative_ratio = torch.maximum(ratio,clipped)
    if dual_clip is not None:
        negative_ratio = negative_ratio.clamp_max(dual_clip)
    numerator = (-positive_advantage[:,None]*positive_ratio-negative_advantage[:,None]*negative_ratio)*token_mask
    denominator = (token_mask*leaf_count[:,None]).sum().clamp_min(1)
    return numerator.sum()/denominator
