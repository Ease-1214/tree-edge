"""Explicit gradient synchronization for full FSDP replicas with a singleton shard group."""
import hashlib
import torch
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, ShardingStrategy

def uses_explicit_replica_sync(module):
    if not dist.is_initialized() or dist.get_world_size() == 1:
        return False
    if not isinstance(module, FSDP):
        return False
    return module.sharding_strategy == ShardingStrategy.NO_SHARD and dist.get_world_size(module.process_group) == 1

def summed_denominators(response_mask, weights, kd_weights):
    """One collective per minibatch, independent of each rank's microbatch count."""
    values = torch.stack((
        (response_mask * weights.unsqueeze(-1)).sum(),
        (response_mask * (kd_weights > 0).unsqueeze(-1)).sum(),
    )).to(dtype=torch.float64)
    dist.all_reduce(values, op=dist.ReduceOp.SUM)
    return values.clamp_min(1)

def average_replica_gradients(module):
    """Missing local gradients contribute zero; a globally unused parameter keeps grad=None."""
    parameters = [p for p in module.parameters() if p.requires_grad]
    if not parameters:
        raise RuntimeError("Explicit replica synchronization found no trainable parameters")
    device = parameters[0].device
    present = torch.tensor([p.grad is not None for p in parameters], device=device, dtype=torch.int32)
    dist.all_reduce(present, op=dist.ReduceOp.SUM)
    active = [p for p, count in zip(parameters, present.tolist()) if count]
    if not active:
        return
    gradients = torch.cat([
        p.grad.detach().reshape(-1).float() if p.grad is not None
        else torch.zeros(p.numel(), device=device, dtype=torch.float32)
        for p in active
    ])
    dist.all_reduce(gradients, op=dist.ReduceOp.SUM)
    gradients.div_(dist.get_world_size())
    offset = 0
    for parameter in active:
        gradient = gradients[offset:offset + parameter.numel()].view_as(parameter)
        # FSDP no_sync retains gradients in its compute dtype (bf16), while
        # master LoRA parameters and Adam moments remain fp32. copy_ alone
        # preserves the wrong grad dtype and makes foreach AdamW fail.
        if parameter.grad is None or parameter.grad.dtype != parameter.dtype:
            parameter.grad = gradient.to(parameter.dtype).clone()
        else:
            parameter.grad.copy_(gradient)
        offset += parameter.numel()

def verify_replica_parameters(module):
    digest = hashlib.sha256()
    for name, parameter in module.named_parameters():
        if parameter.requires_grad:
            digest.update(name.encode())
            digest.update(parameter.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes())
    local = digest.hexdigest()
    hashes = [None] * dist.get_world_size()
    dist.all_gather_object(hashes, local)
    if len(set(hashes)) != 1:
        raise RuntimeError(f"T069 replica parameter hashes differ: {hashes}")
    return local
