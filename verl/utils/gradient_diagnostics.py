"""Read-only PG/KD/SFT gradient comparison on shared local microbatches.

Each of the four replicas computes its local microbatch gradients, which are
globally averaged before norm/cosine measurement. PG and KD share a KD-active
probe; PG and SFT share a separate SFT-active probe. This avoids comparing
SFT against a PG vector from a different sampling context.
"""
from copy import deepcopy
import json
import os
from pathlib import Path
import torch
import torch.distributed as dist
from verl import DataProto
from verl.utils.seqlen_balancing import rearrange_micro_batches


def select_probe_microbatches(actor, data):
    batch = data.batch
    if "dedup_mini" in batch:
        batch = batch[batch["dedup_mini"] == batch["dedup_mini"].min()]
    else:
        batch = batch[:actor.config.ppo_mini_batch_size]
    if actor.config.use_dynamic_bsz:
        micros, _ = rearrange_micro_batches(
            batch=batch,
            max_token_len=actor.config.ppo_max_token_len_per_gpu * actor.ulysses_sequence_parallel_size,
        )
    else:
        micros = batch.split(actor.config.ppo_micro_batch_size_per_gpu)
    kd_probe = sft_probe = micros[0]
    for micro in micros:
        weights = micro.get("edge_distill_weight")
        if weights is not None:
            active = weights > 0
            if "kd_canonical" in micro:
                active = active & micro["kd_canonical"] & (~micro["is_padding_copy"])
            if bool(active.any()):
                kd_probe = micro
                break
    for micro in micros:
        weights = micro.get("sft_weight")
        if weights is not None:
            active = weights > 0
            if "is_padding_copy" in micro:
                active = active & (~micro["is_padding_copy"])
            if bool(active.any()):
                sft_probe = micro
                break
    return kd_probe, sft_probe


def _cosine(left, right):
    a = torch.linalg.vector_norm(left)
    b = torch.linalg.vector_norm(right)
    return float(torch.dot(left, right) / (a * b)) if a > 0 and b > 0 else None


def measure_actor_gradients(actor, data):
    kd_micro, sft_micro = select_probe_microbatches(actor, data)
    cpu_rng = torch.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state() if torch.cuda.is_available() else None
    counters = {name: getattr(actor, name, None) for name in (
        "_optimizer_steps_this_update", "_edge_teacher_rows",
        "_edge_teacher_rows_total", "_policy_updates_seen",
    )}
    probes = (
        ("pg", "pg", kd_micro),
        ("kd", "kd", kd_micro),
        ("pg_sft", "pg_sft", sft_micro),
        ("sft", "sft", sft_micro),
    )
    vectors = {}
    try:
        for name, component, micro in probes:
            torch.set_rng_state(cpu_rng)
            if cuda_rng is not None:
                torch.cuda.set_rng_state(cuda_rng)
            metadata = deepcopy(data.meta_info)
            metadata["gradient_diagnostic_component"] = component
            probe = DataProto.from_dict(
                tensors={k: v.clone() for k, v in micro.items()},
                meta_info=metadata,
            )
            actor.update_policy(probe)
            vectors[name] = actor._last_diagnostic_gradient
            del actor._last_diagnostic_gradient
    finally:
        actor.actor_optimizer.zero_grad(set_to_none=True)
        torch.set_rng_state(cpu_rng)
        if cuda_rng is not None:
            torch.cuda.set_rng_state(cuda_rng)
        for name, value in counters.items():
            if value is None:
                if hasattr(actor, name):
                    delattr(actor, name)
            else:
                setattr(actor, name, value)
    pg, kd, pg_sft, sft = (vectors[x].float() for x in ("pg", "kd", "pg_sft", "sft"))
    pg_norm, kd_norm = torch.linalg.vector_norm(pg), torch.linalg.vector_norm(kd)
    pg_sft_norm, sft_norm = torch.linalg.vector_norm(pg_sft), torch.linalg.vector_norm(sft)
    kd_mask = kd_micro.get("loss_mask", kd_micro["attention_mask"])[:, -kd_micro["responses"].size(1):]
    kd_tokens = (kd_mask * (kd_micro["edge_distill_weight"] > 0).unsqueeze(-1)).sum().double()
    sft_mask = sft_micro.get("loss_mask", sft_micro["attention_mask"])[:, -sft_micro["responses"].size(1):]
    sft_tokens = (sft_mask * (sft_micro["sft_weight"] > 0).unsqueeze(-1)).sum().double()
    counts = torch.stack([kd_tokens, sft_tokens]).to(next(actor.actor_module.parameters()).device)
    dist.all_reduce(counts, op=dist.ReduceOp.SUM)
    result = dict(
        step=int(data.meta_info["global_step"]),
        pg_grad_norm=float(pg_norm),
        weighted_kd_grad_norm=float(kd_norm),
        kd_to_pg_grad_norm=float(kd_norm / pg_norm) if pg_norm > 0 else None,
        cosine=_cosine(pg, kd),
        kd_coefficient=float(actor.config.edge_distillation.get("coef", 1.)),
        sft_probe_pg_grad_norm=float(pg_sft_norm),
        weighted_sft_grad_norm=float(sft_norm),
        sft_to_pg_grad_norm=float(sft_norm / pg_sft_norm) if pg_sft_norm > 0 else None,
        sft_pg_cosine=_cosine(pg_sft, sft),
        sft_coefficient=float(actor.config.success_sft.get("coef", 0.)),
        kd_probe_active_tokens=float(counts[0]),
        sft_probe_active_tokens=float(counts[1]),
        scope=(
            "First minibatch, one KD-active and one SFT-active microbatch per rank; "
            "PG/KD and PG/SFT are paired within each probe; full-replica "
            "gradients averaged globally. This is not a full-step mean."
        ),
    )
    metrics = {
        "grad/pg_norm": result["pg_grad_norm"],
        "grad/kd_norm": result["weighted_kd_grad_norm"],
        "grad/kd_to_pg": result["kd_to_pg_grad_norm"],
        "grad/pg_kd_cosine": result["cosine"],
        "grad/sft_probe_pg_norm": result["sft_probe_pg_grad_norm"],
        "grad/sft_norm": result["weighted_sft_grad_norm"],
        "grad/sft_to_pg": result["sft_to_pg_grad_norm"],
        "grad/pg_sft_cosine": result["sft_pg_cosine"],
        "grad/kd_probe_active_tokens": result["kd_probe_active_tokens"],
        "grad/sft_probe_active_tokens": result["sft_probe_active_tokens"],
    }
    if dist.get_rank() == 0:
        directory = Path(data.meta_info["gradient_diagnostic_dir"])
        directory.mkdir(parents=True, exist_ok=True)
        target = directory / f"gradient-step{result['step']}.json"
        temporary = target.with_suffix(".tmp")
        temporary.write_text(json.dumps(result, indent=2))
        os.replace(temporary, target)
        print("T70_GRAD_DIAGNOSTIC " + json.dumps(result), flush=True)
    return {k: v for k, v in metrics.items() if v is not None}
