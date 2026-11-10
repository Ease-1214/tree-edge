"""Read-only PG/KD gradient measurement on one local microbatch per replica."""
from copy import deepcopy
import json
import os
from pathlib import Path
import torch
import torch.distributed as dist
from verl import DataProto
from verl.utils.seqlen_balancing import rearrange_micro_batches

def measure_actor_gradients(actor, data):
    batch=data.batch
    if "dedup_mini" in batch:
        batch=batch[batch["dedup_mini"] == batch["dedup_mini"].min()]
    else:
        batch=batch[:actor.config.ppo_mini_batch_size]
    if actor.config.use_dynamic_bsz:
        micros,_=rearrange_micro_batches(batch=batch,
            max_token_len=actor.config.ppo_max_token_len_per_gpu * actor.ulysses_sequence_parallel_size)
    else:
        micros=batch.split(actor.config.ppo_micro_batch_size_per_gpu)
    selected=micros[0]
    for micro in micros:
        weights=micro.get("edge_distill_weight",None)
        if weights is not None:
            active=weights>0
            if "kd_canonical" in micro:
                active=active & micro["kd_canonical"] & (~micro["is_padding_copy"])
            if bool(active.any()):
                selected=micro
                break
    cpu_rng=torch.get_rng_state()
    cuda_rng=torch.cuda.get_rng_state() if torch.cuda.is_available() else None
    counters={name:getattr(actor,name,None) for name in
        ["_optimizer_steps_this_update","_edge_teacher_rows","_edge_teacher_rows_total","_policy_updates_seen"]}
    vectors={}
    try:
        for component in ["pg","kd"]:
            torch.set_rng_state(cpu_rng)
            if cuda_rng is not None:torch.cuda.set_rng_state(cuda_rng)
            metadata=deepcopy(data.meta_info)
            metadata["gradient_diagnostic_component"]=component
            probe=DataProto.from_dict(tensors={k:v.clone() for k,v in selected.items()},meta_info=metadata)
            actor.update_policy(probe)
            vectors[component]=actor._last_diagnostic_gradient
            del actor._last_diagnostic_gradient
    finally:
        actor.actor_optimizer.zero_grad(set_to_none=True)
        torch.set_rng_state(cpu_rng)
        if cuda_rng is not None:torch.cuda.set_rng_state(cuda_rng)
        for name,value in counters.items():
            if value is None:
                if hasattr(actor,name):delattr(actor,name)
            else:setattr(actor,name,value)
    pg=vectors["pg"].float()
    kd=vectors["kd"].float()
    pg_norm=torch.linalg.vector_norm(pg)
    kd_norm=torch.linalg.vector_norm(kd)
    cosine=float(torch.dot(pg,kd)/(pg_norm*kd_norm)) if pg_norm>0 and kd_norm>0 else None
    result=dict(step=int(data.meta_info["global_step"]),pg_grad_norm=float(pg_norm),
        weighted_kd_grad_norm=float(kd_norm),
        kd_to_pg_grad_norm=float(kd_norm/pg_norm) if pg_norm>0 else None,
        cosine=cosine,kd_coefficient=float(actor.config.edge_distillation.get("coef",1.)),
        scope="One local microbatch per rank from first minibatch, preferring a KD-active microbatch; gradients globally averaged before measuring.")
    if dist.get_rank()==0:
        directory=Path(data.meta_info["gradient_diagnostic_dir"])
        directory.mkdir(parents=True,exist_ok=True)
        target=directory/f"gradient-step{result['step']}.json"
        temporary=target.with_suffix(".tmp")
        temporary.write_text(json.dumps(result,indent=2))
        os.replace(temporary,target)
        print("T069_GRAD_DIAGNOSTIC "+json.dumps(result),flush=True)
    return result
