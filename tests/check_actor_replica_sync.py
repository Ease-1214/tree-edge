"""Exercise the production Actor update entry point across four full replicas."""
import argparse,copy,json,os,sys
from pathlib import Path
from types import SimpleNamespace,MethodType
from contextlib import nullcontext
import torch
import torch.distributed as dist
from omegaconf import OmegaConf
from verl import DataProto
from verl.workers.actor import dp_actor
from verl.trainer.ppo.core_algos import compute_policy_loss,compute_edge_teacher_topk_distillation_loss

def forward(self,micro_batch,temperature,calculate_entropy=False,calculate_log_probs=True,
            token_selection_top_k=None,token_selection_ids=None):
    n,R=micro_batch["responses"].shape
    logits=self.actor_module(micro_batch["input_ids"][:,:1].float())[:,None,:].expand(n,R,4)
    lp=logits.log_softmax(-1).gather(-1,micro_batch["responses"].unsqueeze(-1)).squeeze(-1)
    selected=ids=None
    if token_selection_top_k is not None:
        ids=logits.topk(token_selection_top_k,-1).indices if token_selection_ids is None else token_selection_ids
        selected=logits.gather(-1,ids)
        self._last_selected_mass=logits.softmax(-1).gather(-1,ids).sum(-1)
    entropy=-(logits.softmax(-1)*logits.log_softmax(-1)).sum(-1) if calculate_entropy else None
    return entropy,lp,selected,ids

def batch(rank,device,dedup,baseline=False):
    B,R,P=[2,3,5,7][rank],3,2
    mask=torch.ones(B,P+R,dtype=torch.long,device=device)
    mask[::2,-1]=0
    kwargs=dict(responses=torch.ones(B,R,dtype=torch.long,device=device),
        input_ids=torch.full((B,P+R),rank+1,dtype=torch.long,device=device),
        attention_mask=mask,position_ids=torch.arange(P+R,device=device).expand(B,-1),
        old_log_probs=torch.full((B,R),-1.3,device=device),
        advantages=torch.linspace(-1,1,B,device=device)[:,None].expand(B,R),
        edge_teacher_input_ids=torch.full((B,P+R),3,dtype=torch.long,device=device),
        edge_teacher_attention_mask=mask.clone(),
        edge_teacher_position_ids=torch.arange(P+R,device=device).expand(B,-1),
        edge_distill_weight=torch.tensor([.5 if rank==0 else 0.]*B,device=device),
        kd_student_input_ids=torch.full((B,P+R),2,dtype=torch.long,device=device),
        kd_student_attention_mask=mask.clone(),
        kd_student_position_ids=torch.arange(P+R,device=device).expand(B,-1),
        kd_canonical=torch.full((B,),rank==0,dtype=torch.bool,device=device),
        kd_path_has_T=torch.full((B,),rank==0,dtype=torch.bool,device=device),
        is_padding_copy=torch.zeros(B,dtype=torch.bool,device=device))
    if baseline:
        for key in list(kwargs):
            if key.startswith("kd_") or key=="is_padding_copy":
                kwargs.pop(key)
    if dedup:
        weights=torch.arange(1,B+1,device=device).float()
        kwargs.update(w_tok=weights,A_pos=kwargs["advantages"][:,0].clamp_min(0)*weights,
            A_neg=kwargs["advantages"][:,0].clamp_max(0)*weights,
            dedup_mini=torch.zeros(B,dtype=torch.long,device=device),
            dedup_mini_weight=torch.ones(B,device=device))
    return DataProto.from_dict(tensors=kwargs,meta_info={"temperature":1.})

def main(gpu,baseline,diagnostic):
    if not baseline:
        from agent_system.tree.loss import tree_topk_loss
        from agent_system.tree.dedup import weighted_node_policy_loss
    rank=int(os.environ["RANK"])
    if gpu:torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    device=torch.device("cuda",int(os.environ["LOCAL_RANK"])) if gpu else torch.device("cpu")
    dist.init_process_group("nccl" if gpu else "gloo")
    dp_actor.get_torch_device=lambda:SimpleNamespace(current_device=lambda:device)
    import verl.utils.debug.performance as performance
    performance._get_current_mem_info=lambda:(0,0,0,0)
    for dedup in ([False] if baseline else [False,True]):
        torch.manual_seed(19)
        plain=torch.nn.Linear(1,4,bias=False).to(device)
        reference=copy.deepcopy(plain)
        if gpu:
            from torch.distributed.device_mesh import init_device_mesh
            from torch.distributed.fsdp import FullyShardedDataParallel as FSDP,ShardingStrategy
            mesh=init_device_mesh("cuda",(4,1),mesh_dim_names=["ddp","fsdp"])
            model=FSDP(plain,device_mesh=mesh,device_id=device,
                sharding_strategy=ShardingStrategy.HYBRID_SHARD,use_orig_params=False,sync_module_states=True)
        else:
            model=plain;model.no_sync=lambda:nullcontext()
        actor=object.__new__(dp_actor.DataParallelPPOActor)
        actor.actor_module=model
        actor.actor_optimizer=torch.optim.SGD(model.parameters(),lr=.01)
        actor._explicit_replica_sync=True
        actor.use_fused_kernels=actor.use_ulysses_sp=False
        actor._forward_micro_batch=MethodType(forward,actor)
        actor.config=OmegaConf.create(dict(edge_distillation=dict(enable=True,token_selection="teacher_topk",
            top_k=2,coef=.2,path_tier=2),edge_teacher_active_only=True,ppo_mini_batch_size=8,
            ppo_micro_batch_size_per_gpu=2,use_dynamic_bsz=False,use_kl_loss=False,ppo_epochs=1,
            clip_ratio=.2,clip_ratio_low=None,clip_ratio_high=None,clip_ratio_c=3.,
            entropy_coeff=0.,loss_agg_mode="token-mean",policy_loss=dict(loss_mode="vanilla"),grad_clip=100.))
        captured=[]
        for p in model.parameters():
            p.register_hook(lambda grad: captured.append(grad.detach().clone()))
        training_batch=batch(rank,device,dedup,baseline)
        if diagnostic:
            training_batch.meta_info.update(global_step=1,gradient_diagnostic_dir="/tmp/t069-gradient-entry-test")
        actor.update_policy(training_batch)
        all_batches=[batch(source,device,dedup,baseline) for source in range(4)]
        tensors={key:torch.cat([b.batch[key] for b in all_batches]) for key in all_batches[0].batch.keys()}
        mask=tensors["attention_mask"][:,-3:]
        teacher=SimpleNamespace(actor_module=reference)
        with torch.no_grad():
            _,_,tlogits,tids=forward(teacher,dict(tensors,input_ids=tensors["edge_teacher_input_ids"]),
                1.,token_selection_top_k=2)
        student=SimpleNamespace(actor_module=reference)
        _,log_probs,slogits,_=forward(student,tensors,1.,token_selection_top_k=2,token_selection_ids=tids)
        # All KD rows carry T, so their student forward must omit it.
        if not baseline:
            _,_,noexp,_=forward(student,dict(tensors,input_ids=tensors["kd_student_input_ids"]),
                1.,token_selection_top_k=2,token_selection_ids=tids)
            slogits=torch.where(tensors["kd_path_has_T"][:,None,None],noexp,slogits)
        if dedup:
            pg=weighted_node_policy_loss(tensors["old_log_probs"],log_probs,tensors["A_pos"],tensors["A_neg"],
                mask,tensors["w_tok"],.2,.2,3.)
            kd=tree_topk_loss(slogits,tlogits,mask,tensors["edge_distill_weight"],
                mask*tensors["w_tok"][:,None])[0]
        else:
            pg=compute_policy_loss(tensors["old_log_probs"],log_probs,tensors["advantages"],mask,
                cliprange=.2,cliprange_low=.2,cliprange_high=.2,clip_ratio_c=3.)[0]
            kd=(compute_edge_teacher_topk_distillation_loss(slogits,tlogits,mask,tensors["edge_distill_weight"])[0]
                if baseline else tree_topk_loss(slogits,tlogits,mask,tensors["edge_distill_weight"],mask)[0])
        # Both modes here are tree rows. Baseline has its own active-token denominator.
        reference_optimizer=torch.optim.SGD(reference.parameters(),lr=.01)
        (pg+.2*kd).backward();reference_optimizer.step()
        actual=torch.cat([p.detach().reshape(-1) for p in model.parameters()])
        expected=torch.cat([p.detach().reshape(-1) for p in reference.parameters()])
        error=float((actual-expected).abs().max())
        assert error<2e-7,(dedup,error,actual,expected)
        if rank==0:print("T069_ACTOR_ENTRY",json.dumps(dict(gpu=gpu,baseline=baseline,dedup=dedup,diagnostic=diagnostic,
            unequal_micro_counts=True,three_ranks_without_kd=True,max_parameter_difference=error)),flush=True)
    dist.destroy_process_group()

if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--gpu",action="store_true");parser.add_argument("--baseline",action="store_true")
    parser.add_argument("--diagnostic",action="store_true")
    args=parser.parse_args();main(args.gpu,args.baseline,args.diagnostic)
