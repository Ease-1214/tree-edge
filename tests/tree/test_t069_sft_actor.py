"""Small real actor-update tests without loading models or using any GPU."""
from types import SimpleNamespace, MethodType
import torch
from omegaconf import OmegaConf
from verl import DataProto
import verl.workers.actor.dp_actor as module


def actor_run(monkeypatch, enabled, diagnostic_component=None, fsdp_sync=False, active_sft=True):
    import verl.utils.debug.performance as performance
    monkeypatch.setattr(performance, "_get_current_mem_info", lambda: (0,0,0,0))
    monkeypatch.setattr(module, "get_torch_device",
                        lambda: SimpleNamespace(current_device=lambda: "cpu"))
    if fsdp_sync:
        # Exercise the sharded control flow without allocating GPUs or a real PG.
        monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 1)
        monkeypatch.setattr(torch.distributed, "get_rank", lambda: 0)
        monkeypatch.setattr(
            torch.distributed, "all_gather",
            lambda outputs, value: outputs[0].copy_(value),
        )
    actor = object.__new__(module.DataParallelPPOActor)
    actor._fsdp_sync_optional_forwards = fsdp_sync
    actor.config = OmegaConf.create(dict(
        edge_distillation=dict(enable=False),
        success_sft=dict(enable=enabled, coef=0.05),
        ppo_mini_batch_size=4,
        ppo_micro_batch_size_per_gpu=2,
        use_dynamic_bsz=False,
        use_kl_loss=False,
        ppo_epochs=1,
        clip_ratio=0.2,
        clip_ratio_low=None,
        clip_ratio_high=None,
        clip_ratio_c=3.0,
        entropy_coeff=0.0,
        loss_agg_mode="token-mean",
        policy_loss=dict(loss_mode="vanilla"),
    ))
    actor.use_fused_kernels = actor.use_ulysses_sp = False
    actor.actor_module = torch.nn.Linear(1, 4, bias=False)
    with torch.no_grad():
        actor.actor_module.weight[:] = torch.tensor([[0.1], [0.8], [-0.2], [0.0]])
    before = actor.actor_module.weight.detach().clone()
    actor.actor_optimizer = torch.optim.SGD(actor.actor_module.parameters(), lr=0.1)
    seen = []
    def forward(self, micro_batch, temperature, calculate_entropy=False,
                calculate_log_probs=True, token_selection_top_k=None,
                token_selection_ids=None):
        n, length = micro_batch["responses"].shape
        x = micro_batch["input_ids"][:, :1].float()
        logits = self.actor_module(x)[:,None,:].expand(n,length,4)
        lp = torch.log_softmax(logits,-1).gather(
            -1, micro_batch["responses"].unsqueeze(-1)).squeeze(-1)
        seen.append(x.detach().flatten().tolist())
        return None, lp, None, None
    actor._forward_micro_batch = MethodType(forward, actor)
    def optimizer_step(self, component=None):
        if component is not None:
            vector = torch.cat([p.grad.detach().flatten() for p in self.actor_module.parameters()])
            self._last_diagnostic_gradient = vector.detach().clone()
            return vector.norm()
        self._optimizer_steps_this_update += 1
        norm = torch.nn.utils.clip_grad_norm_(self.actor_module.parameters(), 1.)
        self.actor_optimizer.step()
        return norm
    actor._optimizer_step = MethodType(optimizer_step, actor)
    B,R,P=4,3,2
    inp=torch.ones(B,P+R,dtype=torch.long)
    inp[0]=3 # Tree success actually conditioned on T
    sft_inp=inp.clone()
    sft_inp[0]=2 # Student sees same history without T
    tensors=dict(
        responses=torch.ones(B,R,dtype=torch.long),
        input_ids=inp,
        attention_mask=torch.ones(B,P+R,dtype=torch.long),
        position_ids=torch.arange(P+R).expand(B,-1),
        old_log_probs=torch.zeros(B,R),
        advantages=torch.zeros(B,R), # naturally 4/4: no GRPO gradient
        edge_teacher_input_ids=inp,
        edge_teacher_attention_mask=torch.ones(B,P+R,dtype=torch.long),
        edge_teacher_position_ids=torch.arange(P+R).expand(B,-1),
        edge_distill_weight=torch.zeros(B),
        kd_student_input_ids=inp,
        kd_student_attention_mask=torch.ones(B,P+R,dtype=torch.long),
        kd_student_position_ids=torch.arange(P+R).expand(B,-1),
        kd_canonical=torch.zeros(B,dtype=torch.bool),
        kd_path_has_T=torch.zeros(B,dtype=torch.bool),
        is_padding_copy=torch.tensor([False,False,False,True]),
        sft_weight=torch.tensor([2.,1.,0.,0.]) if active_sft else torch.zeros(B),
        sft_student_input_ids=sft_inp,
        sft_student_attention_mask=torch.ones(B,P+R,dtype=torch.long),
        sft_student_position_ids=torch.arange(P+R).expand(B,-1),
    )
    meta={"temperature":1.0}
    if diagnostic_component is not None:
        meta["gradient_diagnostic_component"] = diagnostic_component
    metrics=actor.update_policy(DataProto.from_dict(
        tensors=tensors, meta_info=meta
    ))
    return before,actor.actor_module.weight.detach().clone(),seen,metrics


def test_actor_sft_updates_even_when_grpo_advantage_zero(monkeypatch):
    before,after,seen,metrics=actor_run(monkeypatch,True)
    assert not torch.equal(before,after)
    assert len(seen)==3 # Two policy forwards and ONE additional noexp SFT forward
    assert [2.] in seen
    assert metrics["sft/active_tokens"]==9.
    assert metrics["sft/active_rows"]==2.
    assert metrics["sft/loss_token_mean"]>0


def test_actor_sft_opt_in_off_preserves_zero_advantage_policy(monkeypatch):
    before,after,seen,metrics=actor_run(monkeypatch,False)
    torch.testing.assert_close(before,after)
    assert len(seen)==2
    assert "sft/active_tokens" in metrics and metrics["sft/active_tokens"]==0

def test_sft_diagnostic_is_standalone_and_does_not_step_optimizer(monkeypatch):
    before, after, seen, metrics = actor_run(monkeypatch, True, diagnostic_component="sft")
    torch.testing.assert_close(before,after)
    assert metrics["actor/grad_norm"][0] > 0
    assert metrics["sft/active_tokens"] == 9.
    assert [2.] in seen # no-experience SFT prompt remains part of measurement


def test_pg_diagnostic_omits_sft_even_when_successful(monkeypatch):
    before, after, seen, metrics = actor_run(monkeypatch, True, diagnostic_component="pg_sft")
    torch.testing.assert_close(before,after)
    assert metrics["actor/grad_norm"][0] == 0


def test_fsdp_sft_dummy_forward_is_zero_contribution(monkeypatch):
    # Simulate a rank with no active SFT rows while another rank has them.
    before, after, seen, metrics = actor_run(
        monkeypatch, True, fsdp_sync=True, active_sft=False
    )
    torch.testing.assert_close(before, after)
    assert len(seen) == 4  # Two policy + two zero-weight SFT placeholder forwards
    assert metrics["sft/active_tokens"] == 0.0


def test_fsdp_sft_real_rank_matches_unpadded_gradient(monkeypatch):
    # Simulate the active rank: an extra placeholder on the inactive microbatch
    # must leave its actual optimizer update unchanged.
    before, original_after, original_seen, original_metrics = actor_run(monkeypatch, True)
    before_sync, synced_after, synced_seen, synced_metrics = actor_run(
        monkeypatch, True, fsdp_sync=True
    )
    torch.testing.assert_close(before, before_sync)
    torch.testing.assert_close(original_after, synced_after)
    assert len(original_seen) == 3
    assert len(synced_seen) == 4
    assert original_metrics["sft/active_tokens"] == synced_metrics["sft/active_tokens"]


def test_fsdp_teacher_zero_gate_keeps_forward_collective(monkeypatch):
    # A zero-KD rank must still run the teacher model once for each microbatch.
    import verl.utils.debug.performance as performance
    monkeypatch.setattr(performance, "_get_current_mem_info", lambda: (0, 0, 0, 0))
    actor = object.__new__(module.DataParallelPPOActor)
    actor.config = OmegaConf.create({"edge_teacher_active_only": True})
    actor._fsdp_sync_optional_forwards = True
    seen = []

    def forward(self, micro_batch, temperature, calculate_entropy=False,
                calculate_log_probs=True, token_selection_top_k=None,
                token_selection_ids=None):
        seen.append(micro_batch["responses"].shape[0])
        n, r = micro_batch["responses"].shape
        return None, None, torch.ones(n, r, token_selection_top_k), torch.ones(
            n, r, token_selection_top_k, dtype=torch.long
        )

    actor._forward_micro_batch = MethodType(forward, actor)
    batch = {
        "edge_distill_weight": torch.zeros(2),
        "edge_teacher_input_ids": torch.ones(2, 5, dtype=torch.long),
        "edge_teacher_attention_mask": torch.ones(2, 5, dtype=torch.long),
        "edge_teacher_position_ids": torch.arange(5).expand(2, -1),
    }
    result = actor._edge_teacher_forward_active(
        batch, torch.ones(2, 3, dtype=torch.long), 1.0, top_k=20
    )
    assert seen == [1]
    assert result == (None, None, None)
    assert actor._edge_teacher_rows == 0
    assert actor._edge_teacher_rows_total == 2
