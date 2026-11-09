from types import SimpleNamespace, MethodType
import torch
import pytest
from omegaconf import OmegaConf
from verl import DataProto
import verl.workers.actor.dp_actor as module


@pytest.mark.parametrize(
    "active,has_T,expected_forwards",
    [(False, False, 2), (True, False, 3), (True, True, 4)],
)
def test_real_update_policy_joint_backward(
    monkeypatch, active, has_T, expected_forwards
):
    import verl.utils.debug.performance as performance

    monkeypatch.setattr(performance, "_get_current_mem_info", lambda: (0, 0, 0, 0))
    monkeypatch.setattr(
        module,
        "get_torch_device",
        lambda: SimpleNamespace(current_device=lambda: "cpu"),
    )
    actor = object.__new__(module.DataParallelPPOActor)
    actor.config = OmegaConf.create(
        dict(
            edge_distillation=dict(
                enable=True,
                token_selection="teacher_topk",
                top_k=2,
                coef=1.0,
                path_tier=2,
            ),
            edge_teacher_active_only=True,
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
        )
    )
    actor.use_fused_kernels = actor.use_ulysses_sp = False
    actor.actor_module = torch.nn.Linear(1, 4, bias=False)
    actor.actor_optimizer = torch.optim.SGD(actor.actor_module.parameters(), lr=0.01)
    calls = []
    backwards = []
    steps = []
    actor.actor_module.weight.register_hook(lambda grad: backwards.append(grad.clone()))

    def forward(
        self,
        micro_batch,
        temperature,
        calculate_entropy=False,
        calculate_log_probs=True,
        token_selection_top_k=None,
        token_selection_ids=None,
    ):
        n, R = micro_batch["responses"].shape
        x = micro_batch["input_ids"][:, 0:1].float()
        logits = self.actor_module(x)[:, None, :].expand(n, R, 4)
        lp = (
            torch.log_softmax(logits, -1)
            .gather(-1, micro_batch["responses"].unsqueeze(-1))
            .squeeze(-1)
        )
        calls.append(
            dict(grad=torch.is_grad_enabled(), rows=n, input=x.detach().clone())
        )
        selected = ids = None
        if token_selection_top_k is not None:
            ids = (
                logits.topk(token_selection_top_k, -1).indices
                if token_selection_ids is None
                else token_selection_ids
            )
            selected = logits.gather(-1, ids)
            self._last_selected_mass = torch.softmax(logits, -1).gather(-1, ids).sum(-1)
        return None, lp, selected, ids

    actor._forward_micro_batch = MethodType(forward, actor)

    def optimizer_step(self):
        steps.append(1)
        self._optimizer_steps_this_update += 1
        norm = torch.nn.utils.clip_grad_norm_(self.actor_module.parameters(), 1.0)
        self.actor_optimizer.step()
        return norm

    actor._optimizer_step = MethodType(optimizer_step, actor)
    B, R, P = 4, 3, 2
    tensors = dict(
        responses=torch.ones(B, R, dtype=torch.long),
        input_ids=torch.ones(B, P + R, dtype=torch.long),
        attention_mask=torch.ones(B, P + R, dtype=torch.long),
        position_ids=torch.arange(P + R).expand(B, -1),
        old_log_probs=torch.zeros(B, R),
        advantages=torch.ones(B, R),
        edge_teacher_input_ids=torch.full((B, P + R), 3, dtype=torch.long),
        edge_teacher_attention_mask=torch.ones(B, P + R, dtype=torch.long),
        edge_teacher_position_ids=torch.arange(P + R).expand(B, -1),
        edge_distill_weight=torch.tensor([0.5 if active else 0.0, 0.0, 0.0, 0.0]),
        kd_student_input_ids=torch.full((B, P + R), 2, dtype=torch.long),
        kd_student_attention_mask=torch.ones(B, P + R, dtype=torch.long),
        kd_student_position_ids=torch.arange(P + R).expand(B, -1),
        kd_canonical=torch.tensor([active, False, False, False]),
        kd_path_has_T=torch.tensor([has_T, False, False, False]),
        is_padding_copy=torch.zeros(B, dtype=torch.bool),
    )
    metrics = actor.update_policy(
        DataProto.from_dict(tensors=tensors, meta_info={"temperature": 1.0})
    )
    assert len(calls) == expected_forwards
    assert len(steps) == 1 and len(backwards) == 2
    assert metrics["actor/optimizer_steps_per_train_step"] == [1]
    assert metrics["perf/edge_teacher_row_fraction"] == (0.25 if active else 0.0)
    if active:
        teacher = [c for c in calls if not c["grad"]]
        assert len(teacher) == 1 and teacher[0]["rows"] == 1
        assert all(c["grad"] for c in calls if not torch.all(c["input"] == 3))
        if has_T:
            assert any(torch.all(c["input"] == 2) and c["grad"] for c in calls)
    else:
        assert all(c["grad"] for c in calls)
