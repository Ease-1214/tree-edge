"""T70 only: assert launch overrides and 5 outer-step LR schedule.

No server connections, model loads, GPU operations or training are started.
"""
from pathlib import Path
import torch
from verl.utils.torch_functional import get_constant_schedule_with_warmup

ROOT = Path(__file__).resolve().parents[2]

def test_t70_launcher_isolated_and_explicit_optimizer_override():
    script = (ROOT / "examples/grpo_trainer/run_t70.sh").read_text()
    assert "ROOT=/home/zehao/echo-collab/T070" in script
    assert 'T062_ROOT="$ROOT"' in script
    assert 'T062_ROOT="$RUN_DIR"' not in script
    assert "DATA_ROOT=/home/zehao/echo-collab/T069" in script
    assert '!= "t70"' in script
    assert "T70_RUN_DIR" in script
    assert 'T062_ROOT="$ROOT"' in script
    assert "WANDB_RUN_GROUP=t70" in script
    assert "WANDB_RESUME=${WANDB_RESUME:-never}" in script
    assert "actor_rollout_ref.actor.optim.lr=2e-5" in script
    assert "actor_rollout_ref.actor.optim.lr_warmup_steps=5" in script
    assert "actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.0" in script
    assert "actor_rollout_ref.actor.edge_distillation.coef=0.05" in script
    assert "actor_rollout_ref.actor.edge_distillation.normalization=active_kd_tokens" in script
    assert "actor_rollout_ref.actor.success_sft.coef=0.05" in script
    assert "trainer.val_before_train=True trainer.test_freq=10" in script
    assert "+trainer.t70_test6125_freq=50" in script
    assert "+trainer.t70_test6125_file=" in script
    assert "data.val_batch_size=128" in script

    assert "env.tree_experience.min_delete_questions" not in script
    assert "env.tree_experience.utility_ema_beta=0.1" in script
    assert "env.experience_memory.management.eviction_utility_threshold=-0.1" in script
    wrapper = (ROOT / "examples/grpo_trainer/run_t062_perf.sh").read_text()
    assert '"$@"' in wrapper

def test_actual_constant_scheduler_has_five_outer_step_warmup():
    p = torch.nn.Parameter(torch.tensor(1.))
    opt = torch.optim.AdamW([p], lr=2e-5)
    sched = get_constant_schedule_with_warmup(opt, num_warmup_steps=5)
    applied = []
    for _ in range(7):
        applied.append(opt.param_groups[0]["lr"])
        (p * p).backward()
        opt.step()
        opt.zero_grad()
        sched.step()
    expected = [0, 4e-6, 8e-6, 12e-6, 16e-6, 20e-6, 20e-6]
    assert all(abs(actual - target) < 1e-12 for actual, target in zip(applied, expected))
