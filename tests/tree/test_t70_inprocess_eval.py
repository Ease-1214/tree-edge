"""T70 synchronous Dev1000 / Test6125 validation on shared Actor GPUs."""
from pathlib import Path
import pytest
from omegaconf import OmegaConf
from agent_system.tree.t70_inprocess_eval import (
    t70_test6125_due, run_t70_test6125,
)


class FakeTrainer:
    def __init__(self):
        self.global_steps = 0
        self.val_dataloader = object()
        self.t70_test6125_dataloader = object()
        self.config = OmegaConf.create({"trainer": {"t70_test6125_freq": 50}})
        self.history = []

    def _validate(self):
        self.history.append(self.val_dataloader)
        return {"val/strict_em": 0.3, "val/nq_success_rate": 0.4,
                "val/bamboogle_success_rate": 0.2, "val/macro_em": 0.3}


@pytest.mark.parametrize(
    "step,last,expected",
    [(0,False,False),(0,True,False),(1,False,False),(10,False,False),
     (49,False,False),(50,False,True),(100,False,True),
     (199,False,False),(200,True,True),(97,True,True)],
)
def test_6125_schedule(step,last,expected):
    trainer = FakeTrainer()
    trainer.global_steps = step
    assert t70_test6125_due(trainer,is_last_step=last) == expected


def test_6125_runs_in_same_trainer_and_restores_dev_loader():
    trainer = FakeTrainer()
    trainer.global_steps = 50
    original = trainer.val_dataloader
    metrics = run_t70_test6125(trainer)
    assert trainer.history == [trainer.t70_test6125_dataloader]
    assert trainer.val_dataloader is original
    assert metrics["test6125/strict_em"] == 0.3
    assert metrics["test6125/macro_em"] == 0.3
    assert metrics["test6125/checkpoint_step"] == 50
    assert all(key.startswith("test6125/") for key in metrics)


def test_inprocess_validation_restores_loader_on_failure():
    trainer = FakeTrainer()
    original = trainer.val_dataloader
    def fail():
        raise RuntimeError("GPU error")
    trainer._validate = fail
    with pytest.raises(RuntimeError,match="GPU error"):
        run_t70_test6125(trainer)
    assert trainer.val_dataloader is original


def test_t70_launcher_no_external_eval_process():
    root = Path(__file__).resolve().parents[2]
    script = (root/"examples/grpo_trainer/run_t70.sh").read_text()
    assert "CUDA_VISIBLE_DEVICES=2,3,4,5" in script
    assert "trainer.val_before_train=True trainer.test_freq=10" in script
    assert "+trainer.t70_test6125_freq=50" in script
    assert "data.val_batch_size=128" in script
    assert "T069_RUN_DIR=" not in script
    assert not (root/"examples/grpo_trainer/run_t70_eval_queue.sh").exists()
