"""The reflector failure signal must survive both legacy and dynamic-management callers."""
import json
from pathlib import Path
from types import SimpleNamespace
from omegaconf import OmegaConf
from verl.trainer.ppo.ray_trainer import RayPPOTrainer

def trainer(tmp_path,success=False):
    return SimpleNamespace(
        global_steps=5,
        config=OmegaConf.create(dict(env=dict(experience_memory=dict(update_threshold=.3,max_failed_trajectories=10)),
            trainer=dict(default_local_dir=str(tmp_path)))),
        experience_updater=SimpleNamespace(
            model="fake",last_update_succeeded=success,last_failure_reason=None if success else "TimeoutError",
            analyze_failures=lambda **kwargs:[]))

def test_reflector_failure_is_recorded_and_returned(tmp_path):
    actor=trainer(tmp_path)
    metrics=RayPPOTrainer._update_experiences_from_contrasts(actor,
        contrast_examples=[dict(task="synthetic")],success_rate={"nq":.1},
        retrieval_memory=SimpleNamespace(experiences={}))
    assert metrics["reflector/failed"]==1
    assert metrics["reflector/submitted"]==1
    saved=json.loads((tmp_path/"reflector_status_step5.json").read_text())
    assert saved["failure_reason"]=="TimeoutError"

def test_valid_empty_response_is_not_failure(tmp_path):
    actor=trainer(tmp_path,success=True)
    metrics=RayPPOTrainer._update_experiences_from_contrasts(actor,
        contrast_examples=[dict(task="synthetic")],success_rate={"nq":.1},
        retrieval_memory=SimpleNamespace(experiences={}))
    assert metrics["reflector/failed"]==0

def test_baseline_threshold_is_preserved(tmp_path):
    actor=trainer(tmp_path)
    metrics=RayPPOTrainer._update_experiences_from_contrasts(actor,
        contrast_examples=[dict(task="synthetic")],success_rate={"nq":.5},
        retrieval_memory=SimpleNamespace(experiences={}))
    assert metrics["reflector/failed"]==0 and metrics["reflector/submitted"]==0
    assert metrics["reflector/skip_reason"]==2
    assert not (tmp_path/"reflector_status_step5.json").exists()
