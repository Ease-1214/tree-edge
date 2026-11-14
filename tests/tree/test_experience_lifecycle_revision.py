"""Regression coverage for stable experience IDs and EDGE-compatible eviction."""
import json
from types import SimpleNamespace

import pytest
from omegaconf import OmegaConf

from agent_system.memory.experience_memory import ExperienceMemory
from agent_system.memory.experience_updater import ExperienceUpdater
from agent_system.tree.core import QuestionEvidence
from agent_system.tree.training import restore_tree_state


def memory():
    return ExperienceMemory(load_initial_experiences=False, retrieval_mode="template")


def item(name, **meta):
    return {"title": name, "principle": "Action for " + name, "task_type": "general", **meta}


def test_task_step_ids_stay_unique_after_last_id_deletion_and_file_roundtrip(tmp_path):
    m = memory()
    assert m.upsert_experience_groups(
        task_experiences=[item("task1")], step_experiences=[item("step1")]
    ) == 2
    assert m.experiences["task_experiences"][0]["experience_id"] == "task_000001"
    assert m.experiences["step_experiences"][0]["experience_id"] == "step_000001"
    m.remove_experience("task_000001")
    m.remove_experience("step_000001")
    assert m.upsert_experience_groups(
        task_experiences=[item("task2")], step_experiences=[item("step2")]
    ) == 2
    assert m.experiences["task_experiences"][0]["experience_id"] == "task_000002"
    assert m.experiences["step_experiences"][0]["experience_id"] == "step_000002"

    path = tmp_path / "bank.json"
    m.save_experiences(str(path))
    restarted = ExperienceMemory(str(path), retrieval_mode="template")
    restarted.remove_experience("task_000002")
    restarted.remove_experience("step_000002")
    restarted.upsert_experience_groups(
        task_experiences=[item("task3")], step_experiences=[item("step3")]
    )
    assert restarted.experiences["task_experiences"][0]["experience_id"] == "task_000003"
    assert restarted.experiences["step_experiences"][0]["experience_id"] == "step_000003"


def test_stale_explicit_id_and_exp_prefix_are_never_reused():
    m = memory()
    m.upsert_experience_groups(task_experiences=[item("old task")])
    m.remove_experience("task_000001")
    m.upsert_experience_groups(task_experiences=[item("new task", experience_id="task_000001")])
    assert m.experiences["task_experiences"][0]["experience_id"] == "task_000002"

    m.upsert_experience_groups(step_experiences=[item("legacy old", experience_id="exp_000017")])
    m.remove_experience("exp_000017")
    updater = object.__new__(ExperienceUpdater)
    assert updater._next_exp_index(m.experiences) == 18
    m.upsert_experience_groups(step_experiences=[item("legacy new", experience_id="exp_000017")])
    assert m.experiences["step_experiences"][0]["experience_id"] == "exp_000018"


def test_restore_legacy_checkpoint_uses_history_to_advance_id_watermark(tmp_path):
    m = memory()
    # Historical evidence contains deleted IDs that are absent from the bank.
    old_bank = {
        "version": 3, "metadata": {},
        "task_experiences": [item("still here", experience_id="task_000003")],
        "step_experiences": [],
    }
    state = dict(
        checkpoint_step=20, bank=old_bank,
        evidence={"observations": {"task_000047": {"q": [-1]}}, "pending": {}, "contrasts": []},
    )
    checkpoint = tmp_path / "global_step_20"
    checkpoint.mkdir()
    (checkpoint / "tree_state.json").write_text(json.dumps(state))
    trainer = SimpleNamespace(
        config=OmegaConf.create({
            "env": {"tree_experience": {"enable": True}},
            "trainer": {"default_local_dir": str(tmp_path)},
        }),
        global_steps=20, envs=SimpleNamespace(retrieval_memory=m),
    )
    restore_tree_state(trainer)
    m.upsert_experience_groups(task_experiences=[item("brand new")])
    assert m.experiences["task_experiences"][-1]["experience_id"] == "task_000048"
    assert trainer.tree_evidence.state["observations"]["task_000047"] == {"q": [-1]}


def test_edge_hard_delete_uses_all_four_guards_and_requires_consecutive_windows():
    m = memory()
    m.upsert_experience_groups(task_experiences=[item("harmful", created_at_step=10)])
    x = m.experiences["task_experiences"][0]
    x["retrieval_count"] = 4
    e = QuestionEvidence()
    for step in (20, 25, 30):
        e.record({(x["experience_id"], str(step)): -1})
        e.merge_window(m.experiences, step, beta=0.1, management_interval_steps=5)
        removed = m.evict_excess_experiences(
            step, max_task_experiences=200, max_step_experiences=200,
            min_exposures=5, utility_threshold=-0.25, negative_windows=2,
            protect_recent_steps=20,
        )["removed"]
        assert not removed  # even negative + 3 windows cannot pass exposure guard
    assert x["utility"] == pytest.approx(-0.271)
    assert x["evidence"]["consecutive_negative_windows"] == 3
    x["retrieval_count"] = 5
    removed = m.evict_excess_experiences(
        30, max_task_experiences=200, max_step_experiences=200,
        min_exposures=5, utility_threshold=-0.25, negative_windows=2,
        protect_recent_steps=20,
    )["removed"]
    assert len(removed) == 1
    assert removed[0]["reason"] == "hard_negative_utility"


def test_new_experience_protection_and_positive_window_block_eviction():
    m = memory()
    m.upsert_experience_groups(task_experiences=[item("protected", created_at_step=20)])
    x = m.experiences["task_experiences"][0]
    x["retrieval_count"] = 10
    e = QuestionEvidence()
    for step in (25, 30, 35):
        e.record({(x["experience_id"], str(step)): -1})
        e.merge_window(m.experiences, step, beta=0.1, management_interval_steps=5)
    args = dict(max_task_experiences=200, max_step_experiences=200,
                min_exposures=5, utility_threshold=-0.25,
                negative_windows=2, protect_recent_steps=20)
    assert m.evict_excess_experiences(35, **args)["removed"] == []
    # A neutral window ends the negative sequence, even while Utility stays negative.
    e.record({(x["experience_id"], "q40"): 0})
    e.merge_window(m.experiences, 40, beta=0.1, management_interval_steps=5)
    assert x["evidence"]["consecutive_negative_windows"] == 0
    assert m.evict_excess_experiences(40, **args)["removed"] == []


def test_tree_manager_reports_negative_and_capacity_eviction_separately(tmp_path):
    from agent_system.tree.training import manage_tree_experiences
    from test_tree_method import example

    m = memory()
    m.upsert_experience_groups(task_experiences=[item("harmful", created_at_step=0)])
    x = m.experiences["task_experiences"][0]
    x["utility"] = -0.3
    x["retrieval_count"] = 10
    x["evidence"] = {
        "consecutive_negative_windows": 1,
        "last_negative_management_step": 25,
    }
    evidence = QuestionEvidence()
    evidence.record({(x["experience_id"], "q30"): -1})
    trainer = SimpleNamespace(
        config=OmegaConf.create({
            "env": {
                "tree_experience": {
                    "utility_update_interval": 5,
                    "utility_ema_beta": 0.1,
                    "protect_recent_steps": 20,
                },
                "experience_memory": {"management": {}},
            },
            "trainer": {"default_local_dir": str(tmp_path)},
        }),
        global_steps=30,
        tree_evidence=evidence,
        envs=SimpleNamespace(retrieval_memory=m),
        _sync_experiences_to_retrieval_server=lambda *args, **kwargs: None,
    )
    tree = example((0,), (0,))
    tree.metrics = {}
    metrics = manage_tree_experiences(trainer, tree)
    assert metrics["bank/deleted_negative"] == 1
    assert metrics["bank/deleted_capacity"] == 0
    assert metrics["bank/size"] == 0
    report = json.loads((tmp_path / "tree_management_step30.json").read_text())
    assert report["deleted_negative"][0]["reason"] == "hard_negative_utility"
    assert not report["deleted_capacity"]["removed"]



def test_tree_only_override_eviction_threshold_and_baseline_untouched():
    from pathlib import Path
    root = Path(__file__).resolve().parents[2]
    script = (root / "examples/grpo_trainer/run_t069.sh").read_text()
    assert script.count("env.experience_memory.management.eviction_utility_threshold=-0.1") == 1
    tree_section = script.split('if [[ "$MODE" == tree ]]; then', 1)[1].split("\nfi\n", 1)[0]
    assert "env.experience_memory.management.eviction_utility_threshold=-0.1" in tree_section
    assert "env.experience_memory.management.eviction_utility_threshold=-0.1" not in script.split('if [[ "$MODE" == tree ]]; then', 1)[0]
    config = (root / "verl/trainer/config/ppo_trainer.yaml").read_text()
    assert "eviction_utility_threshold: -0.25" in config


def test_negative_point_one_threshold_triggers_at_original_timing():
    m = memory()
    m.upsert_experience_groups(task_experiences=[item("harmful", created_at_step=0)])
    x = m.experiences["task_experiences"][0]
    x["retrieval_count"] = 5
    x["utility"] = -0.12
    x["evidence"] = {"consecutive_negative_windows": 2, "last_negative_management_step": 20}
    common = dict(
        max_task_experiences=200, max_step_experiences=200,
        protect_recent_steps=20, min_exposures=5, negative_windows=2,
    )
    assert m.evict_excess_experiences(
        20, utility_threshold=-0.25, **common
    )["removed"] == []
    deleted = m.evict_excess_experiences(
        20, utility_threshold=-0.1, **common
    )["removed"]
    assert len(deleted) == 1
    assert deleted[0]["reason"] == "hard_negative_utility"
