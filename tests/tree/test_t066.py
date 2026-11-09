import copy
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
from omegaconf import OmegaConf
from test_tree_method import example
from agent_system.tree.core import (
    fork_reflector_examples,
    select_reflector_examples,
    QuestionEvidence,
)
from agent_system.tree.training import (
    manage_tree_experiences,
    write_new_experiences,
    save_tree_checkpoint,
    restore_tree_state,
)
from agent_system.memory.experience_memory import ExperienceMemory
from agent_system.memory.experience_updater import ExperienceUpdater


def test_reverse_only_reflector_and_weak_excluded():
    reverse = example((1,), (0,))
    assert not reverse.audits[0]["gate"]
    assert fork_reflector_examples(reverse)[0]["effect"] == "harmed"
    assert not fork_reflector_examples(example((1, 0), (1, 1)))
    assert not fork_reflector_examples(example((0,), (1,), same=True))


def test_window_distinct_questions_and_top_ten():
    base = fork_reflector_examples(example((0,), (1,)))[0]
    values = []
    for i in range(15):
        item = copy.deepcopy(base)
        item.update(question_id=str(i), fork_id=str(i), delta=i / 20)
        values.append(item)
    duplicate = copy.deepcopy(values[-1])
    duplicate.update(fork_id="stronger", delta=1.0)
    selected = select_reflector_examples(values + [duplicate])
    assert len(selected) == 10 and selected[0]["fork_id"] == "stronger"
    assert len({x["question_id"] for x in selected}) == 10


def memory(tmp_path):
    path = tmp_path / "bank.json"
    path.write_text(json.dumps(dict(task_experiences=[], step_experiences=[])))
    return ExperienceMemory(str(path), retrieval_mode="embedding", device="cpu")


def generated(kind, text):
    return dict(
        experience_type=kind,
        task_type="direct_retrieval",
        trigger=text,
        recommended_action="Search " + text,
        retrieval_text=text,
        evidence={"source_traj_uids": ["real-leaf"]},
    )


def test_split_write_cache_remote_retrieval_and_duplicate(tmp_path, monkeypatch):
    m = memory(tmp_path)
    m._task_experience_embeddings_cache = {"stale": True}
    result = write_new_experiences(
        m, [generated("task", "task strategy"), generated("step", "step strategy")], 5
    )
    assert result["reflector/added_task"] == result["reflector/added_step"] == 1
    assert m._task_experience_embeddings_cache is None
    task = m.experiences["task_experiences"][0]
    step = m.experiences["step_experiences"][0]
    assert task["experience_id"] != step["experience_id"]
    assert all(
        x["utility"] == 0 and x["created_at_step"] == 5 and x["retrieval_count"] == 0
        for x in (task, step)
    )
    calls = []
    monkeypatch.setenv("EDGE_EMBED_URL", "http://127.0.0.1:29700")

    def remote(url, texts, normalize):
        calls.append((url, texts))
        return np.ones((len(texts), 4), dtype=np.float32) / 2

    monkeypatch.setattr(m, "_encode_remote", remote)
    retrieved = m.retrieve_task_experiences_batch(["task strategy"], top_k=1)
    assert task["experience_id"] in retrieved[0]["experience_ids"]
    assert calls and m._task_experience_embeddings_cache is not None
    task["utility"] = -0.2
    task["retrieval_count"] = 7
    result = write_new_experiences(m, [generated("task", "task strategy")], 10)
    assert result["bank/added"] == 0
    assert (
        task["utility"] == -0.2
        and task["retrieval_count"] == 7
        and task["created_at_step"] == 5
    )


def test_fallback_type_and_context():
    updater = object.__new__(ExperienceUpdater)
    source = fork_reflector_examples(example((0,), (1,)))[0]
    item = generated("invalid", "generic")
    item["source_examples"] = [1]
    output = updater._postprocess_search_experiences([item], [source])
    assert output[0]["experience_type"] == "task"
    source["fork_depth"] = 2
    item = generated("invalid", "another")
    item.pop("experience_type")
    item["source_examples"] = [1]
    assert (
        updater._postprocess_search_experiences([item], [source])[0]["experience_type"]
        == "step"
    )
    prompt = updater._format_search_example(1, source, {})
    assert "Common earlier task experience" in prompt and "1/1" in prompt


def test_no_candidate_skips_and_checkpoint_keeps_window(tmp_path):
    m = memory(tmp_path)
    trainer = SimpleNamespace(
        config=OmegaConf.create(
            {
                "env": {
                    "tree_experience": {
                        "enable": True,
                        "utility_update_interval": 5,
                        "utility_ema_beta": 0.3,
                        "min_delete_questions": 20,
                        "protect_recent_steps": 20,
                    },
                    "experience_memory": {"update_threshold": 0.3},
                },
                "trainer": {"default_local_dir": str(tmp_path)},
            }
        ),
        global_steps=5,
        envs=SimpleNamespace(retrieval_memory=m),
        tree_evidence=QuestionEvidence(),
        _sync_experiences_to_retrieval_server=lambda *a, **kw: None,
    )
    tree = example((0,), (0,))
    tree.metrics = {}
    output = manage_tree_experiences(trainer, tree)
    assert output["reflector/skip_reason"] == 1 and output["reflector/submitted"] == 0
    assert not hasattr(trainer, "experience_updater")
    trainer.tree_evidence.state["contrasts"] = fork_reflector_examples(
        example((0,), (1,))
    )
    write_new_experiences(m, [generated("task", "task"), generated("step", "step")], 5)
    before = copy.deepcopy(trainer.tree_evidence.state)
    bank = copy.deepcopy(m.experiences)
    save_tree_checkpoint(trainer, tmp_path / "global_step_5")
    trainer.tree_evidence = QuestionEvidence()
    m.experiences = {"task_experiences": [], "step_experiences": []}
    restore_tree_state(trainer)
    assert trainer.tree_evidence.state == before and m.experiences == bank


@pytest.mark.parametrize("step", [100, 101, 200])
def test_full_tree_route_at_late_step(step, tmp_path, monkeypatch):
    from agent_system.multi_turn_rollout.rollout_loop import TrajectoryCollector
    import agent_system.tree.runner as runner

    cfg = OmegaConf.create({"env": {"tree_experience": {"enable": True}}})
    collector = TrajectoryCollector(cfg, None)
    sentinel = object()
    monkeypatch.setattr(runner, "tree_multi_turn_loop", lambda *args: sentinel)
    batch = SimpleNamespace(meta_info={"global_step": step})
    assert collector.multi_turn_loop(batch, None, None, is_train=True) is sentinel
    source = Path(runner.__file__).parent.parent / "multi_turn_rollout/rollout_loop.py"
    assert "TREE_PHASE_SWITCH" not in source.read_text()


def test_wandb_step_commits_immediately():
    from verl.utils.tracking import Tracking
    calls=[]
    tracker=object.__new__(Tracking)
    tracker._finished_del=True
    tracker.logger={"wandb":SimpleNamespace(log=lambda **kwargs:calls.append(kwargs))}
    tracker.log({"metric":1},step=1)
    assert calls==[{"data":{"metric":1},"step":1,"commit":True}]
