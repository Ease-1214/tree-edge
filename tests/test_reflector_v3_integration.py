"""Regression checks for selection + complete prompt. Run with pytest -q."""
import copy
import os
from pathlib import Path

import pytest

from agent_system.tree.core import trajectory_reflector_examples, normalize_action
from agent_system.memory.reflector_prompt_v3 import build_reflector_v3_prompt
from agent_system.memory.experience_updater import ExperienceUpdater

TREE_PATH = Path(os.environ.get(
    "REFLECTOR_TEST_TREE",
    "/home/zehao/echo-collab/T069/tv2/tree-private/tree-step80.pt",
))


@pytest.fixture(scope="module")
def sampled_tree():
    if not TREE_PATH.is_file():
        pytest.skip("saved real tree not found")
    import torch
    tree = torch.load(str(TREE_PATH), map_location="cpu", weights_only=False)
    return tree, trajectory_reflector_examples(tree, k=10, seed=80)


def test_real_forks_are_same_state_and_distinct_questions(sampled_tree):
    tree, examples = sampled_tree
    assert len(examples) == 10
    assert len({e["task"] for e in examples}) == len(examples)
    assert any(e["fork_depth"] == 1 for e in examples)
    assert all(e["source_leaf_ids"][0] != e["source_leaf_ids"][1] for e in examples)
    for e in examples:
        a = tree.nodes[e["failed_turns"][e["fork_depth"] - 1]["node_id"]]
        b = tree.nodes[e["success_turns"][e["fork_depth"] - 1]["node_id"]]
        assert a.parent_id == b.parent_id
        assert a.state_before.question == b.state_before.question
        assert normalize_action(a.action) != normalize_action(b.action)
        assert e["left_subtree_rate"] != e["right_subtree_rate"]


def test_selector_is_reproducible_and_limited(sampled_tree):
    tree, examples = sampled_tree
    again = trajectory_reflector_examples(tree, k=10, seed=80)
    assert [e["fork_id"] for e in examples] == [e["fork_id"] for e in again]
    assert len(trajectory_reflector_examples(tree, k=3, seed=80)) == 3


def test_prompt_keeps_tail_of_full_observation_and_bank(sampled_tree):
    tree, examples = sampled_tree
    rows = copy.deepcopy(examples)
    tail_marker = "NONTRUNCATED_VISIBLE_OBSERVATION_END_99887"
    rows[0]["success_turns"][-1]["observation"] += ("E" * 4500) + tail_marker
    prompt = build_reflector_v3_prompt(rows, tree.bank_snapshot, 1, max_new=5)
    assert tail_marker in prompt
    assert "CANDIDATE_IDS" in prompt
    assert "exp_000005" in prompt
    assert "COMPLETE EXAMPLES" in prompt
    import json
    assert all(json.dumps(x["task"], ensure_ascii=False)[1:-1] in prompt
               for x in examples)
    for pool in ("task_experiences", "step_experiences"):
        for x in tree.bank_snapshot[pool]:
            assert x["experience_id"] in prompt


def test_prompt_never_forces_t_and_supports_zero(sampled_tree):
    tree, examples = sampled_tree
    prompt = build_reflector_v3_prompt(examples, tree.bank_snapshot, 5, max_new=5)
    assert "does not automatically imply T" in prompt
    assert "Do not require both types" in prompt
    assert "0 is okay" in prompt
    assert all(f"exp_{i:06d}" in prompt for i in range(5,10))


def test_parser_accepts_five_without_api_call(monkeypatch):
    for name in ("ALL_PROXY","all_proxy","HTTP_PROXY","http_proxy","HTTPS_PROXY","https_proxy"):
        monkeypatch.delenv(name, raising=False)
    # Constructor without network or secret; parsing is a pure method.
    updater = ExperienceUpdater(
        api_key="unit-test-only", base_url="http://127.0.0.1:9/v1",
        model="unit-test", max_new_experiences_per_update=5,
    )
    import json
    items = [
        dict(experience_id=f"exp_{i:06d}", experience_type="step",
             task_type="direct_retrieval", trigger="test condition",
             recommended_action="search a missing relation",
             avoid_action="guess", expected_effect="answer correct relation",
             state_cues=["missing relation"],
             retrieval_text="test condition has missing relation",
             source_examples=[1])
        for i in range(1, 6)
    ]
    parsed = updater._parse_experiences_response(json.dumps(items))
    assert updater.last_parse_succeeded
    assert len(parsed) == 5


def test_v3_never_infers_experience_type_from_fork_depth(sampled_tree, monkeypatch):
    for name in ("ALL_PROXY", "all_proxy", "HTTP_PROXY", "http_proxy", "HTTPS_PROXY", "https_proxy"):
        monkeypatch.delenv(name, raising=False)
    updater = ExperienceUpdater(
        api_key="unit-test-only", base_url="http://127.0.0.1:9/v1",
        model="unit-test", max_new_experiences_per_update=5,
    )
    _, examples = sampled_tree
    exp = {
        "experience_id": "exp_000001",
        "task_type": "direct_retrieval",
        "trigger": "A source is missing its relation",
        "recommended_action": "Search entity and relation",
        "avoid_action": "Guess",
        "expected_effect": "Follow evidence",
        "state_cues": ["missing relation"],
        "retrieval_text": "Find the missing relation",
        "source_examples": [1],
    }
    result = updater._postprocess_search_experiences([exp], examples)
    assert result == []
    assert updater.last_invalid_v3_dropped == 1


def test_generated_t_and_s_can_enter_cloned_bank(sampled_tree):
    from agent_system.memory.experience_memory import ExperienceMemory
    from agent_system.tree.training import write_new_experiences
    from copy import deepcopy
    tree, examples = sampled_tree
    mem = ExperienceMemory(load_initial_experiences=False)
    mem.experiences = deepcopy(tree.bank_snapshot)
    mem._normalize_all_experience_meta()
    initial_t = len(mem.experiences["task_experiences"])
    initial_s = len(mem.experiences["step_experiences"])
    additions = []
    for kind in ("task", "step"):
        additions.append({
            "experience_id": "exp_000001",
            "experience_type": kind,
            "task_type": "multi_hop_reasoning",
            "trigger": "a synthetic test pattern for " + kind,
            "recommended_action": "search for [entity] plus [missing relation] and stop once supported for " + kind,
            "avoid_action": "guess a missing relation",
            "expected_effect": "a supported final answer",
            "state_cues": ["missing relation", kind],
            "retrieval_text": "unique synthetic retrieval matching rule for " + kind,
            "evidence": {"source_traj_uids": examples[0]["source_leaf_ids"]},
        })
    metrics = write_new_experiences(mem, additions, step=101)
    assert metrics["bank/added"] == 2
    assert len(mem.experiences["task_experiences"]) == initial_t + 1
    assert len(mem.experiences["step_experiences"]) == initial_s + 1
    assert mem.experiences["task_experiences"][-1]["utility"] == 0
    assert mem.experiences["step_experiences"][-1]["created_at_step"] == 101


def test_real_api_s_alias_is_accepted(sampled_tree, monkeypatch):
    for name in ("ALL_PROXY", "all_proxy", "HTTP_PROXY", "http_proxy", "HTTPS_PROXY", "https_proxy"):
        monkeypatch.delenv(name, raising=False)
    import json
    source = "/home/zehao/echo-collab/T069/reflector-v3-results/raw-step80.private.txt"
    if not Path(source).exists():
        pytest.skip("raw real-API sample absent")
    raw = json.loads(Path(source).read_text())
    if not raw or raw[0].get("experience_type") != "S":
        pytest.skip("real sample does not contain a T/S alias")
    updater = ExperienceUpdater(
        api_key="unit-test-only", base_url="http://127.0.0.1:9/v1",
        model="unit-test", max_new_experiences_per_update=5,
    )
    _, examples = sampled_tree
    parsed = updater._parse_experiences_response(json.dumps(raw))
    kept = updater._postprocess_search_experiences(parsed, examples)
    assert len(kept) == 1
    assert kept[0]["experience_type"] == "step"
    assert kept[0]["evidence"]["source_traj_uids"]


def test_api_case_selector_uses_ranked_ten_and_never_modifies_pool(sampled_tree):
    from agent_system.tree.reflector_case_selector_v3 import (
        select_reflector_examples_with_api,
    )
    from agent_system.tree.core import trajectory_reflector_examples
    from types import SimpleNamespace
    import json
    tree, _ = sampled_tree
    pool = trajectory_reflector_examples(tree, k=40, seed=80)
    assert len(pool) >= 25

    class Fake:
        model = "fake"
        token_limit_param = "max_tokens"
        def _completion_with_deadline(self, kwargs):
            return SimpleNamespace(choices=[
                SimpleNamespace(message=SimpleNamespace(
                    content=json.dumps({"selected_indices": [3,4,7,9,11,14,17,19,28,32]})
                ))
            ])

    fake = Fake()
    cases = select_reflector_examples_with_api(fake, pool, tree.bank_snapshot, k=10)
    assert [x["fork_id"] for x in cases] == [
        pool[i - 1]["fork_id"] for i in (3,4,7,9,11,14,17,19,28,32)
    ]
    assert fake.last_selector_status["used_llm"] is True


def test_api_case_selector_falls_back_if_api_unavailable(sampled_tree):
    from agent_system.tree.reflector_case_selector_v3 import (
        select_reflector_examples_with_api,
    )
    from agent_system.tree.core import trajectory_reflector_examples
    tree, _ = sampled_tree
    pool = trajectory_reflector_examples(tree, k=25, seed=80)

    class Fake:
        model = "fake"
        token_limit_param = "max_tokens"
        def _completion_with_deadline(self, kwargs):
            raise ConnectionError("offline")
    fake = Fake()
    cases = select_reflector_examples_with_api(fake, pool, tree.bank_snapshot, k=10)
    assert [e["fork_id"] for e in cases] == [e["fork_id"] for e in pool[:10]]
    assert fake.last_selector_status["used_llm"] is False


def test_real_step5_malformed_json_is_repaired_and_three_entries_upsert(monkeypatch):
    for key in ("ALL_PROXY","all_proxy","HTTP_PROXY","http_proxy","HTTPS_PROXY","https_proxy"):
        monkeypatch.delenv(key, raising=False)
    import json
    from copy import deepcopy
    from agent_system.memory.experience_memory import ExperienceMemory
    from agent_system.tree.training import write_new_experiences
    root = Path("/home/zehao/echo-collab/T069/reflector-v3-results")
    raw_path = root / "selector-reflect-raw-step5.private.txt"
    example_path = root / "selector-selected-step5.private.json"
    bank_path = Path("/home/zehao/echo-collab/T069/tv2/checkpoints/experience_bank_step5.json")
    if not all(p.exists() for p in (raw_path, example_path, bank_path)):
        pytest.skip("real-API saved fixture unavailable")
    writer = ExperienceUpdater(api_key="offline", base_url="http://127.0.0.1:9/v1",
                               model="offline")
    original = writer._parse_experiences_response(raw_path.read_text())
    assert writer.last_parse_succeeded and writer.last_json_repaired
    assert len(original) == 3
    examples = json.loads(example_path.read_text())
    accepted = writer._postprocess_search_experiences(original,examples)
    assert [e["experience_type"] for e in accepted] == ["step","task","step"]
    memory = ExperienceMemory(load_initial_experiences=False)
    memory.experiences = json.loads(bank_path.read_text())
    memory._normalize_all_experience_meta()
    metrics=write_new_experiences(memory, accepted,step=999)
    assert metrics["bank/added"]==3
    assert metrics["reflector/added_task"]==1
    assert metrics["reflector/added_step"]==2
