"""Regression tests: preserved Fork filters, random sample, two-stage output gates."""
import json
from pathlib import Path
from types import SimpleNamespace
import pytest

from agent_system.memory.experience_updater import ExperienceUpdater
from agent_system.tree.reflection_two_stage import analyze_forks_two_stage, STAGE2


def _samples():
    return [
        {"fork_id":"s:test:1","task":"When did a country host a tournament?",
         "mode":"trajectory_pair","reflector_version":"v3",
         "success_turns":[],"failed_turns":[],"source_leaf_ids":["a","b"],
         "fork_depth":1,"left_subtree_rate":1.0,"right_subtree_rate":0.0},
        {"fork_id":"s:test:2","task":"Which state hosts a museum?",
         "mode":"trajectory_pair","reflector_version":"v3",
         "success_turns":[],"failed_turns":[],"source_leaf_ids":["c","d"],
         "fork_depth":2,"left_subtree_rate":0.9,"right_subtree_rate":0.0},
    ]


def _updater(monkeypatch, reflection_statuses=("valid","insufficient"), second_fork_id="s:test:1"):
    # The client is a dummy; every completion is mocked, no API request.
    for name in ("ALL_PROXY","all_proxy","HTTP_PROXY","http_proxy","HTTPS_PROXY","https_proxy"):
        monkeypatch.delenv(name,raising=False)
    updater=ExperienceUpdater(
        api_key="test-only",base_url="http://127.0.0.1:9/v1",
        model="test",max_new_experiences_per_update=5,
        max_completion_tokens=6144,token_limit_param="max_completion_tokens")
    called=[]
    def fake(kwargs,total_seconds=300):
        prompt=kwargs["messages"][0]["content"]
        called.append(prompt)
        if "STAGE_ONE_REFLECTIONS:" in prompt:
            items=[{
                "experience_type":"step",
                "trigger":"A result supplies an event year but not the hosting relation",
                "recommended_action":"Search the event and verify the hosting relation",
                "avoid_action":"Do not use a victory year as the hosting year",
                "state_cues":["event and year"],
                "retrieval_text":"Verify an event's hosting relation to the country.",
                "source_fork_ids":[second_fork_id],
            }]
        else:
            assert "CASES:" in prompt
            items=[{"fork_id":src["fork_id"],"status":status,
                    "decision_state":"", "observed_difference":"",
                    "success_mechanism":"", "failure_pattern":"",
                    "reusable_insight":"test" if status=="valid" else None,
                    "evidence":[],"uncertainty":""}
                   for src,status in zip(_samples(), reflection_statuses)]
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(items)))])
    monkeypatch.setattr(updater,"_completion_with_deadline",fake)
    return updater,called


def test_valid_first_stage_produces_provenanced_experience(monkeypatch):
    updater,calls=_updater(monkeypatch)
    result=analyze_forks_two_stage(updater,_samples(),{"task_experiences":[],"step_experiences":[]})
    assert updater.last_update_succeeded
    assert updater.last_two_stage_counts == {"reflections":2,"valid":1,"extracted":1}
    assert len(result)==1
    assert result[0]["experience_type"]=="step"
    assert result[0]["evidence"]["source_traj_uids"] == ["a","b"]
    assert len(calls)==2


def test_stage_two_rejects_citation_to_filtered_fork(monkeypatch):
    updater,_=_updater(monkeypatch,second_fork_id="s:test:2")
    result=analyze_forks_two_stage(updater,_samples(),{"task_experiences":[],"step_experiences":[]})
    assert result == []
    assert not updater.last_update_succeeded
    assert updater.last_failure_reason=="two_stage_ValueError"


def test_zero_valid_skips_stage_two(monkeypatch):
    updater,calls=_updater(monkeypatch,reflection_statuses=("insufficient","insufficient"))
    result=analyze_forks_two_stage(updater,_samples(),{"task_experiences":[],"step_experiences":[]})
    assert result==[]
    assert updater.last_update_succeeded
    assert len(calls)==1


def test_stage_two_prompt_is_capped_at_five():
    assert "FIVE IS A HARD MAXIMUM" in STAGE2
    assert "ZERO to FIVE is allowed" in STAGE2


def test_stage_two_sees_full_old_bank_action_index(monkeypatch):
    updater, calls = _updater(monkeypatch)
    bank = {
        "task_experiences": [
            {
                "experience_id": "task_000008",
                "when_to_apply": "When a question contains two relations.",
                "principle": "Resolve the intermediate entity before the final relation.",
                "retrieval_obs": "multihop bridge",
            }
        ],
        "step_experiences": [
            {
                "experience_id": "exp_000011",
                "trigger": "An intermediate entity is resolved.",
                "recommended_action": "Search the missing final relation.",
                "avoid_action": "Do not repeat the first-hop query.",
            }
        ],
    }
    result = analyze_forks_two_stage(updater, _samples(), bank)
    assert updater.last_update_succeeded and len(result) == 1
    assert "EXISTING_EXPERIENCE_BANK:" not in calls[0]  # Stage one unchanged.
    prompt = calls[-1]
    assert "NOVELTY AGAINST THE EXISTING BANK" in prompt
    encoded = prompt.split("EXISTING_EXPERIENCE_BANK:\n", 1)[1].split(
        "\n\nSTAGE_ONE_REFLECTIONS:\n", 1
    )[0]
    index = json.loads(encoded)
    assert len(index) == 2
    assert index[0]["id"] == "task_000008"
    assert index[0]["action"] == bank["task_experiences"][0]["principle"]
    assert index[1]["id"] == "exp_000011"
    assert index[1]["avoid"] == "Do not repeat the first-hop query."


def test_bank_aware_stage_two_can_generate_zero_with_valid_reflections(monkeypatch):
    updater, calls = _updater(monkeypatch)
    original = updater._completion_with_deadline

    def no_novel_experiences(kwargs, total_seconds=300):
        if "STAGE_ONE_REFLECTIONS:" in kwargs["messages"][0]["content"]:
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content="[]"))]
            )
        return original(kwargs, total_seconds=total_seconds)

    monkeypatch.setattr(updater, "_completion_with_deadline", no_novel_experiences)
    result = analyze_forks_two_stage(
        updater, _samples(), {"task_experiences": [], "step_experiences": []}
    )
    assert result == []
    assert updater.last_update_succeeded
    assert updater.last_two_stage_counts == {
        "reflections": 2, "valid": 1, "extracted": 0
    }


def test_random_forks_preserve_existing_eligibility():
    import torch
    from agent_system.tree.core import trajectory_reflector_examples
    p=Path("/home/zehao/echo-collab/T070/conditional-natural4-sft/v2-opsd-active-005/tree-private/tree-step42.pt")
    if not p.exists():
        pytest.skip("real T70 tree snapshot unavailable")
    tree=torch.load(p,map_location="cpu",weights_only=False)
    a=trajectory_reflector_examples(tree,k=10,seed=42,sampling="random")
    b=trajectory_reflector_examples(tree,k=10,seed=42,sampling="random")
    c=trajectory_reflector_examples(tree,k=10,seed=43,sampling="random")
    assert len(a)==10
    assert [x["fork_id"] for x in a]==[x["fork_id"] for x in b]
    assert [x["fork_id"] for x in a]!=[x["fork_id"] for x in c]
    assert len({x["task"].casefold().strip() for x in a})==10
    assert all(x["left_subtree_rate"]!=x["right_subtree_rate"] for x in a)
