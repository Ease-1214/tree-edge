"""Offline regression for T73 Task/Step classification; no network or trainer."""
import json
from types import SimpleNamespace

import pytest

from agent_system.memory.experience_type_validation import (
    CLASSIFICATION_GUIDANCE,
    VALIDATION_INSTRUCTION,
    validate_experience_types,
)
from agent_system.memory.experience_updater import ExperienceUpdater


def experience(kind, trigger, *, cues=(), action="Check the relevant evidence."):
    return {
        "experience_id": "exp_000025",
        "experience_type": kind,
        "task_type": "direct_retrieval",
        "trigger": trigger,
        "recommended_action": action,
        "avoid_action": "Avoid mixing unrelated entities.",
        "expected_effect": "Correct answer.",
        "state_cues": list(cues),
        "retrieval_text": trigger,
        "source_examples": [1],
    }


MISMATCHED = experience(
    "task",
    "Search results list an original artist and several cover performers.",
    cues=["Results distinguish writer from original performer"],
)
CORRECT_TASK = experience(
    "task",
    "The question asks for the population of a region located using a landmark.",
    cues=["Which county contains a landmark?"],
    action="Find county first, then use search results to verify population.",
)
CORRECT_STEP = experience(
    "step",
    "The search result resolves the intermediate entity but not its later role.",
    cues=["An entity has already been identified"],
)
AMBIGUOUS = experience(
    "task",
    "The question asks for a person and there may be conflicting snippets.",
)


def mock_result(content):
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
    )


def classify(accepted):
    def completion(request):
        payload = json.loads(request["messages"][0]["content"].split("EXPERIENCES:\n", 1)[1])
        assert "fork_depth" not in request["messages"][0]["content"]
        assert len(payload) == len(accepted)
        return mock_result(json.dumps([
            {"index": i, "inferred_type": t, "reason": "Independently audited trigger"}
            for i, t in enumerate(accepted, 1)
        ]))
    return completion


def test_batch_semantic_validation_accepts_and_rejects_without_rewriting():
    items = [MISMATCHED, CORRECT_TASK, CORRECT_STEP, AMBIGUOUS]
    kept, decisions = validate_experience_types(
        items, classify(["step", "task", "step", "ambiguous"]), model="fake"
    )
    assert kept == [CORRECT_TASK, CORRECT_STEP]
    assert kept[0] is CORRECT_TASK
    assert [x["accepted"] for x in decisions] == [False, True, True, False]
    assert [x["inferred_type"] for x in decisions] == [
        "step", "task", "step", "ambiguous"
    ]
    assert MISMATCHED["experience_type"] == "task"  # never silently relabel


@pytest.mark.parametrize("bad_output", [
    "[]",
    '[{"index": 2,"inferred_type":"task","reason":"fine"}]',
    '[{"index":1,"inferred_type":"unknown","reason":"fine"}]',
    '[{"index":1,"inferred_type":"task"}]',
    "not JSON",
])
def test_malformed_or_partial_validator_response_fails_closed(bad_output):
    with pytest.raises((ValueError, json.JSONDecodeError)):
        validate_experience_types(
            [CORRECT_TASK], lambda _request: mock_result(bad_output), model="fake"
        )


def make_updater(responses):
    updater = ExperienceUpdater(
        api_key="offline", base_url="http://127.0.0.1:1/v1", model="offline"
    )
    updater._prompt_domain = lambda *_: "search"
    updater._build_search_analysis_prompt = lambda *_: "OFFLINE_GENERATION_PROMPT"
    updater._next_exp_index = lambda *_: 25
    updater._parse_experiences_response = lambda raw: (
        setattr(updater, "last_parse_succeeded", True) or json.loads(raw)
    )
    requests = []

    def completion(request):
        requests.append(request)
        reply = responses.pop(0)
        if isinstance(reply, BaseException):
            raise reply
        return mock_result(json.dumps(reply))
    updater._completion_with_deadline = completion
    return updater, requests


FORK = [{
    "fork_id": "fork-1", "fork_depth": 2, "task": "Who is the original artist?",
    "source_leaf_ids": ["s15:q1:node:LL:leaf", "s15:q1:node:LR:leaf"],
    "failed_turns": [], "success_turns": [],
}]


def test_updater_calls_one_validator_and_skips_misclassified_item():
    generated = [dict(MISMATCHED), dict(CORRECT_TASK)]
    result = [
        {"index": 1, "inferred_type": "step", "reason": "Requires actual results"},
        {"index": 2, "inferred_type": "task", "reason": "Only question needed"},
    ]
    updater, calls = make_updater([generated, result])
    kept = updater.analyze_failures(FORK, {"task_experiences": [], "step_experiences": []})
    assert len(calls) == 2
    assert "before a search" in calls[0]["messages"][0]["content"]
    assert "EXPERIENCES:" in calls[1]["messages"][0]["content"]
    assert len(kept) == 1
    assert kept[0]["experience_type"] == "task"
    assert kept[0]["experience_id"] == "exp_000025"
    assert kept[0]["evidence"]["source_traj_uids"] == FORK[0]["source_leaf_ids"]
    assert updater.last_update_succeeded
    assert updater.last_classification_stats == {
        "submitted": 2, "accepted": 1, "rejected": 1,
        "ambiguous": 0, "failed": 0, "invalid_type_dropped": 0,
    }


def test_updater_validator_failure_does_not_let_any_experience_through():
    updater, calls = make_updater([[dict(CORRECT_TASK)], TimeoutError("offline")])
    kept = updater.analyze_failures(FORK, {"task_experiences": [], "step_experiences": []})
    assert kept == []
    assert len(calls) == 2
    assert not updater.last_update_succeeded
    assert updater.last_failure_reason == "classification_validation_failed"
    assert updater.last_classification_stats["failed"] == 1


def test_missing_type_rejected_instead_of_inferred_from_fork_depth():
    invalid = dict(CORRECT_TASK)
    invalid.pop("experience_type")
    updater, calls = make_updater([[invalid]])
    kept = updater.analyze_failures(FORK, {"task_experiences": [], "step_experiences": []})
    assert kept == []
    assert len(calls) == 1  # invalid schema has no validator inputs
    assert updater.last_classification_stats["invalid_type_dropped"] == 1


def test_prompt_definitions_are_consistent_and_not_depth_based():
    assert "original question ALONE" in CLASSIFICATION_GUIDANCE
    assert "TRIGGER and STATE_CUES" in VALIDATION_INSTRUCTION
    assert "Fork depth" in CLASSIFICATION_GUIDANCE
