"""Regression tests: only Fork case selection differs; extraction is one EDGE call."""
import json
from types import SimpleNamespace

from agent_system.memory.experience_updater import ExperienceUpdater
from agent_system.tree.edge_fork_adapter import edge_search_examples_from_forks


def _fork():
    return {
        "fork_id": "f-1", "task": "Which author wrote the work?",
        "fork_depth": 2, "fork_observation": "First retrieved passage",
        "common_search_history": [{"query": "author work"}],
        "left_action": "<search>work title</search>",
        "right_action": "<search>work author</search>",
        "left_experience_ids": [], "right_experience_ids": ["step_000001"],
        "left_subtree_rate": 0.0, "right_subtree_rate": 1.0,
        "left_subtree_leaves": 2, "right_subtree_leaves": 2,
        "source_leaf_ids": ["failed-1", "success-1"],
        "success_turns": [
            {"depth": 2, "action": "<search>work author</search>",
             "observation": "Found author in passage"},
        ],
        "failed_turns": [
            {"depth": 2, "action": "<search>work title</search>",
             "observation": "Unrelated title"},
        ],
    }


def test_fork_adapter_preserves_both_paths_and_outcomes():
    original = _fork()
    examples = edge_search_examples_from_forks([original])
    assert len(examples) == 1
    item = examples[0]
    assert "mode" not in item
    assert item["fork_id"] == original["fork_id"]
    assert item["success_turns"] == original["success_turns"]
    assert item["failed_turns"] == original["failed_turns"]
    assert item["paired_rollout"]["with_experience"]["score"] == 1.0
    assert item["paired_rollout"]["without_experience"]["score"] == 0.0
    assert item["source_leaf_ids"] == ["failed-1", "success-1"]


def test_extraction_calls_edge_single_stage_with_existing_bank(monkeypatch):
    monkeypatch.setenv("EDGE_REFLECTOR_PROMPT", "search")
    updater = object.__new__(ExperienceUpdater)
    updater.model = "fake"
    updater.max_completion_tokens = 2048
    updater.max_new_experiences_per_update = 3
    updater.token_limit_param = "max_tokens"
    updater._audit_request_count = 0
    updater.update_history = []
    existing = {
        "metadata": {"env": "search"},
        "step_experiences": [{
            "experience_id": "step_000001",
            "trigger": "Check which relation is requested",
            "recommended_action": "Search the exact relation",
            "task_type": "direct_retrieval",
        }],
        "task_experiences": [],
    }
    output = [{
        "experience_id": "exp_999999", "experience_type": "step",
        "task_type": "direct_retrieval",
        "trigger": "The retrieved passage names a work but not its author",
        "recommended_action": "Search the author relation",
        "avoid_action": "Guessing a title",
        "expected_effect": "Find evidence", "state_cues": ["work name"],
        "retrieval_text": "When the work is known, search for its author.",
        "source_examples": [1],
    }]
    prompts = []
    replies = [
        output,
        [{"index": 1, "inferred_type": "step", "reason": "Requires retrieved passage"}],
    ]

    def completion(args):
        prompts.append(args["messages"][0]["content"])
        reply = replies[min(len(prompts) - 1, len(replies) - 1)]
        return SimpleNamespace(choices=[SimpleNamespace(
            message=SimpleNamespace(content=json.dumps(reply))
        )])

    updater._completion_with_deadline = completion
    # Default: single-call mode (ENABLE_EXPERIENCE_TYPE_VALIDATION=0)
    monkeypatch.setenv("ENABLE_EXPERIENCE_TYPE_VALIDATION", "0")
    new_single = updater.analyze_failures(
        edge_search_examples_from_forks([_fork()]), existing
    )
    assert len(prompts) == 1
    assert len(new_single) == 1

    # Optional 2-call validator mode (ENABLE_EXPERIENCE_TYPE_VALIDATION=1)
    prompts.clear()
    monkeypatch.setenv("ENABLE_EXPERIENCE_TYPE_VALIDATION", "1")
    new = updater.analyze_failures(
        edge_search_examples_from_forks([_fork()]), existing
    )
    assert len(prompts) == 2
    prompt = prompts[0]
    assert "EXISTING EXPERIENCES:" in prompt
    assert "step_000001" in prompt
    assert "Successful trajectory:" in prompt
    assert "Failed trajectory:" in prompt
    assert "Generate 0-3 experiences" in prompt
    assert "NOVELTY AGAINST THE EXISTING BANK" not in prompt
    assert "STAGE_ONE_REFLECTIONS" not in prompt
    assert len(new) == 1
    assert new[0]["experience_type"] == "step"
    assert new[0]["evidence"]["source_traj_uids"] == ["failed-1", "success-1"]
    assert updater.last_update_succeeded


def test_edge_fork_skips_extra_answer_novelty_filter():
    # EDGE relies on the reflector prompt and bank fingerprint for admission;
    # no custom answer-substring gate may silently discard selected output.
    updater = object.__new__(ExperienceUpdater)
    fork = _fork()
    fork["left_action"] = "<answer>relation-specific answer</answer>"
    item = {
        "experience_type": "step", "task_type": "direct_retrieval",
        "trigger": "When evidence supports a relation-specific answer",
        "recommended_action": "Verify the evidence before answering",
        "avoid_action": "Guessing without checking",
        "retrieval_text": "Verify a relation-specific answer against the evidence",
        "state_cues": ["relation"], "source_examples": [1],
    }
    result = updater._postprocess_search_experiences(
        [item], edge_search_examples_from_forks([fork])
    )
    assert len(result) == 1
    assert result[0]["evidence"]["source_traj_uids"] == ["failed-1", "success-1"]
    assert updater.last_answer_leak_dropped == 0
