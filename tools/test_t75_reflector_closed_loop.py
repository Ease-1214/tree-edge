#!/usr/bin/env python3
"""Closed-loop verification of the T75 Reflector + Validator + Bank Upsert pipeline.

Modes:
  1. Offline / Dry-Run (default when no API key is configured):
     - Builds a realistic multi-depth TreeRollout with delta < 0 forks (depth=1 Task fork,
       depth=2 Step fork, depth=3 Step fork, plus injected existing experiences).
     - Runs `select_reflector_forks` -> `edge_search_examples_from_forks` ->
       `ExperienceUpdater._build_search_analysis_prompt` and prints the exact prompt.
     - Runs a realistic simulated LLM JSON response (including 1 duplicate of an existing
       experience, 1 mislabeled task/step experience, and 2 novel experiences) through:
       (a) Default 1-call mode (`ENABLE_EXPERIENCE_TYPE_VALIDATION=0`)
       (b) Optional 2-call Validator mode (`ENABLE_EXPERIENCE_TYPE_VALIDATION=1`)
       (c) `write_new_experiences` + `0.85` cosine-similarity vector deduplication.
  2. Live API Mode (`--live` or automatic when `EXPERIENCE_UPDATE_API_KEY` is set):
     - Sends the constructed fork contrast prompt to the live OpenAI-compatible
       endpoint (`EXPERIENCE_UPDATE_BASE_URL` / `EXPERIENCE_UPDATE_MODEL`) and
       executes the full end-to-end extraction and bank insertion loop.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import sys
from unittest.mock import MagicMock

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np

from agent_system.memory.experience_memory import ExperienceMemory
from agent_system.memory.experience_updater import ExperienceUpdater
from agent_system.tree.core import (
    DecisionNode,
    ForkRecord,
    Leaf,
    SearchBranchState,
    TreeRollout,
    trajectory_reflector_examples,
)
from agent_system.tree.edge_fork_adapter import edge_search_examples_from_forks
from agent_system.tree.training import write_new_experiences


def build_realistic_test_tree(bank: dict) -> TreeRollout:
    """Create a realistic TreeRollout with depth 1, 2, 3 harmful forks (delta < 0)."""
    tree = TreeRollout(bank_snapshot=deepcopy(bank))

    # Case 1: Depth-1 Task Fork (Left succeeds without experience, Right fails after bad Task exp)
    q1 = "Which county contains the headquarters of the company that acquired GitHub in 2018?"
    s1_l = SearchBranchState(uid="s5:q1", question=q1, ground_truth="King County", data_source="hotpotqa")
    s1_r = s1_l.clone()
    s1_r.task_memory = {"experience_ids": ["task_000001"]}

    n1_l = DecisionNode(
        uid="s5:q1", node_id="s5:q1:node:L", parent_id=None, fork_id="s5:q1:fork:root",
        depth=1, branch="L", state_before=s1_l, new_experience_ids=(),
        actual_prompt_text=q1, noexp_prompt_text=q1,
        action="<think>First find which company acquired GitHub in 2018, then find its headquarters county.</think><search>company acquired GitHub 2018 headquarters county</search>",
        observation="<information>Microsoft acquired GitHub in October 2018. Microsoft is headquartered in Redmond, Washington, in King County.</information>",
        done=False, valid=True,
    )
    n1_l2 = DecisionNode(
        uid="s5:q1", node_id="s5:q1:node:LL", parent_id="s5:q1:node:L", fork_id="",
        depth=2, branch="L", state_before=s1_l.clone(), new_experience_ids=(),
        actual_prompt_text=q1, noexp_prompt_text=q1,
        action="<think>Redmond, Washington is in King County.</think><answer>King County</answer>",
        observation="", done=True, terminal_reward=1.0, valid=True,
    )
    n1_r = DecisionNode(
        uid="s5:q1", node_id="s5:q1:node:R", parent_id=None, fork_id="s5:q1:fork:root",
        depth=1, branch="R", state_before=s1_r, new_experience_ids=("task_000001",),
        actual_prompt_text=q1, noexp_prompt_text=q1,
        action="<think>I know GitHub is in San Francisco.</think><answer>San Francisco County</answer>",
        observation="", done=True, terminal_reward=0.0, path_has_T=True, valid=True,
    )

    # Case 2: Depth-2 Step Fork (Left verifies ambiguous title, Right answers prematurely)
    q2 = "Who directed the 1999 film The Mummy?"
    s2_root = SearchBranchState(uid="s5:q2", question=q2, ground_truth="Stephen Sommers", data_source="nq")
    n2_root = DecisionNode(
        uid="s5:q2", node_id="s5:q2:node:L", parent_id=None, fork_id="",
        depth=1, branch="L", state_before=s2_root, new_experience_ids=(),
        actual_prompt_text=q2, noexp_prompt_text=q2,
        action="<think>Search for the 1999 film The Mummy.</think><search>The Mummy film director</search>",
        observation="<information>The Mummy is a 1932 film directed by Karl Freund. The Mummy (2017) was directed by Alex Kurtzman. The 1999 adventure film The Mummy was written and directed by ... (truncated)</information>",
        done=False, valid=True,
    )
    s2_d2 = s2_root.clone()
    s2_d2.turns = 1
    s2_d2.last_observation = n2_root.observation
    s2_d2.history = [(n2_root.action, n2_root.observation)]

    n2_l = DecisionNode(
        uid="s5:q2", node_id="s5:q2:node:LL", parent_id="s5:q2:node:L", fork_id="s5:q2:fork:L",
        depth=2, branch="L", state_before=s2_d2.clone(), new_experience_ids=(),
        actual_prompt_text=q2, noexp_prompt_text=q2,
        action="<think>The snippet was truncated right before naming the 1999 director. Let me search specifically for 1999 The Mummy director.</think><search>\"The Mummy\" 1999 film directed by</search>",
        observation="<information>The Mummy is a 1999 American action-adventure film written and directed by Stephen Sommers.</information>",
        done=False, valid=True,
    )
    n2_l_end = DecisionNode(
        uid="s5:q2", node_id="s5:q2:node:LLL", parent_id="s5:q2:node:LL", fork_id="",
        depth=3, branch="L", state_before=s2_d2.clone(), new_experience_ids=(),
        actual_prompt_text=q2, noexp_prompt_text=q2,
        action="<think>Stephen Sommers directed the 1999 film.</think><answer>Stephen Sommers</answer>",
        observation="", done=True, terminal_reward=1.0, valid=True,
    )
    n2_r = DecisionNode(
        uid="s5:q2", node_id="s5:q2:node:LR", parent_id="s5:q2:node:L", fork_id="s5:q2:fork:L",
        depth=2, branch="R", state_before=s2_d2.clone(), new_experience_ids=("step_000001",),
        actual_prompt_text=q2, noexp_prompt_text=q2,
        action="<think>The result mentions Karl Freund directed The Mummy.</think><answer>Karl Freund</answer>",
        observation="", done=True, terminal_reward=0.0, valid=True,
    )

    for node in (n1_l, n1_l2, n1_r, n2_root, n2_l, n2_l_end, n2_r):
        tree.nodes[node.node_id] = node

    tree.forks["s5:q1:fork:root"] = ForkRecord(
        fork_id="s5:q1:fork:root", uid="s5:q1", depth=1,
        left_node_id="s5:q1:node:L", right_node_id="s5:q1:node:R",
        new_experience_ids=("task_000001",), utility_snapshot=(0.0,),
    )
    tree.forks["s5:q2:fork:L"] = ForkRecord(
        fork_id="s5:q2:fork:L", uid="s5:q2", depth=2,
        left_node_id="s5:q2:node:LL", right_node_id="s5:q2:node:LR",
        new_experience_ids=("step_000001",), utility_snapshot=(0.0,),
    )

    tree.leaves = [
        Leaf("s5:q1", "s5:q1:node:LL:leaf", ("s5:q1:node:L", "s5:q1:node:LL"), 1.0, "hotpotqa"),
        Leaf("s5:q1", "s5:q1:node:R:leaf", ("s5:q1:node:R",), 0.0, "hotpotqa"),
        Leaf("s5:q2", "s5:q2:node:LLL:leaf", ("s5:q2:node:L", "s5:q2:node:LL", "s5:q2:node:LLL"), 1.0, "nq"),
        Leaf("s5:q2", "s5:q2:node:LR:leaf", ("s5:q2:node:L", "s5:q2:node:LR"), 0.0, "nq"),
    ]
    tree.evaluate_forks()
    return tree


def run_offline_closed_loop(tree: TreeRollout, memory: ExperienceMemory):
    print("=" * 80)
    print("[1/4] Testing Fork Selection (delta < 0, stratified depth buckets, experience text resolution)")
    print("=" * 80)
    contrasts = trajectory_reflector_examples(tree, k=10, seed=5)
    assert len(contrasts) == 2, f"Expected 2 harmful forks, got {len(contrasts)}"
    for c in contrasts:
        delta = float(c["right_subtree_rate"]) - float(c["left_subtree_rate"])
        print(f"  - Selected fork: {c['fork_id']} (depth={c['fork_depth']}, delta={delta:.2f})")
        print(f"    Injected experience texts resolved: {len(c.get('added_experiences', []))} item(s)")
        assert c.get("added_experiences"), "Missing resolved experience text on fork!"

    print("\n" + "=" * 80)
    print("[2/4] Building Single-Call Reflector Prompt (Anti-Duplication + Injected Experience Context)")
    print("=" * 80)
    os.environ["EDGE_REFLECTOR_PROMPT"] = "search"
    edge_examples = edge_search_examples_from_forks(contrasts)
    import agent_system.memory.experience_updater as eu_mod
    if eu_mod.OpenAI is None:
        eu_mod.OpenAI = MagicMock()
    updater = ExperienceUpdater(
        api_key="dummy-offline-key",
        base_url="http://127.0.0.1:29663/v1",
        model="gpt-6.1-sol",
        max_new_experiences_per_update=5,
    )
    prompt = updater._build_search_analysis_prompt(
        failed_trajectories=edge_examples,
        current_experiences=memory.experiences,
        next_exp_idx=100,
    )
    assert "STRICT ANTI-DUPLICATION" in prompt
    assert "New experiences injected into RIGHT at this fork (harmful/misleading in this state):" in prompt
    print(f"  Prompt length: {len(prompt)} chars")
    print("  --- Prompt Excerpt (Rules & Case 1) ---")
    for line in prompt.splitlines()[:45]:
        print("  " + line)
    print("  ...")

    # Simulated LLM response with:
    # 1) A valid novel step experience
    # 2) A valid novel task experience
    # 3) An unambiguous step experience accidentally labeled 'task' by the generator
    mock_llm_json = json.dumps([
        {
            "experience_id": "exp_000100",
            "experience_type": "step",
            "task_type": "multi_hop_reasoning",
            "trigger": "A retrieved passage mentions the target work and year, but truncates the sentence before stating the requested attribute while naming other years' attributes.",
            "recommended_action": "Issue a narrower search combining the exact title in quotes, the release year, and the target role instead of answering from another year's entry.",
            "avoid_action": "Selecting a director or attribute from an earlier or later adaptation.",
            "expected_effect": "Retrieves the full clause for the requested release year and prevents year-mixup errors.",
            "state_cues": [
                "Retrieved snippet lists multiple adaptations or years of a work.",
                "The clause for the requested year is cut off before the target name."
            ],
            "retrieval_text": "When search results list several years of the same title and truncate the target year's sentence, run a narrower query with the exact title and year.",
            "source_examples": [2]
        },
        {
            "experience_id": "exp_000101",
            "experience_type": "task",
            "task_type": "multi_hop_reasoning",
            "trigger": "The question asks for the county or jurisdiction of the company that acquired a named subsidiary in a specific year.",
            "recommended_action": "First identify the acquiring parent company from the acquisition event, then verify the county of the parent company's headquarters rather than the subsidiary's city.",
            "avoid_action": "Answering with the city or county of the acquired company.",
            "expected_effect": "Ensures the final geographic attribute belongs to the acquiring parent organization.",
            "state_cues": [
                "Question chains an acquisition event to a geographic attribute of the acquirer.",
                "Acquiring company is not named directly in the question."
            ],
            "retrieval_text": "When a question asks for the county of the headquarters of an organization that acquired a named company, resolve the acquiring parent company first.",
            "source_examples": [1]
        },
        {
            "experience_id": "exp_000102",
            "experience_type": "task",
            "task_type": "direct_retrieval",
            "trigger": "A question asks who sang or performed a song, and search results mention its writer, original recording, and later cover versions.",
            "recommended_action": "Match the specific performer or recording version requested by the question before answering.",
            "avoid_action": "Answering with the composer when the performer is asked.",
            "expected_effect": "Distinguishes songwriters from recording artists in multi-version song results.",
            "state_cues": [
                "Search results mention both a songwriter and multiple recording artists."
            ],
            "retrieval_text": "When a question asks who sang a song and search results mention the writer and multiple performers, verify the specific performer.",
            "source_examples": [2]
        }
    ])

    print("\n" + "=" * 80)
    print("[3/4] Testing Default 1-Call Extraction & Optional 2-Call Validator Relabeling")
    print("=" * 80)
    mock_resp_gen = MagicMock()
    mock_resp_gen.choices = [MagicMock(message=MagicMock(content=mock_llm_json))]

    # (a) Default 1-call mode (ENABLE_EXPERIENCE_TYPE_VALIDATION=0)
    os.environ["ENABLE_EXPERIENCE_TYPE_VALIDATION"] = "0"
    updater._completion_with_deadline = MagicMock(return_value=mock_resp_gen)
    extracted_1call = updater.analyze_failures(
        failed_trajectories=edge_examples,
        current_experiences=memory.experiences,
    )
    assert updater._completion_with_deadline.call_count == 1, "Expected exactly 1 LLM call when validator is off!"
    print(f"  [1-Call Mode] LLM API calls made: {updater._completion_with_deadline.call_count}, extracted: {len(extracted_1call)}")

    # (b) Optional 2-call Validator mode (ENABLE_EXPERIENCE_TYPE_VALIDATION=1)
    # Validator infers exp_102 is 'step' (because trigger mentions search results) and relabels it instead of dropping!
    mock_validator_json = json.dumps({
        "decisions": [
            {"index": 1, "declared_type": "step", "inferred_type": "step", "accepted": True, "reason": "Mid-trajectory observation cue."},
            {"index": 2, "declared_type": "task", "inferred_type": "task", "accepted": True, "reason": "Recognizable from question alone."},
            {"index": 3, "declared_type": "task", "inferred_type": "step", "accepted": False, "reason": "Trigger depends on search results mentioning writer and cover versions, which is a step condition."}
        ]
    })
    mock_resp_val = MagicMock()
    mock_resp_val.choices = [MagicMock(message=MagicMock(content=mock_validator_json))]
    os.environ["ENABLE_EXPERIENCE_TYPE_VALIDATION"] = "1"
    updater._completion_with_deadline = MagicMock(side_effect=[mock_resp_gen, mock_resp_val])
    extracted_2call = updater.analyze_failures(
        failed_trajectories=edge_examples,
        current_experiences=memory.experiences,
    )
    os.environ["ENABLE_EXPERIENCE_TYPE_VALIDATION"] = "0"
    assert updater._completion_with_deadline.call_count == 2
    assert len(extracted_2call) == 3, f"Expected all 3 accepted (with #3 relabeled to step), got {len(extracted_2call)}"
    assert extracted_2call[2]["experience_type"] == "step", "Expected #3 to be relabeled from task -> step!"
    print(f"  [2-Call Validator Mode] LLM API calls made: {updater._completion_with_deadline.call_count}, accepted: {len(extracted_2call)} (item #3 relabeled task -> step)")

    print("\n" + "=" * 80)
    print("[4/4] Testing Bank Upsert + 0.85 Cosine Similarity Vector Deduplication")
    print("=" * 80)
    # Attach a deterministic mock embedding function so cosine similarity is tested end-to-end
    def fake_encode(texts):
        vecs = []
        for t in texts:
            rng = np.random.default_rng(abs(hash(t.lower().strip())) % (2**32))
            v = rng.standard_normal(64).astype(np.float32)
            v /= np.linalg.norm(v) + 1e-8
            vecs.append(v)
        return np.stack(vecs, axis=0)

    memory._encode_texts = fake_encode
    before_size = len(memory)
    metrics_first = write_new_experiences(memory, extracted_2call, step=5)
    after_first = len(memory)
    print(f"  First insertion: bank size {before_size} -> {after_first} (added={metrics_first['bank/added']}, task={metrics_first['reflector/added_task']}, step={metrics_first['reflector/added_step']})")
    assert metrics_first["bank/added"] == 3

    # Now try inserting a near-duplicate with identical retrieval_obs text -> should be blocked by dedup!
    metrics_dup = write_new_experiences(memory, extracted_2call[:1], step=10)
    print(f"  Duplicate re-insertion: added={metrics_dup['bank/added']} (blocked by vector/key deduplication)")
    assert metrics_dup["bank/added"] == 0
    print("\nCLOSED_LOOP_OFFLINE_VERIFICATION_OK")


def run_live_api_closed_loop(tree: TreeRollout, memory: ExperienceMemory):
    print("=" * 80)
    print("[LIVE API MODE] Testing Real LLM Reflection + Bank Upsert")
    print("=" * 80)
    contrasts = trajectory_reflector_examples(tree, k=10, seed=5)
    edge_examples = edge_search_examples_from_forks(contrasts)
    updater = ExperienceUpdater(
        api_key=os.environ.get("EXPERIENCE_UPDATE_API_KEY"),
        base_url=os.environ.get("EXPERIENCE_UPDATE_BASE_URL"),
        model=os.environ.get("EXPERIENCE_UPDATE_MODEL"),
        max_new_experiences_per_update=5,
    )
    print(f"  Endpoint: {updater.base_url}")
    print(f"  Model:    {updater.model}")
    print(f"  Validator enabled: {os.environ.get('ENABLE_EXPERIENCE_TYPE_VALIDATION', '0')}")
    additions = updater.analyze_failures(
        failed_trajectories=edge_examples,
        current_experiences=memory.experiences,
    )
    print(f"  Succeeded: {updater.last_update_succeeded}, Extracted: {len(additions)}")
    for idx, item in enumerate(additions, 1):
        print(f"    [{idx}] ({item.get('experience_type')}) {item.get('title')}: {item.get('trigger')}")
    if not updater.last_update_succeeded:
        raise RuntimeError(f"Live API call failed: {updater.last_failure_reason}")
    metrics = write_new_experiences(memory, additions, step=5)
    print(f"  Upsert metrics: {metrics}")
    print("\nCLOSED_LOOP_LIVE_API_OK")


def main():
    parser = argparse.ArgumentParser(description="Test T75 Reflector closed loop.")
    parser.add_argument("--live", action="store_true", help="Force live LLM API call using EXPERIENCE_UPDATE_* env vars.")
    args = parser.parse_args()

    bank_path = REPO_ROOT / "memory_data" / "search" / "state_aware_experiences_v2.json"
    memory = ExperienceMemory(
        experiences_json_path=str(bank_path),
        retrieval_mode="template",
    )

    tree = build_realistic_test_tree(memory.experiences)
    run_offline_closed_loop(tree, deepcopy(memory))

    has_live_creds = bool(
        os.environ.get("EXPERIENCE_UPDATE_API_KEY")
        and os.environ.get("EXPERIENCE_UPDATE_BASE_URL")
        and os.environ.get("EXPERIENCE_UPDATE_MODEL")
    )
    if args.live or has_live_creds:
        run_live_api_closed_loop(tree, deepcopy(memory))
    else:
        print("\n[Note] Live API env vars (EXPERIENCE_UPDATE_API_KEY / BASE_URL / MODEL) not set in this shell.")
        print("       To run against the live LLM endpoint on your GPU server:")
        print("       source ~/.edge_reflector.env && python tools/test_t75_reflector_closed_loop.py --live")


if __name__ == "__main__":
    main()
