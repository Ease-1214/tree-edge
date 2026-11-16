"""Comprehensive CPU regression tests for T75 Modules 1-5."""
from pathlib import Path
from types import SimpleNamespace
import json
import math
import numpy as np
import pytest

from agent_system.memory.experience_memory import ExperienceMemory, _TASK_QUERY_HINTS
from agent_system.memory.experience_updater import ExperienceUpdater
from agent_system.memory.experience_type_validation import validate_experience_types
from agent_system.tree.core import (
    DecisionNode,
    ForkRecord,
    Leaf,
    QuestionEvidence,
    SearchBranchState,
    TreeRollout,
    subtree_value,
    trajectory_reflector_examples,
)
from agent_system.tree.edge_fork_adapter import edge_search_examples_from_forks


ROOT = Path(__file__).resolve().parents[1]


def test_module1_v3_dual_pool_bank_and_no_alfworld_pollution():
    bank_path = ROOT / "memory_data" / "search" / "state_aware_experiences_v2.json"
    memory = ExperienceMemory(
        experiences_json_path=str(bank_path),
        retrieval_mode="template",
        load_initial_experiences=True,
    )
    task_exps = memory.experiences["task_experiences"]
    step_exps = memory.experiences["step_experiences"]
    assert len(task_exps) == 11
    assert len(step_exps) == 13
    assert {x["experience_id"] for x in task_exps} == {f"task_{i:06d}" for i in range(1, 12)}
    assert {x["experience_id"] for x in step_exps} == {f"step_{i:06d}" for i in range(1, 14)}

    # Ensure zero ALFWorld strings in search hints or bank
    for key in ("general", "unknown", "all"):
        assert "alfworld" not in _TASK_QUERY_HINTS[key].lower()
    raw_dump = json.dumps(memory.experiences).lower()
    assert "alfworld" not in raw_dump

    # Format check: no duplicate title == when_to_apply, and includes anti-oversearching note
    formatted = memory.format_for_prompt({
        "task_experiences": [task_exps[0]],
        "step_experiences": [step_exps[0]],
    })
    assert "### Task-level experience" in formatted
    assert "### Step-level experience" in formatted
    assert "output `<answer>` directly" in formatted


def test_module2_similarity_threshold_filters_before_ucb_and_skips_fork(monkeypatch):
    memory = ExperienceMemory(
        load_initial_experiences=False,
        retrieval_mode="embedding",
        retrieval_alpha=0.5,
        retrieval_ucb_c=0.5,
        similarity_threshold=0.4,
    )
    pool = "step_experiences"
    # Item 1 has high similarity (0.55 >= 0.4) and high exposure (count=20 -> low UCB bonus).
    # Item 2 has low similarity (0.30 < 0.4) and zero exposure (count=0 -> huge UCB bonus).
    items = [
        {
            "experience_id": "step_000001",
            "task_type": "general",
            "title": "Valid match",
            "principle": "Use snippet",
            "when_to_apply": "When matched",
            "retrieval_obs": "matched",
            "utility": 0.0,
            "retrieval_count": 20,
        },
        {
            "experience_id": "step_000002",
            "task_type": "general",
            "title": "Irrelevant unvisited",
            "principle": "Noise",
            "when_to_apply": "Never",
            "retrieval_obs": "noise",
            "utility": 0.2,
            "retrieval_count": 0,
        },
    ]
    memory.experiences[pool] = items
    memory._normalize_all_experience_meta()

    sims = [0.55, 0.30]
    vectors = np.asarray([[s, math.sqrt(1 - s * s)] for s in sims], dtype=np.float32)
    monkeypatch.setattr(
        memory, "_compute_pool_embeddings", lambda kind: {"embeddings": vectors, "items": items}
    )
    monkeypatch.setattr(
        memory, "_encode_texts", lambda texts, normalize_embeddings=True: np.array([[1.0, 0.0]], dtype=np.float32)
    )

    res = memory.retrieve_step_experiences_batch(["query"], top_k=1, semantic_candidate_k=12)
    # Item 2 (sim=0.30 < 0.4) must be filtered out BEFORE UCB, so Item 1 is returned!
    assert res[0]["experience_ids"] == ["step_000001"]

    # Now if all similarities are below 0.4 (e.g. 0.35 and 0.20), returns empty -> skip fork
    sims_low = [0.35, 0.20]
    vectors_low = np.asarray([[s, math.sqrt(1 - s * s)] for s in sims_low], dtype=np.float32)
    monkeypatch.setattr(
        memory, "_compute_pool_embeddings", lambda kind: {"embeddings": vectors_low, "items": items}
    )
    res_low = memory.retrieve_step_experiences_batch(["query"], top_k=1, semantic_candidate_k=12)
    assert res_low[0]["experience_ids"] == []


def test_module3_unbiased_recursive_subtree_value_on_asymmetric_tree():
    """Verify early-answering branch (1 leaf) is weighted equally (50%) against
    a deep searching sibling branch (4 leaves) at their shared fork."""
    state = SearchBranchState("q1", "Who wrote Book X?", ["Author Y"], "nq")
    tree = TreeRollout()
    for nid, parent, depth, branch in [
        ("L", None, 1, "L"),
        ("R", None, 1, "R"),
        ("RL", "R", 2, "L"),
        ("RR", "R", 2, "R"),
        ("RRL", "RR", 3, "L"),
        ("RRR", "RR", 3, "R"),
    ]:
        tree.nodes[nid] = DecisionNode(
            "q1", nid, parent, f"f_{depth}", depth, branch, state.clone(),
            ("task_000001",) if nid == "R" else (), "actual", "noexp",
            action=f"<search>{nid}</search>",
        )
    tree.forks["root"] = ForkRecord("root", "q1", 1, "L", "R", ("task_000001",), (0.0,))
    # Left fails (1 leaf at L)
    tree.leaves.append(Leaf("q1", "leaf_L", ("L",), 0.0, "nq"))
    # Right child RL answers early at depth 2 and succeeds (1 leaf)
    tree.leaves.append(Leaf("q1", "leaf_RL", ("R", "RL"), 1.0, "nq"))
    # Right child RR searches to depth 3/4 and produces 4 failing leaves
    for i in range(2):
        tree.leaves.append(Leaf("q1", f"leaf_RRL_{i}", ("R", "RR", "RRL"), 0.0, "nq"))
        tree.leaves.append(Leaf("q1", f"leaf_RRR_{i}", ("R", "RR", "RRR"), 0.0, "nq"))

    audits = tree.evaluate_forks()
    assert len(audits) == 1
    audit = audits[0]
    # Flat leaf mean on Right would have been 1/5 = 0.20.
    # Recursive subtree_value gives 0.5 * V(RL=1.0) + 0.5 * V(RR=0.0) = 0.50!
    assert audit["left_mean"] == pytest.approx(0.0)
    assert audit["right_mean"] == pytest.approx(0.5)
    assert audit["delta"] == pytest.approx(0.5)
    assert audit["weight"] == pytest.approx(0.5)


def test_module4_stratified_depth_bucketing_and_strict_harmful_delta():
    tree = TreeRollout()
    tree.bank_snapshot = {
        "task_experiences": [
            {"experience_id": "task_000001", "trigger": "Root trigger", "recommended_action": "Root action"}
        ],
        "step_experiences": [
            {"experience_id": "step_000001", "trigger": "Step trigger", "recommended_action": "Step action"}
        ],
    }
    # Create 6 forks at depth 1, 6 at depth 2, 6 at depth 3 (all delta < 0),
    # plus 2 forks with delta > 0 (which must be excluded!).
    idx = 0
    for depth in (1, 2, 3):
        for j in range(6):
            idx += 1
            uid = f"q_{depth}_{j}"
            st = SearchBranchState(uid, f"Question {depth} {j}?", ["ans"], "nq")
            lid, rid = f"L_{idx}", f"R_{idx}"
            fid = f"fork_{idx}"
            eid = ("task_000001",) if depth == 1 else ("step_000001",)
            tree.nodes[lid] = DecisionNode(
                uid, lid, None, fid, depth, "L", st.clone(), (), "p", "n",
                action=f"<search>good query {idx}</search>",
            )
            tree.nodes[rid] = DecisionNode(
                uid, rid, None, fid, depth, "R", st.clone(), eid, "p", "n",
                action=f"<search>bad query {idx}</search>",
            )
            tree.forks[fid] = ForkRecord(fid, uid, depth, lid, rid, eid, (0.0,))
            # Left succeeds (1.0), Right fails (0.0) -> delta = -1.0 < 0
            tree.leaves.append(Leaf(uid, f"leaf_{lid}", (lid,), 1.0, "nq"))
            tree.leaves.append(Leaf(uid, f"leaf_{rid}", (rid,), 0.0, "nq"))

    # Add a positive-delta fork (Left=0.0, Right=1.0 -> delta = +1.0 > 0)
    st_pos = SearchBranchState("q_pos", "Positive question?", ["ans"], "nq")
    tree.nodes["L_pos"] = DecisionNode(
        "q_pos", "L_pos", None, "fork_pos", 1, "L", st_pos, (), "p", "n",
        action="<search>wrong</search>",
    )
    tree.nodes["R_pos"] = DecisionNode(
        "q_pos", "R_pos", None, "fork_pos", 1, "R", st_pos, ("task_000001",), "p", "n",
        action="<search>right</search>",
    )
    tree.forks["fork_pos"] = ForkRecord("fork_pos", "q_pos", 1, "L_pos", "R_pos", ("task_000001",), (0.0,))
    tree.leaves.append(Leaf("q_pos", "leaf_L_pos", ("L_pos",), 0.0, "nq"))
    tree.leaves.append(Leaf("q_pos", "leaf_R_pos", ("R_pos",), 1.0, "nq"))

    tree.evaluate_forks()
    selected_12 = trajectory_reflector_examples(tree, k=12, seed=42)
    assert len(selected_12) == 12
    # Must never include delta > 0 fork
    assert all(c["fork_id"] != "fork_pos" for c in selected_12)
    assert all(c["right_subtree_rate"] < c["left_subtree_rate"] for c in selected_12)
    # Exact 4-4-4 stratified split across depth==1, depth==2, depth>=3
    depths = [c["fork_depth"] for c in selected_12]
    assert depths.count(1) == 4
    assert depths.count(2) == 4
    assert depths.count(3) == 4
    # Resolved added_experiences must contain actual trigger & recommended_action text
    assert selected_12[0]["added_experiences"][0]["trigger"] in ("Root trigger", "Step trigger")


def test_module5_grpo_advantage_decoupled_from_invalid_penalty():
    state = SearchBranchState("q_fail", "Hard question", ["ans"], "nq")
    tree = TreeRollout(invalid_action_penalty_coef=0.5)
    for i in range(4):
        nid = f"N{i}"
        tree.nodes[nid] = DecisionNode(
            "q_fail", nid, None, "f", 1, "L", state.clone(), (), "p", "n",
            valid=(i < 3),  # N0, N1, N2 valid format; N3 invalid format (truncated)
        )
        tree.leaves.append(Leaf("q_fail", f"L{i}", (nid,), 0.0, "nq"))
    adv = tree.training_advantages()
    # Valid-format wrong leaves get 0.0 (not positive reward!), invalid leaf gets -0.5
    assert adv["L0"] == 0.0
    assert adv["L1"] == 0.0
    assert adv["L2"] == 0.0
    assert adv["L3"] == pytest.approx(-0.5)


def test_sft_weight_excludes_invalid_actions_in_rescued_trajectory():
    state = SearchBranchState("q_rescue", "Question?", ["ans"], "nq")
    tree = TreeRollout(invalid_action_penalty_coef=0.5)
    # Turn 1: invalid format action (valid=False)
    tree.nodes["N1_invalid"] = DecisionNode(
        "q_rescue", "N1_invalid", None, "", 1, "L", state.clone(), (), "p", "n",
        action="<think>truncated...", valid=False, done=False,
    )
    # Turn 2: valid answer that rescues the trajectory (reward=1.0, valid=True)
    tree.nodes["N2_valid"] = DecisionNode(
        "q_rescue", "N2_valid", "N1_invalid", "", 2, "L", state.clone(), (), "p", "n",
        action="<think>ok</think><answer>ans</answer>", valid=True, done=True, terminal_reward=1.0,
    )
    tree.leaves.append(Leaf("q_rescue", "L_rescue", ("N1_invalid", "N2_valid"), 1.0, "nq"))
    rows = {r["node_id"]: r for r in tree.expanded_rows()}
    assert rows["N1_invalid"]["sft_weight"] == 0.0
    assert rows["N2_valid"]["sft_weight"] == 1.0


def test_clean_wandb_metrics_groups_into_six_panels():
    from agent_system.tree.metrics import build_clean_wandb_metrics

    raw = {
        # 1_eval fallbacks and dynamic val_<src>_em
        "val/success_rate": 0.42,
        "val/macro_em": 0.39,
        "val/hotpotqa_success_rate": 0.51,
        "val/2wikimultihopqa_success_rate": 0.33,
        "natural/pass_rate": 0.45,
        "success/pure_noexp": 0.40,  # should prefer natural/pass_rate
        "tree/rescue_rate": 0.25,
        "success/leaf_mean": 0.60,
        # 2_tree_exp
        "tree/trigger_rate": 0.55,
        "gate/fork_count": 12,
        "exp_ab/task_T/delta_mean": 0.08,
        "exp_ab/step_S/delta_mean": 0.04,
        "exp_ab/all/helped_rate": 0.30,
        "exp_ab/all/harmed_rate": 0.05,
        "gate/same_action_rate": 0.15,
        # 3_memory
        "bank/size": 34,
        "bank/utility_mean": 0.12,
        "bank/utility_positive_rate": 0.65,
        "retrieval/exposed_experiences": 28,
        "reflector/submitted": 4,
        "reflector/added_task": 2,
        "reflector/added_step": 1,
        "bank/deleted_negative": 1,
        "bank/deleted_capacity": 0,
        # 4_loss (testing fallback aliases and NaN filtering)
        "actor/pg_loss": -0.02,
        "kd/weighted_loss_active_tokens": 0.14,
        "kd/canonical_rows": 16,
        "sft/loss_token_mean": 0.75,
        "sft/active_rows": 8,
        "actor/grad_norm": 1.2,
        "actor/ppo_kl": 0.004,
        "actor/pg_clipfrac": 0.03,
        "actor/lr": 1e-6,
        # 5_behavior
        "episode/valid_action_ratio": 0.96,
        "tree/search_calls_per_question": 2.4,
        "actor/response_length_expanded": 180.0,
        # 6_perf
        "timing_s/step": 42.5,
        "timing_s/gen": 18.0,
        "timing_s/update_actor": 14.0,
        "timing_s/experience_management": 6.5,
        "perf/max_memory_allocated_gb": 21.3,
        "perf/throughput": 1250.0,
        # Noisy / NaN / Inf keys that must be excluded
        "raw/noisy_debug_metric": 999.0,
        "1_main/legacy_key": 0.1,
        "actor/entropy": float("nan"),
        "perf/edge_teacher_rows": float("inf"),
        "val/strict_em": float("nan"),  # should fall back to val/success_rate
    }

    cleaned = build_clean_wandb_metrics(raw)
    assert cleaned["1_eval/val_strict_em"] == pytest.approx(0.42)
    assert cleaned["1_eval/val_macro_em"] == pytest.approx(0.39)
    assert cleaned["1_eval/val_hotpotqa_em"] == pytest.approx(0.51)
    assert cleaned["1_eval/val_2wikimultihopqa_em"] == pytest.approx(0.33)
    assert cleaned["1_eval/train_natural_pass_rate"] == pytest.approx(0.45)
    assert cleaned["1_eval/train_tree_rescue_rate"] == pytest.approx(0.25)
    assert cleaned["1_eval/train_leaf_mean_reward"] == pytest.approx(0.60)
    assert cleaned["2_tree_exp/task_delta_mean"] == pytest.approx(0.08)
    assert cleaned["3_memory/bank_size"] == pytest.approx(34.0)
    assert cleaned["4_loss/pg_loss"] == pytest.approx(-0.02)
    assert cleaned["4_loss/opsd_loss"] == pytest.approx(0.14)
    assert cleaned["4_loss/opsd_active_rows"] == pytest.approx(16.0)
    assert cleaned["5_behavior/valid_action_ratio"] == pytest.approx(0.96)
    assert cleaned["6_perf/reflector_time_s"] == pytest.approx(6.5)

    allowed_prefixes = ("1_eval/", "2_tree_exp/", "3_memory/", "4_loss/", "5_behavior/", "6_perf/")
    assert all(k.startswith(allowed_prefixes) for k in cleaned)
    assert "raw/noisy_debug_metric" not in cleaned
    assert "1_main/legacy_key" not in cleaned
    assert "val/strict_em" not in cleaned


def test_build_step_retrieval_query_summarizes_queries_and_obs():
    pytest.importorskip("torch")
    from agent_system.tree.runner import build_step_retrieval_query

    state = SearchBranchState("q_step", "Who directed the 1999 film The Mummy?", ["Stephen Sommers"], "nq")
    state.turns = 1
    state.history = [("<think>try</think><search>The Mummy director</search>", "<information>Doc 1</information>")]
    state.last_observation = "<information>The Mummy (1932) directed by Karl Freund.</information>"
    q = build_step_retrieval_query(state, max_steps=4)
    assert "Previous search queries: The Mummy director" in q
    assert "Karl Freund" in q
    assert "<information>" not in q



