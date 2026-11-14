import copy
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from omegaconf import OmegaConf
from agent_system.tree.core import *
from agent_system.tree.loss import tree_topk_loss


def example(
    l_rewards=(0, 0), r_rewards=(0, 1), utility=0.0, same=False, ancestor=False
):
    state = SearchBranchState("q", "question", ["gold"], "nq")
    tree = TreeRollout()
    for side in "LR":
        tree.nodes[side] = DecisionNode(
            "q",
            side,
            None,
            "fork",
            1,
            side,
            state.clone(),
            ("e",) if side == "R" else (),
            "actual",
            "noexp",
            action="<search>" + ("same" if same else side) + "</search>",
            path_has_T=ancestor,
        )
    tree.forks["fork"] = ForkRecord("fork", "q", 1, "L", "R", ("e",), (utility,))
    for side, rewards in [("L", l_rewards), ("R", r_rewards)]:
        for i, reward in enumerate(rewards):
            tree.leaves.append(Leaf("q", side + str(i), (side,), reward, "nq"))
    tree.evaluate_forks()
    return tree


def test_independent_state():
    a = SearchBranchState(
        "q",
        "question",
        [],
        "nq",
        history=[("a", "o")],
        task_memory={"experience_ids": ["T"]},
    )
    b = a.clone()
    b.history.append(("b", "new"))
    b.task_memory["experience_ids"].append("S")
    b.done = True
    assert (
        len(a.history) == 1 and not a.done and a.task_memory["experience_ids"] == ["T"]
    )


@pytest.mark.parametrize(
    "left,right,u,same,expected",
    [
        ((0, 0), (0, 1), 0, False, True),
        ((1, 0), (1, 1), 0, False, True),
        ((0, 0), (1, 1), -1, False, True),
        ((0,), (1,), -0.001, False, True),
        ((0,), (1,), 0, True, False),
        ((0,), (0,), 0, False, False),
    ],
)
def test_gate(left, right, u, same, expected):
    t = example(left, right, u, same)
    assert t.audits[0]["gate"] == expected
    assert t.audits[0]["utility_mean"] == u
    if same:
        assert not t.question_credits()


def test_utility_records_valid_zero_gain_forks_without_changing_gate():
    # Success/failure homogeneity must not bias the mean A/B gain upwards
    # or downwards when no terminal result changes.
    for left, right in [((0, 0), (0, 0)), ((1, 1), (1, 1)),
                        ((0, 1), (0, 1))]:
        tree = example(left, right)
        assert list(tree.question_credits().values()) == [0.0]
        assert not tree.audits[0]["gate"]
    # Same action is still excluded; this proposal does not change that policy.
    assert example((0,), (0,), same=True).question_credits() == {}
    # Actual gains and counterfactual losses are unchanged.
    assert list(example((0,), (1,)).question_credits().values()) == [1.0]
    assert list(example((1,), (0,)).question_credits().values()) == [-1.0]


@pytest.mark.parametrize(
    "rewards,weight", [((1,), 1.0), ((1, 1), 1.0), ((1, 1, 1, 0), 0.75)]
)
def test_t71_opsd_raw_gain_without_shrinkage(rewards, weight):
    assert example((0,), rewards).audits[0]["weight"] == weight


@pytest.mark.parametrize(
    "left,right,utility,same,ancestor,expected_gate,expected_weight",
    [
        # Positive difference despite L having some correct terminal leaves.
        ((0, 1, 0, 0), (1, 1, 1, 0), -0.5, False, False, True, 0.5),
        # Exactly equal success rates: no advantage to distill.
        ((1, 0), (0, 1), 0.8, False, False, False, 0.0),
        # Historical Utility does NOT veto a genuine positive contrast.
        ((0,), (1,), -0.9, False, False, True, 1.0),
        # Same current normalized action: apparent reward difference is noise.
        ((0,), (1,), 0.5, True, False, False, 0.0),
        # Positive gain within a history that already contains T: no 0.5 penalty.
        ((0,), (1,), 0.5, False, True, True, 1.0),
        # Reverse contrast is not a positive KD teacher.
        ((1, 1), (1, 0), 0.5, False, False, False, 0.0),
        # Unequal subtree sizes do not further shrink the observed mean gain.
        ((0, 0, 1), (0, 1), 0.0, False, False, True, 1/6),
    ],
)
def test_t71_opsd_subtree_gain_gate_without_shrinkage(
    left, right, utility, same, ancestor, expected_gate, expected_weight
):
    t = example(left, right, utility=utility, same=same, ancestor=ancestor)
    audit = t.audits[0]
    assert audit["gate"] == expected_gate
    assert audit["weight"] == pytest.approx(expected_weight)
    assert audit["delta"] == pytest.approx(
        sum(right)/len(right) - sum(left)/len(left)
    )
    if expected_gate:
        rows = t.expanded_rows()
        assert sum(row["kd_canonical"] for row in rows) == 1
        assert sum(row["kd_weight"] for row in rows) == pytest.approx(expected_weight)
    else:
        assert sum(row["kd_canonical"] for row in t.expanded_rows()) == 0


def test_leaf_statistics_unique_and_groups():
    leaves = [
        Leaf("q", "a", ("n",), 0, "nq"),
        Leaf("q", "b", ("n", "m"), 1, "nq"),
        Leaf("other", "c", ("c",), 1, "nq"),
    ]
    expected = leaf_advantages(leaves)
    assert expected == leaf_advantages(leaves + [leaves[1]])
    assert expected["c"] == 0 and expected["a"] == -expected["b"]
    assert all(
        x == 0
        for x in leaf_advantages([leaves[1], Leaf("q", "d", (), 1, "nq")]).values()
    )
    row_mean = (0 + 1 + 1) / 3
    assert row_mean != 0.5


def test_canonical_and_order():
    t = example((0, 0, 0), (0, 1))
    rows = t.expanded_rows()
    assert sum(x["kd_canonical"] for x in rows) == 1
    assert len({x["uid"] for x in rows}) == 1
    shuffled = list(reversed(t.leaves))
    t.leaves = shuffled
    assert rows == t.expanded_rows()
    assert len(t.nodes) == 2 and len(rows) == 5


def test_empty_tree_no_forks():
    t = TreeRollout(
        nodes={"L": example().nodes["L"]}, leaves=[Leaf("q", "leaf", ("L",), 0, "nq")]
    )
    t.evaluate_forks()
    assert t.question_credits() == {} and t.expanded_rows()[0]["advantage"] == 0


def test_action_normalization():
    assert normalize_action("thinking <search> Foo </search>") == normalize_action(
        "<search>foo</search>"
    )
    assert normalize_action("<answer>foo</answer>") != normalize_action(
        "<search>foo</search>"
    )


def test_distinct_question_evidence_and_recovery():
    e = QuestionEvidence()
    e.record({("e", "q"): -1})
    e.record({("e", "q"): -1})
    assert e.statistics("e")["n"] == 1
    bank = {
        "task_experiences": [
            {"experience_id": "e", "utility": -0.5, "created_at_step": 0}
        ]
    }
    updates = e.merge_window(bank, 5)
    assert updates[0]["utility"] < -0.5
    assert updates[0]["group_count"] == 1
    assert bank["task_experiences"][0]["evidence"]["consecutive_negative_windows"] == 1
    e.record({("e", "other"): 1})
    e.merge_window(bank, 10)
    assert bank["task_experiences"][0]["utility"] > updates[0]["utility"]
    assert bank["task_experiences"][0]["evidence"]["consecutive_negative_windows"] == 0
    json.dumps(e.statistics("missing"), allow_nan=False)


def test_negative_window_continuity_and_unobserved_reset():
    e = QuestionEvidence()
    bank = {"task_experiences": [
        {"experience_id": "e", "utility": 0.0, "created_at_step": 0}
    ]}
    e.record({("e", "q5"): -1})
    e.merge_window(bank, 5)
    assert bank["task_experiences"][0]["evidence"]["consecutive_negative_windows"] == 1
    e.merge_window(bank, 10)  # no evidence, so the streak cannot continue
    assert bank["task_experiences"][0]["evidence"]["consecutive_negative_windows"] == 0
    e.record({("e", "q15"): -1})
    e.merge_window(bank, 15)
    assert bank["task_experiences"][0]["evidence"]["consecutive_negative_windows"] == 1
    e.record({("e", "q20"): 0})
    e.merge_window(bank, 20)
    assert bank["task_experiences"][0]["evidence"]["consecutive_negative_windows"] == 0


def test_cumulative_negative_is_diagnostic_not_immediate_deletion():
    e = QuestionEvidence()
    e.record({("e", str(i)): -1 for i in range(50)})
    bank = {"task_experiences": [
        {"experience_id": "e", "utility": -0.5, "created_at_step": 0}
    ]}
    e.merge_window(bank, 100)
    assert len(bank["task_experiences"]) == 1
    assert bank["task_experiences"][0]["evidence"]["tree_question_statistics"]["n"] == 50
    assert bank["task_experiences"][0]["evidence"]["consecutive_negative_windows"] == 1


def test_full_training_tree_enabled():
    cfg=OmegaConf.create({"env":{"tree_experience":{"enable":True}}})
    assert all(tree_enabled(cfg, step) for step in (1,100,101,200))
    cfg.env.tree_experience.enable=False
    assert not tree_enabled(cfg,1)


def test_reflector_both_signs_and_provenance():
    for left, right, sign in [((0,), (1,), "helped"), ((1,), (0,), "harmed")]:
        t = example(left, right)
        t.bank_snapshot = {
            "task_experiences": [{"experience_id": "e", "content": "strategy"}]
        }
        x = fork_reflector_examples(t)[0]
        assert x["effect"] == sign and x["source_leaf_ids"] == ["L0", "R0"]
        assert x["added_experiences"][0]["experience_id"] == "e"


def test_kd_teacher_detached_padding_and_denominator():
    torch.manual_seed(0)
    s = torch.randn(2, 5, 4, requires_grad=True)
    t = torch.randn(2, 5, 4, requires_grad=True)
    mask = torch.ones(2, 5)
    weights = torch.tensor([1.0, 0.0])
    grpo = torch.ones(2, 50)
    loss, kl, _ = tree_topk_loss(s, t, mask, weights, grpo)
    # Only the five KD-active tokens are included in the denominator.
    assert torch.allclose(loss, kl[0].mean())
    assert not torch.allclose(loss, kl[0].sum() / grpo.sum())
    loss.backward()
    assert t.grad is None and s.grad[0].abs().sum() > 0 and s.grad[1].abs().sum() == 0
    zero, _, _ = tree_topk_loss(s, t, mask, weights * 0, grpo * 0)
    assert torch.isfinite(zero) and zero == 0


def test_kd_extra_student_grad_same_backward():
    a = torch.nn.Linear(3, 4, bias=False)
    grpo = a(torch.ones(1, 3))
    noexp = a(torch.zeros(1, 3) + 2)
    teacher = a(torch.ones(1, 3) * 3).detach()
    kd, _, _ = tree_topk_loss(
        noexp[:, None],
        teacher[:, None],
        torch.ones(1, 1),
        torch.ones(1),
        torch.ones(1, 5),
    )
    (grpo.square().mean() + kd).backward()
    assert a.weight.grad.abs().sum() > 0


def test_padding_copy_no_kd_and_preserves_ids():
    from agent_system.multi_turn_rollout.utils import adjust_batch
    from verl import DataProto

    cfg = OmegaConf.create(
        {
            "trainer": {"n_gpus_per_node": 4, "nnodes": 1},
            "actor_rollout_ref": {
                "rollout": {"log_prob_micro_batch_size_per_gpu": 4},
                "ref": {"log_prob_micro_batch_size_per_gpu": 4},
                "actor": {"use_kl_loss": False, "ppo_micro_batch_size_per_gpu": 2},
            },
            "algorithm": {"use_kl_in_reward": False},
        }
    )
    d = DataProto.from_dict(
        tensors={
            "input_ids": torch.ones(1, 3, dtype=torch.long),
            "is_padding_copy": torch.zeros(1, dtype=torch.bool),
            "kd_canonical": torch.ones(1, dtype=torch.bool),
            "kd_weight": torch.ones(1),
            "edge_distill_weight": torch.ones(1),
        },
        non_tensors={"node_id": np.array(["L"], dtype=object)},
    )
    b = adjust_batch(cfg, d)
    assert len(b) == 16 and b.batch["kd_canonical"].sum() == 1
    assert (
        b.batch["edge_distill_weight"].sum() == 1
        and b.batch["is_padding_copy"].sum() == 15
    )
    assert set(b.non_tensor_batch["node_id"]) == {"L"}


def test_trace_student_teacher_and_shuffled_rows(tmp_path):
    from agent_system.tree.runner import pack_training_rows, audit_training_tokens
    from test_tree_integration import Collector

    t = example((0, 0, 0), (0, 1), ancestor=True)
    for index, node in enumerate(t.nodes.values()):
        prompt = torch.full((64,), 10 + index, dtype=torch.long)
        response = torch.tensor([1, 2, 0, 0])
        mask = torch.cat([torch.ones(64, dtype=torch.long), torch.tensor([1, 1, 0, 0])])
        node.tensors = dict(
            input_ids=torch.cat([prompt, response]),
            responses=response,
            attention_mask=mask,
            position_ids=torch.arange(68),
            prompts=prompt,
        )
        node.actual_prompt_ids = prompt.tolist()
        node.response_ids = response.tolist()
    b = pack_training_rows(Collector(tmp_path), t, t.expanded_rows())
    assert b.non_tensor_batch["is_action_valid"].dtype == np.dtype(bool)
    assert hasattr(b[0].non_tensor_batch["is_action_valid"], "astype")
    b.batch["response_mask"] = b.batch["attention_mask"][:, -4:]
    b = b.select_idxs(np.array([4, 2, 0, 3, 1]))
    result = audit_training_tokens(b, t, 1, tmp_path / "trace.json")
    assert result["trace/passed"] == 1 and result["trace/kd_sampled_tokens"] == 2
    selected = (b.batch["edge_distill_weight"] > 0).nonzero()[0, 0]
    b.batch["edge_teacher_input_ids"][selected, 0] += 1
    with pytest.raises(AssertionError):
        audit_training_tokens(b, t, 1, tmp_path / "bad.json")


def test_tree_checkpoint_roundtrip_and_identity(tmp_path):
    from agent_system.tree.training import save_tree_checkpoint, restore_tree_state

    class Memory:
        experiences = {
            "task_experiences": [{"experience_id": "e", "utility": 0.2}],
            "step_experiences": [],
        }

        def replace_experiences_keep_cache_incremental(self, bank):
            self.experiences = bank

        def reserve_experience_ids(self, historical_ids):
            self.reserved_historical_ids = set(historical_ids)

    config = OmegaConf.create(
        {
            "env": {"tree_experience": {"enable": True}},
            "trainer": {"default_local_dir": str(tmp_path)},
        }
    )
    memory = Memory()
    trainer = SimpleNamespace(
        config=config,
        global_steps=2,
        envs=SimpleNamespace(retrieval_memory=memory),
        tree_evidence=QuestionEvidence(),
    )
    trainer.tree_evidence.record({("e", "same-question"): -0.5})
    save_tree_checkpoint(trainer, tmp_path / "global_step_2")
    trainer.tree_evidence = QuestionEvidence()
    restore_tree_state(trainer)
    assert trainer.tree_evidence.statistics("e")["n"] == 1
    path = tmp_path / "global_step_2/tree_state.json"
    state = json.loads(path.read_text())
    state["checkpoint_step"] = 1
    path.write_text(json.dumps(state))
    with pytest.raises(ValueError):
        restore_tree_state(trainer)
