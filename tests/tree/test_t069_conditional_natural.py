"""T069 conditional natural-four rollouts + unified GRPO + successful SFT tests."""
from collections import Counter, defaultdict
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf
from verl import DataProto
from verl.utils.model import compute_position_id_with_mask

from agent_system.tree import runner
from agent_system.tree.core import leaf_advantages
from agent_system.tree.question_batches import collapse_rows, prepare_dedup_batch


class Collector:
    def __init__(self, root, enable_sft=True):
        self.config = OmegaConf.create({
            "env": {
                "max_steps": 2, "history_length": 2,
                "tree_experience": {
                    "task_top_k": 4, "step_top_k": 6,
                    "conditional_natural": {"enable": True, "natural_rollouts": 4},
                    "dedup_shared_prefix": True, "dedup_batching": "node_mixed",
                },
                "seed": 1,
            },
            "actor_rollout_ref": {"actor": {
                "edge_distillation": {"path_tier": 2},
                "success_sft": {"enable": enable_sft, "coef": 0.05},
            }},
            "trainer": {"default_local_dir": str(root / "checkpoints")},
        })
        self.tokenizer = SimpleNamespace(batch_decode=lambda rows, **kwargs:
                                         ["<search>example</search>"] * len(rows))

    def _apply_chat_template_to_text_obs(self, text):
        return text

    def _tokenize_text_prompts(self, texts, device="cpu"):
        ids = torch.tensor([
            [ord(c) % 100 + 1 for c in t[-63:]] +
            [0] * (64 - min(63, len(t))) for t in texts
        ], dtype=torch.long)
        mask = (ids != 0).long()
        return dict(input_ids=ids, attention_mask=mask,
                    position_ids=compute_position_id_with_mask(mask))


class Memory:
    experiences = {
        "task_experiences": [{"experience_id": "T", "utility": 0}],
        "step_experiences": [{"experience_id": "S", "utility": 0}],
    }

    def __init__(self):
        self.task_queries = []
        self.step_queries = []

    def retrieve_task_experiences_batch(self, texts, **kwargs):
        self.task_queries.append(list(texts))
        return [self.payload("T") for _ in texts]

    def retrieve_step_experiences_batch(self, texts, **kwargs):
        self.step_queries.append(list(texts))
        return [self.payload("S") for _ in texts]

    def payload(self, experience_id):
        return {
            "experience_ids": [experience_id], "task_experiences": [],
            "step_experiences": [],
        }

    def format_for_prompt(self, item):
        return "EXPERIENCE:" + ",".join(item["experience_ids"])


class Actor:
    world_size = 4

    def generate_sequences(self, batch):
        response = torch.ones(len(batch), 4, dtype=torch.long)
        ids = torch.cat([batch.batch["input_ids"], response], dim=-1)
        mask = torch.cat([batch.batch["attention_mask"], torch.ones_like(response)], -1)
        return DataProto.from_dict(tensors={
            "responses": response,
            "input_ids": ids, "attention_mask": mask,
            "position_ids": compute_position_id_with_mask(mask),
            "rollout_log_probs": torch.full_like(response, -0.5, dtype=torch.float32),
        })


def case_batch(counts, step=1):
    items = [dict(question=f"q{i}", ground_truth=["answer"], data_source="nq")
             for i in range(len(counts))]
    return DataProto.from_dict(
        tensors={"input_ids": torch.ones(len(items), 2, dtype=torch.long)},
        non_tensors={"env_kwargs": runner.object_array(items)},
        meta_info={"global_step": step},
    )


def run_case(monkeypatch, tmp_path, counts, tree_success=True):
    def execute(config, backend, nodes, actions):
        outputs = []
        for n in nodes:
            q = int(n.state_before.question[1:])
            if n.source == "natural":
                assert n.new_experience_ids == ()
                assert n.actual_prompt_text == n.noexp_prompt_text
                assert not n.state_before.task_memory
                reward = float(n.state_before.natural_sample_id < counts[q])
            else:
                assert counts[q] <= 1, "Tree launched for naturally-solved question"
                reward = float(tree_success and n.branch == "R")
            outputs.append((("observation", reward, True, {"tool_calling": 1}), []))
        return outputs

    monkeypatch.setattr(runner, "execute_decisions", execute)
    collector = Collector(tmp_path)
    memory = Memory()
    manager = SimpleNamespace(
        retrieval_memory=memory, envs=None,
        # Synthetic projection deliberately alternates real current actions.
        # With the T70 gate, same-action terminal reward differences must NOT
        # be counted as an OPSD supervision signal.
        projection_f=lambda texts: (
            [f"<search>query-{i % 2}</search>" for i in range(len(texts))],
            [True] * len(texts),
        ),
    )
    batch = runner.tree_multi_turn_loop(
        collector, case_batch(counts), Actor(), manager
    )
    return collector.last_tree, batch, memory


def test_mixed_trigger_all_cases_and_identity(monkeypatch, tmp_path):
    tree, batch, memory = run_case(monkeypatch, tmp_path, [0, 1, 2, 3, 4])
    assert len([x for x in tree.leaves if x.source == "natural"]) == 20
    assert len([x for x in tree.leaves if x.source == "tree"]) == 4
    assert len(tree.forks) == 2
    assert len(memory.task_queries) == 1 and len(memory.task_queries[0]) == 2
    assert memory.step_queries == []
    for p in range(5):
        assert tree.metrics[f"natural/pass_{p}_questions"] == 1
    assert tree.metrics["tree/trigger_rate"] == pytest.approx(0.4)
    assert tree.metrics["tree/rescue_rate"] == pytest.approx(1)
    assert len(set(x.uid for x in tree.leaves)) == 5
    for q in range(5):
        leaves = [x for x in tree.leaves if x.uid == f"s1:q{q}"]
        assert len({l.leaf_id for l in leaves}) == len(leaves)
    assert sum(x["gate"] for x in tree.audits) == 2
    # GRPO advantages are calculated across NATURAL + TREE leaves, not separately.
    assert all(tree.training_advantages()[x.leaf_id] == 0
               for x in tree.leaves if x.uid == "s1:q4")
    assert any(tree.training_advantages()[x.leaf_id] > 0
               for x in tree.leaves if x.uid == "s1:q0")

    real = ~batch.batch["is_padding_copy"]
    assert batch.meta_info["tree_mode"]
    assert batch.batch["sft_weight"][real].sum() == 12
    assert batch.batch["edge_distill_weight"][~batch.batch["kd_canonical"]].sum() == 0
    assert set(batch.non_tensor_batch["rollout_source"][real.numpy()]) == {"tree", "natural"}


def test_all_four_pass_zero_advantage_but_sft_active(monkeypatch, tmp_path):
    tree, batch, memory = run_case(monkeypatch, tmp_path, [4])
    assert len(tree.forks) == 0 and memory.task_queries == []
    assert len(tree.leaves) == 4
    assert all(v == 0 for v in tree.training_advantages().values())
    assert batch.batch["sft_weight"].sum() == 4
    assert not batch.batch["edge_distill_weight"].any()
    assert tree.metrics["tree/trigger_rate"] == 0


def test_all_zero_tree_fails_no_invented_positive_reward(monkeypatch, tmp_path):
    tree, batch, memory = run_case(monkeypatch, tmp_path, [0], tree_success=False)
    assert len(tree.leaves) == 6
    assert not any(x.reward > 0 for x in tree.leaves)
    assert all(v == 0 for v in tree.training_advantages().values())
    assert batch.batch["sft_weight"].sum() == 0
    assert not batch.batch["edge_distill_weight"].any()
    assert tree.metrics["tree/rescue_rate"] == 0


def test_natural_one_of_four_mixed_advantage_and_sft(monkeypatch, tmp_path):
    tree, batch, _ = run_case(monkeypatch, tmp_path, [1])
    assert len(tree.leaves) == 6
    assert sum(l.reward == 1 for l in tree.leaves) == 2
    assert len(set(round(float(v), 4) for v in tree.training_advantages().values())) == 2
    assert batch.batch["sft_weight"].sum() == 2


def test_shared_node_success_multiplicity_and_ppo_equivalence():
    rows = [
        dict(node_id="n0", uid="q", reward=1., advantage=1.5,
             kd_weight=0., sft_weight=1.),
        dict(node_id="n0", uid="q", reward=1., advantage=1.5,
             kd_weight=0., sft_weight=1.),
        dict(node_id="n0", uid="q", reward=0., advantage=-1.5,
             kd_weight=0., sft_weight=0.),
    ]
    collapsed = collapse_rows(rows, list(range(3)))
    assert len(collapsed) == 1
    c = collapsed[0]
    assert (c["w_tok"], c["sft_weight"], c["A_pos"], c["A_neg"]) == (3, 2., 3., -1.5)
    from agent_system.tree.dedup import weighted_node_policy_loss
    old = torch.tensor([[-1.2, -1.1, -1.0]])
    new = old + torch.tensor([[0.02, 0.01, -0.03]])
    mask = torch.ones_like(old)
    compact = weighted_node_policy_loss(
        old, new, torch.tensor([c["A_pos"]]), torch.tensor([c["A_neg"]]),
        mask, torch.tensor([c["w_tok"]]),
    )
    expanded = weighted_node_policy_loss(
        old.repeat(3, 1), new.repeat(3, 1),
        torch.tensor([1.5, 1.5, 0.0]),
        torch.tensor([0.0, 0.0, -1.5]),
        mask.repeat(3, 1), torch.ones(3),
    )
    torch.testing.assert_close(compact, expanded)
    lp = torch.tensor([[-1.0, -1.5, -2.0]], requires_grad=True)
    compact_sft = -((lp * mask) * c["sft_weight"]).sum() / (mask.sum() * c["sft_weight"])
    expanded_sft = -(lp.repeat(2, 1) * mask.repeat(2, 1)).sum() / (mask.sum() * 2)
    torch.testing.assert_close(compact_sft, expanded_sft)
    compact_sft.backward()
    assert lp.grad is not None and lp.grad.abs().sum() > 0


def test_natural_and_tree_dedup_sft_student_prompt_matches_noexp(monkeypatch, tmp_path):
    tree, batch, _ = run_case(monkeypatch, tmp_path, [0])
    real = (~batch.batch["is_padding_copy"]).nonzero().flatten()
    for i in real.tolist():
        source = batch.non_tensor_batch["rollout_source"][i]
        if batch.batch["sft_weight"][i] > 0:
            node = tree.nodes[batch.non_tensor_batch["node_id"][i]]
            R = node.tensors["responses"].shape[-1]
            expected = Collector(tmp_path)._tokenize_text_prompts([node.noexp_prompt_text], "cpu")
            torch.testing.assert_close(
                batch.batch["sft_student_input_ids"][i, :-R], expected["input_ids"][0]
            )
            assert source in ("tree", "natural")



def test_t70_reflector_strictly_uses_real_tree_forks(monkeypatch, tmp_path):
    from agent_system.tree.core import trajectory_reflector_examples
    from agent_system.tree.reflector_case_selector_v3 import selection_prompt
    tree, _, _ = run_case(monkeypatch, tmp_path, [0, 2])
    # Simulate different current actions in the real left/right intervention.
    # Independent natural actions MUST NOT enter Reflector even when outcomes
    # are different (2/4 natural and 0/4 natural).
    for node in tree.nodes.values():
        if node.source == "tree":
            node.action = ("<search>wrong first query</search>" if node.branch == "L"
                           else "<search>specific correct query</search>")
        else:
            node.action = ("<search>failed query</search>" if node.state_before.natural_sample_id >= 2
                           else "<search>successful query</search>")
    tree.evaluate_forks()
    cases = trajectory_reflector_examples(tree, k=40, seed=1)
    assert len(cases) == 1
    assert cases[0]["fork_id"] in tree.forks
    assert cases[0]["source_leaf_ids"]
    tree_leaves = {leaf.leaf_id for leaf in tree.leaves if leaf.source == "tree"}
    assert set(cases[0]["source_leaf_ids"]) <= tree_leaves
    assert all("natural" not in case["fork_id"] for case in cases)
    assert "CANDIDATE PREVIEWS" in selection_prompt(cases, tree.bank_snapshot)
    assert len(tree.question_credits()) == 1  # Only T intervention has utility.

def test_t70_reflector_has_no_cases_when_no_tree_triggered(monkeypatch, tmp_path):
    from agent_system.tree.core import trajectory_reflector_examples
    tree, _, _ = run_case(monkeypatch, tmp_path, [2, 3, 4])
    assert not tree.forks and not tree.audits
    assert trajectory_reflector_examples(tree, k=40, seed=1) == []
