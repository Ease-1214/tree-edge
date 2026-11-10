from types import SimpleNamespace
import math

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from verl import DataProto
from verl.trainer.ppo.core_algos import compute_policy_loss
from agent_system.tree.core import QuestionEvidence
from agent_system.tree.dedup import weighted_node_policy_loss
from agent_system.tree.loss import tree_topk_loss
from agent_system.tree.question_batches import node_mixed_plan, prepare_dedup_batch


def example_rows(questions=64):
    rows = []
    for question in range(questions):
        for index in range(16):
            rows.append(
                dict(
                    uid=str(question),
                    node_id=f"{question}:{index % 4}",
                    advantage=1.2 if index % 3 else -0.7,
                    kd_weight=0.5 if index == 0 else 0.0,
                    kd_canonical=index == 0,
                )
            )
    return rows


def test_mixed_update_count_weights_and_question_diversity():
    rows = example_rows()
    lengths = [8 + index % 5 for index in range(len(rows))]
    plan = node_mixed_plan(rows, lengths, seed=7)
    assert len(plan) == math.ceil(len(rows) / 256)
    packed = [item for mini in plan for rank in mini for item in rank]
    assert len({item["node_id"] for item in packed}) == len(packed)
    assert sum(item["w_tok"] for item in packed) == len(rows)
    assert sorted(index for item in packed for index in item["source_indices"]) == list(
        range(len(rows))
    )
    assert node_mixed_plan(rows, lengths, seed=7) == plan
    questions = [
        {rows[item["source_indices"][0]]["uid"] for rank in mini for item in rank}
        for mini in plan
    ]
    assert sum(map(len, questions)) / len(questions) >= 32
    loads = [sum(item["w_tok"] for rank in mini for item in rank) for mini in plan]
    assert max(loads) - min(loads) <= 4


def test_mixed_packing_preserves_canonical_and_masks_padding():
    rows = example_rows()
    count = len(rows)
    canonical = torch.tensor([row["kd_canonical"] for row in rows])
    weights = torch.tensor([row["kd_weight"] for row in rows])
    batch = DataProto.from_dict(
        tensors=dict(
            attention_mask=torch.ones(count, 8),
            kd_weight=weights,
            edge_distill_weight=weights.clone(),
            kd_canonical=canonical,
            is_padding_copy=torch.zeros(count, dtype=torch.bool),
        ),
        non_tensors=dict(uid=np.array([row["uid"] for row in rows], dtype=object)),
    )
    tree = SimpleNamespace(expanded_rows=lambda: rows)
    result = prepare_dedup_batch(batch, tree, batching="node_mixed")
    assert result.batch["w_tok"].sum() == count
    assert result.batch["kd_canonical"].sum() == 64
    assert len(result) % 16 == 0
    assert all(
        rank.batch["dedup_mini"].unique().numel() == 4 for rank in result.chunk(4)
    )
    padding = result.batch["is_padding_copy"]
    assert result.batch["w_tok"][padding].sum() == 0
    assert result.batch["edge_distill_weight"][padding].sum() == 0
    assert not result.batch["kd_canonical"][padding].any()
    with pytest.raises(ValueError):
        prepare_dedup_batch(batch, tree, batching="unknown")


def test_each_mixed_minibatch_loss_and_gradient_match_its_expansion():
    rows = example_rows(32)
    plan = node_mixed_plan(rows, [8] * len(rows), seed=3)
    names = sorted({row["node_id"] for row in rows})
    positions = {name: index for index, name in enumerate(names)}
    logprob = (
        torch.linspace(-1.0, 1.5, len(names) * 3)
        .reshape(len(names), 3)
        .requires_grad_()
    )
    old = torch.zeros_like(logprob)
    for mini in plan:
        items = [item for rank in mini for item in rank]
        selected = torch.tensor([positions[item["node_id"]] for item in items])
        expanded = [index for item in items for index in item["source_indices"]]
        expanded_selected = torch.tensor(
            [positions[rows[index]["node_id"]] for index in expanded]
        )
        advantage = torch.tensor([rows[index]["advantage"] for index in expanded])
        mask = torch.ones(len(expanded), 3)
        full = compute_policy_loss(
            old[expanded_selected],
            logprob[expanded_selected],
            advantage[:, None].expand(-1, 3),
            mask,
            cliprange=0.2,
            clip_ratio_c=3.0,
        )[0]
        collapsed = weighted_node_policy_loss(
            old[selected],
            logprob[selected],
            torch.tensor([item["A_pos"] for item in items]),
            torch.tensor([item["A_neg"] for item in items]),
            torch.ones(len(items), 3),
            torch.tensor([item["w_tok"] for item in items]),
        )
        student = torch.stack([logprob, -logprob], dim=-1)
        teacher = torch.zeros_like(student)
        full_kd = tree_topk_loss(
            student[expanded_selected],
            teacher[expanded_selected],
            mask,
            torch.tensor([rows[index]["kd_weight"] for index in expanded]),
            mask,
        )[0]
        compact_mask = torch.ones(len(items), 3)
        compact_kd = tree_topk_loss(
            student[selected],
            teacher[selected],
            compact_mask,
            torch.tensor([item["kd_weight"] for item in items]),
            compact_mask * torch.tensor([item["w_tok"] for item in items])[:, None],
        )[0]
        torch.testing.assert_close(full_kd, compact_kd, rtol=1e-5, atol=1e-7)
        full = full + full_kd
        collapsed = collapsed + compact_kd
        expected_gradient = torch.autograd.grad(full, logprob, retain_graph=True)[0]
        actual_gradient = torch.autograd.grad(collapsed, logprob, retain_graph=True)[0]
        torch.testing.assert_close(full, collapsed, rtol=1e-5, atol=1e-7)
        torch.testing.assert_close(
            expected_gradient, actual_gradient, rtol=1e-5, atol=1e-7
        )


def evidence_fixture():
    evidence = QuestionEvidence()
    bank = {"task_experiences": [], "step_experiences": []}
    for index in range(5):
        eid = str(index)
        pool = "task_experiences" if index % 2 else "step_experiences"
        bank[pool].append(dict(experience_id=eid, created_at_step=0, utility=0.0))
        evidence.record(
            {(eid, f"q{question}"): -0.5 + index * 0.1 for question in range(20)}
        )
    return evidence, bank


def test_delete_cap_keeps_lowest_three_upper_bounds_across_pools():
    evidence, bank = evidence_fixture()
    _, removed = evidence.merge_window(bank, 30, max_delete_per_window=3)
    assert [item["experience_id"] for item in removed] == ["0", "1", "2"]
    assert {item["experience_id"] for pool in bank.values() for item in pool} == {
        "3",
        "4",
    }


def test_default_delete_cap_is_unlimited():
    evidence, bank = evidence_fixture()
    _, removed = evidence.merge_window(bank, 30)
    assert len(removed) == 5
    assert not any(bank.values())


def test_options_default_to_current_behavior():
    config = OmegaConf.load("verl/trainer/config/ppo_trainer.yaml")
    assert config.env.tree_experience.dedup_batching == "question_grouped"
    assert config.env.tree_experience.max_delete_per_window == 0
