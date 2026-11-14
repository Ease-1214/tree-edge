"""T70: pre-dedup node packing must preserve every actor tensor exactly."""
import numpy as np
import pytest
import torch
from agent_system.tree import runner
from agent_system.tree.question_batches import (
    prepare_dedup_batch, unique_training_representatives,
    real_node_loads,
)
from test_t069_conditional_natural import Collector, run_case
from test_tree_method import example


def assert_identical_batches(left, right):
    assert set(left.batch.keys()) == set(right.batch.keys())
    assert len(left) == len(right)
    for key in left.batch.keys():
        torch.testing.assert_close(left.batch[key], right.batch[key], rtol=0, atol=0, msg=key)
    assert set(left.non_tensor_batch.keys()) == set(right.non_tensor_batch.keys())
    for key in ("node_id", "uid", "leaf_id", "row_id", "rollout_source", "kd_noexp_prompt_ids"):
        assert [repr(x) for x in left.non_tensor_batch[key]] == [
            repr(x) for x in right.non_tensor_batch[key]
        ], key
    assert left.meta_info == right.meta_info


@pytest.mark.parametrize("success_counts", [[0, 1, 2, 3, 4], [0], [1], [4], [2, 3, 4]])
@pytest.mark.parametrize("sft", [True, False])
def test_t70_actual_runner_prepack_equals_legacy_full_pack(
    monkeypatch, tmp_path, success_counts, sft
):
    tree, fast_batch, _ = run_case(monkeypatch, tmp_path, success_counts)
    rows = tree.expanded_rows()
    collector = Collector(tmp_path, enable_sft=sft)
    legacy_pack = runner.pack_training_rows(
        collector, tree, rows, success_sft=sft,
    )
    legacy = prepare_dedup_batch(
        legacy_pack, tree, world_size=4, batching="node_mixed", seed=2,
    )
    reps = unique_training_representatives(rows)
    compact_pack = runner.pack_training_rows(
        collector, tree, reps, success_sft=sft,
        successful_nodes={r["node_id"] for r in rows if r["reward"] == 1},
    )
    compact = prepare_dedup_batch(
        compact_pack, tree, world_size=4, batching="node_mixed", seed=2,
        expanded_rows=rows,
    )
    assert len(compact_pack) == len({r["node_id"] for r in rows})
    assert len(compact_pack) <= len(legacy_pack)
    assert_identical_batches(legacy, compact)
    if sft:
        assert_identical_batches(fast_batch, compact)
    metrics = real_node_loads(compact, world_size=4)
    assert metrics["dedup/real_rows"] + metrics["dedup/padding_rows"] == len(compact)
    assert metrics["dedup/rank_unique_forward_tokens_imbalance"] >= 1


def test_prepack_shared_prefix_success_count_and_kd_are_exact(tmp_path):
    tree = example(l_rewards=(0, 0, 0), r_rewards=(0, 1), utility=0)
    for idx, node in enumerate(tree.nodes.values()):
        prompt=torch.full((64,), idx+1, dtype=torch.long)
        response=torch.ones(4, dtype=torch.long)
        node.tensors={
            "input_ids":torch.cat([prompt,response]),
            "attention_mask":torch.ones(68,dtype=torch.long),
            "position_ids":torch.arange(68),
            "responses":response,
            "rollout_log_probs":torch.full((4,),-0.5,dtype=torch.float32),
        }
    tree.bank_snapshot={"task_experiences":[],"step_experiences":[]}
    rows=tree.expanded_rows()
    collector=Collector(tmp_path)
    full=runner.pack_training_rows(collector,tree,rows,success_sft=True)
    legacy=prepare_dedup_batch(full,tree,world_size=2,batching="node_mixed",seed=3)
    reps=unique_training_representatives(rows)
    assert len(reps)==2 and len(rows)==5
    compact=prepare_dedup_batch(
        runner.pack_training_rows(
            collector,tree,reps,success_sft=True,
            successful_nodes={r["node_id"] for r in rows if r["reward"]==1},
        ),
        tree,world_size=2,batching="node_mixed",seed=3,expanded_rows=rows
    )
    assert_identical_batches(legacy,compact)
    active=~compact.batch["is_padding_copy"]
    assert compact.batch["w_tok"][active].sum()==5
    assert compact.batch["sft_weight"][active].sum()==1
    assert compact.batch["kd_canonical"][active].sum()==1
    assert abs(float(compact.batch["edge_distill_weight"][active].sum()) - (1/3)) < 1e-7


def test_natural_exact_duplicate_metrics_do_not_change_trajectory_ids(monkeypatch,tmp_path):
    from agent_system.tree.runner import natural_exact_duplicate_metrics
    tree,_,_=run_case(monkeypatch,tmp_path,[4])
    ids=[leaf.leaf_id for leaf in tree.leaves if leaf.source=="natural"]
    assert len(set(ids))==4
    before=natural_exact_duplicate_metrics(tree)
    # The mock rollout returns the exact same output tokens and model prompt
    # four times; that is only a measurement, not an inference cache.
    assert before["natural/exact_duplicate_episode_rate"]==0.75
    assert before["natural/exact_duplicate_node_rate"]==0.75
    target=tree.nodes[next(iter(sorted(n for n in tree.nodes if ":natural:3:" in n)))]
    target.tensors["responses"][0] += 1
    after=natural_exact_duplicate_metrics(tree)
    assert after["natural/exact_duplicate_episode_rate"]==0.5
    assert len([leaf for leaf in tree.leaves if leaf.source=="natural"])==4


def test_rank_compute_proxy_counts_extra_sft_and_kd(monkeypatch,tmp_path):
    tree,batch,_=run_case(monkeypatch,tmp_path,[0,1,2,4])
    metrics=real_node_loads(batch,world_size=4,edge_enabled=True,success_sft=True)
    assert metrics["dedup/rank_estimated_forward_tokens_max"] >= metrics["dedup/rank_unique_forward_tokens_max"]
    assert metrics["dedup/estimated_extra_forward_tokens"] >= 0
    assert metrics["dedup/real_rows"] == int((~batch.batch["is_padding_copy"]).sum())
    assert metrics["dedup/rank_estimated_forward_tokens_imbalance"] >= 1
    without=real_node_loads(batch,world_size=4,edge_enabled=False,success_sft=False)
    assert without["dedup/estimated_extra_forward_tokens"]==0
    assert without["dedup/rank_estimated_forward_tokens_max"]==without["dedup/rank_unique_forward_tokens_max"]
