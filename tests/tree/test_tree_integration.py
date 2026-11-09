from types import SimpleNamespace
from pathlib import Path
import numpy as np
import pytest
import torch
from omegaconf import OmegaConf
from agent_system.tree import runner
from agent_system.tree.core import SearchBranchState
from agent_system.multi_turn_rollout.rollout_loop import TrajectoryCollector
from agent_system.memory.experience_memory import ExperienceMemory
from verl import DataProto
from verl.utils.model import compute_position_id_with_mask


class Tokenizer:
    pad_token_id = 0

    def batch_decode(self, rows, **kwargs):
        return ["<search>query</search>" for _ in rows]


class Collector:
    tokenizer = Tokenizer()

    def __init__(self, root):
        self.config = OmegaConf.create(
            {
                "env": {
                    "max_steps": 4,
                    "history_length": 4,
                    "tree_experience": {"task_top_k": 4, "step_top_k": 6},
                },
                "actor_rollout_ref": {"actor": {"edge_distillation": {"path_tier": 2}}},
                "trainer": {"default_local_dir": str(root / "checkpoints")},
            }
        )

    def _apply_chat_template_to_text_obs(self, text):
        return text

    def _tokenize_text_prompts(self, texts, device):
        ids = torch.tensor(
            [
                [ord(c) % 100 + 1 for c in text[-63:]] + [0] * (64 - min(63, len(text)))
                for text in texts
            ]
        )
        mask = (ids != 0).long()
        return dict(
            input_ids=ids,
            attention_mask=mask,
            position_ids=compute_position_id_with_mask(mask),
        )


class Memory:
    experiences = {
        "task_experiences": [{"experience_id": "T", "utility": 0}],
        "step_experiences": [{"experience_id": "S", "utility": 0}],
    }

    def retrieve_task_experiences_batch(self, texts, **kwargs):
        return [self.payload("T") for _ in texts]

    def retrieve_step_experiences_batch(self, texts, **kwargs):
        return [self.payload("S") for _ in texts]

    def payload(self, eid):
        pool = "task_experiences" if eid == "T" else "step_experiences"
        return {
            "experience_ids": [eid],
            "task_experiences": [],
            "step_experiences": [],
            pool: [{"experience_id": eid, "content": eid}],
        }

    def format_for_prompt(self, m):
        return "EXPERIENCE " + " ".join(m["experience_ids"])


class Actor:
    world_size = 4

    def generate_sequences(self, b):
        responses = torch.ones(len(b), 4, dtype=torch.long)
        ids = torch.cat([b.batch["input_ids"], responses], -1)
        mask = torch.cat([b.batch["attention_mask"], torch.ones_like(responses)], -1)
        return DataProto.from_dict(
            tensors=dict(
                prompts=b.batch["input_ids"],
                responses=responses,
                rollout_log_probs=torch.ones_like(responses, dtype=torch.float32)
                * -0.5,
                input_ids=ids,
                attention_mask=mask,
                position_ids=compute_position_id_with_mask(mask),
            )
        )


def test_depth_bounds_persistence_and_true_state_independence(monkeypatch, tmp_path):
    def execute(config, backend, nodes, actions):
        output = []
        for n in nodes:
            done = n.depth == 4
            reward = float(n.branch == "R") if done else 0.0
            n.state_before.env_history.append({"content": n.node_id})
            output.append((("obs " + n.node_id, reward, done, {}), []))
        return output

    monkeypatch.setattr(runner, "execute_decisions", execute)
    collector = Collector(tmp_path)
    manager = SimpleNamespace(
        retrieval_memory=Memory(),
        envs=None,
        projection_f=lambda responses: (responses, [True] * len(responses)),
    )
    b = DataProto.from_dict(
        tensors={"input_ids": torch.ones(1, 2, dtype=torch.long)},
        non_tensors={
            "env_kwargs": runner.object_array(
                [dict(question="synthetic", ground_truth=["gold"], data_source="nq")]
            )
        },
        meta_info={"global_step": 1},
    )
    out = runner.tree_multi_turn_loop(collector, b, Actor(), manager)
    tree = collector.last_tree
    assert len(tree.nodes) == 30 and len(tree.leaves) == 16
    assert max(x.depth for x in tree.nodes.values()) == 4
    assert all(len(x.node_ids) == 4 for x in tree.leaves)
    for node in tree.nodes.values():
        assert node.path_has_T == (node.node_id.split(":node:")[1][0] == "R")
        assert len(node.state_before.env_history) == 1
        if node.depth > 1 and node.branch == "L":
            assert "EXPERIENCE S" not in node.actual_prompt_text
        assert "EXPERIENCE" not in node.noexp_prompt_text
    assert len(out) == 64
    assert (
        "path_has_T" in out.non_tensor_batch and "kd_triggered" in out.non_tensor_batch
    )
    # A generated prefix is copied bit-for-bit into every descendant leaf row.
    root = "s1:q0:node:L"
    rows = np.flatnonzero(out.non_tensor_batch["node_id"] == root)
    assert len(rows) == 8
    assert all(
        torch.equal(out.batch["input_ids"][rows[0]], out.batch["input_ids"][i])
        for i in rows
    )


def test_prompt_current_step_only_and_no_cross_branch_T():
    state = SearchBranchState(
        "q", "original", [], "nq", task_memory=Memory().payload("T")
    )
    memory = Memory()
    with_s = runner.build_prompt(state, memory, memory.payload("S"))
    assert "T" in with_s and "S" in with_s
    next_prompt = runner.build_prompt(state, memory)
    assert "EXPERIENCE T" in next_prompt and "EXPERIENCE T S" not in next_prompt
    noexp = runner.build_prompt(state, memory, no_experience=True)
    assert "EXPERIENCE" not in noexp
    assert "EXPERIENCE" not in runner.build_prompt(
        SearchBranchState("q", "original", [], "nq"), memory
    )


def test_shared_prefix_copies_generation_logprobs():
    from agent_system.tree.runner import pack_training_rows
    from test_tree_method import example
    from tempfile import TemporaryDirectory

    tree = example((0, 0, 0), (0, 1))
    for node in tree.nodes.values():
        prompt = torch.ones(64, dtype=torch.long)
        response = torch.ones(4, dtype=torch.long)
        node.tensors = dict(
            prompts=prompt,
            responses=response,
            input_ids=torch.cat([prompt, response]),
            attention_mask=torch.ones(68, dtype=torch.long),
            position_ids=torch.arange(68),
            rollout_log_probs=torch.randn(4),
        )
    with TemporaryDirectory() as directory:
        batch = pack_training_rows(
            Collector(Path(directory)), tree, tree.expanded_rows()
        )
    indices = np.flatnonzero(batch.non_tensor_batch["node_id"] == "L")
    assert len(indices) == 3
    assert all(
        torch.equal(
            batch.batch["rollout_log_probs"][indices[0]],
            batch.batch["rollout_log_probs"][i],
        )
        for i in indices
    )
