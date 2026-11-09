"""Trainer hooks for tree evidence, lifecycle transactions, and exact checkpoint state."""

from collections import defaultdict
from copy import deepcopy
from pathlib import Path
import json
import os
import statistics
from agent_system.tree.core import QuestionEvidence, fork_reflector_examples


def atomic_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2))
    os.replace(temporary, path)


def restore_tree_state(trainer):
    if not trainer.config.env.get("tree_experience", {}).get("enable", False):
        return
    if trainer.global_steps == 0:
        trainer.tree_evidence = QuestionEvidence()
        return
    path = (
        Path(trainer.config.trainer.default_local_dir)
        / f"global_step_{trainer.global_steps}"
        / "tree_state.json"
    )
    if not path.exists():
        raise FileNotFoundError("Missing exact tree checkpoint state: " + str(path))
    state = json.loads(path.read_text())
    if state["checkpoint_step"] != trainer.global_steps:
        raise ValueError("Tree checkpoint step identity mismatch")
    trainer.tree_evidence = QuestionEvidence(state["evidence"])
    trainer.envs.retrieval_memory.replace_experiences_keep_cache_incremental(
        state["bank"]
    )


def save_tree_checkpoint(trainer, directory):
    if hasattr(trainer, "tree_evidence"):
        atomic_json(
            Path(directory) / "tree_state.json",
            dict(
                checkpoint_step=trainer.global_steps,
                evidence=trainer.tree_evidence.state,
                bank=trainer.envs.retrieval_memory.experiences,
            ),
        )


def manage_tree_experiences(trainer, tree):
    cfg = trainer.config.env.tree_experience
    step = int(trainer.global_steps)
    memory = trainer.envs.retrieval_memory
    evidence = trainer.tree_evidence
    evidence.record(tree.question_credits())
    evidence.state["contrasts"].extend(fork_reflector_examples(tree))
    metrics = dict(
        tree.metrics,
        **{
            "tree/mode": 1.0,
            "bank/added": 0.0,
            "bank/deleted_negative": 0.0,
            "bank/deleted_capacity": 0.0,
        },
    )
    interval = int(cfg.utility_update_interval)
    if step % interval:
        metrics["bank/window_pending"] = 1.0
        return metrics
    bank = deepcopy(memory.experiences)
    updates, negative = evidence.merge_window(
        bank,
        step,
        beta=float(cfg.utility_ema_beta),
        min_questions=int(cfg.min_delete_questions),
        protect_recent_steps=int(cfg.protect_recent_steps),
    )
    memory.replace_experiences_keep_cache_incremental(bank)
    metrics["bank/utility_updates"] = len(updates)
    metrics["bank/deleted_negative"] = len(negative)
    contrasts = sorted(
        evidence.state["contrasts"], key=lambda x: (-abs(x["delta"]), x["fork_id"])
    )[:10]
    rates = defaultdict(list)
    for leaf in tree.leaves:
        rates[leaf.data_source].append(leaf.reward)
    allowed = any(
        statistics.mean(x)
        < float(trainer.config.env.experience_memory.update_threshold)
        for x in rates.values()
    )
    if contrasts and allowed:
        if not hasattr(trainer, "experience_updater"):
            from agent_system.memory.experience_updater import ExperienceUpdater

            params = trainer.config.env.experience_memory
            trainer.experience_updater = ExperienceUpdater(
                base_url=params.experience_update_base_url,
                model=params.experience_update_model,
                max_new_experiences_per_update=params.max_new_experiences,
                max_completion_tokens=params.max_completion_tokens,
                token_limit_param=params.token_limit_param,
            )
        additions = trainer.experience_updater.analyze_failures(
            contrasts, memory.experiences
        )
        for item in additions:
            item["created_at_step"] = step
            item.setdefault("evidence", {})["last_updated_step"] = step
        metrics["bank/added"] = memory.upsert_experiences(additions)
    evidence.state["contrasts"] = []
    capacity = memory.evict_excess_experiences(
        step,
        max_task_experiences=200,
        max_step_experiences=200,
        protect_recent_steps=int(cfg.protect_recent_steps),
        utility_threshold=None,
    )
    metrics["bank/deleted_capacity"] = len(capacity["removed"])
    metrics["bank/size"] = len(memory)
    utilities = [
        float(x.get("utility", 0.0))
        for pool in ("task_experiences", "step_experiences")
        for x in memory.experiences.get(pool, [])
    ]
    if utilities:
        metrics["bank/utility_mean"] = statistics.mean(utilities)
        metrics["bank/utility_std"] = statistics.pstdev(utilities)
        metrics["bank/utility_max"] = max(utilities)
        metrics["bank/utility_min"] = min(utilities)
        metrics["bank/utility_positive_rate"] = statistics.mean(u > 0 for u in utilities)
    metrics["bank/reflector_allowed"] = float(bool(contrasts and allowed))
    root = Path(trainer.config.trainer.default_local_dir)
    record = dict(
        step=step,
        metrics=metrics,
        utility_updates=updates,
        deleted_negative=negative,
        deleted_capacity=capacity,
        evidence=evidence.state,
        reflector_fork_ids=[x["fork_id"] for x in contrasts],
    )
    atomic_json(root / f"tree_management_step{step}.json", record)
    memory.save_experiences(str(root / f"experience_bank_step{step}.json"))
    trainer._sync_experiences_to_retrieval_server(
        memory, path=str(root / f"experience_bank_step{step}.json")
    )
    return metrics
