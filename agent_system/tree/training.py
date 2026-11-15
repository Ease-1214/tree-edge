"""Trainer hooks for tree evidence, lifecycle transactions, and exact checkpoint state."""

from collections import defaultdict
from copy import deepcopy
from pathlib import Path
import json
import os
import statistics
from agent_system.tree.core import (
    QuestionEvidence,
    trajectory_reflector_examples,
)


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
    # Legacy checkpoints may not persist ID high-water marks. Reserve every
    # historical experience ID before accepting any newly generated entries.
    evidence_state = trainer.tree_evidence.state
    historical_ids = set(evidence_state.get("observations", {}))
    historical_ids.update(evidence_state.get("pending", {}))
    trainer.envs.retrieval_memory.reserve_experience_ids(historical_ids)


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


def tree_exposure_counts(tree):
    """Exposure = one injection of an experience into a RIGHT node (one fork)."""
    counts = defaultdict(int)
    for fork in tree.forks.values():
        for eid in fork.new_experience_ids:
            counts[str(eid)] += 1
    return dict(counts)


def apply_exposure_counts(experiences, counts, step):
    """Add exposures to retrieval_count / last_retrieval_step in place.

    The EDGE path increments retrieval_count inside its utility updates; the tree path
    replaced those updates and never counted, so UCB saw n=0 for every experience
    (identical exploration bonus) -- T066 step-18 finding.
    """
    touched = 0
    for pool in ("task_experiences", "step_experiences"):
        for item in experiences.get(pool, []):
            n = counts.get(str(item.get("experience_id")))
            if n:
                item["retrieval_count"] = int(item.get("retrieval_count", 0) or 0) + int(n)
                item["last_retrieval_step"] = int(step)
                touched += 1
    return touched


def manage_tree_experiences(trainer, tree):
    cfg = trainer.config.env.tree_experience
    step = int(trainer.global_steps)
    memory = trainer.envs.retrieval_memory
    evidence = trainer.tree_evidence
    # Count exposures every step (before any window copy of the bank) so the UCB
    # bonus used by the next step's retrieval reflects real usage.
    exposure_counts = tree_exposure_counts(tree)
    exposed = apply_exposure_counts(memory.experiences, exposure_counts, step)
    evidence.record(tree.fork_credits())
    evidence.state["contrasts"] = []
    metrics = dict(
        tree.metrics,
        **{
            "tree/mode": 1.0,
            "retrieval/exposures": float(sum(exposure_counts.values())),
            "retrieval/exposed_experiences": float(exposed),
            "bank/added": 0.0,
            "bank/deleted_negative": 0.0,
            "bank/deleted_capacity": 0.0,
            "reflector/failed": 0,
            "reflector/submitted": 0,
            "reflector/added_task": 0,
            "reflector/added_step": 0,
            "reflector/skip_reason": 3,
            **{f"reflector/classification_{stat}": 0.0 for stat in (
                "submitted", "accepted", "rejected", "ambiguous",
                "failed", "invalid_type_dropped",
            )},
        },
    )
    interval = int(cfg.utility_update_interval)
    if step % interval:
        metrics["bank/window_pending"] = 1.0
        return metrics
    bank = deepcopy(memory.experiences)
    updates = evidence.merge_window(
        bank,
        step,
        beta=float(cfg.utility_ema_beta),
        management_interval_steps=interval,
    )
    memory.replace_experiences_keep_cache_incremental(bank)
    metrics["bank/utility_updates"] = len(updates)
    metrics["bank/utility_fork_observations"] = sum(
        int(update.get("fork_count", update["group_count"])) for update in updates
    )
    metrics["bank/utility_question_groups"] = sum(
        int(update["group_count"]) for update in updates
    )
    # T71: retain EXACTLY the same Fork eligibility filters; sample randomly
    # from ALL eligible, distinct-question Forks using this Step as the seed.
    # No LLM preselection and no ranking-based top-40 truncation.
    contrast_pool = trajectory_reflector_examples(
        tree, k=100000, seed=step, zero_success_quota=0, sampling="random"
    )
    metrics["reflector/tree_fork_candidates"] = len(contrast_pool)
    contrasts = contrast_pool[:10]
    metrics["reflector/skip_reason"] = 0 if contrasts else 1
    if contrasts:
        metrics["reflector/submitted"] = len(contrasts)
        if not hasattr(trainer, "experience_updater"):
            from agent_system.memory.experience_updater import ExperienceUpdater

            params = trainer.config.env.experience_memory
            trainer.experience_updater = ExperienceUpdater(
                base_url=params.experience_update_base_url,
                model=params.experience_update_model,
                max_new_experiences_per_update=int(params.get("max_new_experiences", 5)),
                max_completion_tokens=int(params.max_completion_tokens),
                token_limit_param=params.token_limit_param,
            )
        from agent_system.tree.edge_fork_adapter import edge_search_examples_from_forks
        # Fork eligibility and selection above produce stratified depth-bucketed
        # harmful forks (delta < 0). Extraction uses a single-call prompt.
        edge_examples = edge_search_examples_from_forks(contrasts)
        metrics["reflector/selector_llm_used"] = 0.0
        additions = trainer.experience_updater.analyze_failures(
            failed_trajectories=edge_examples,
            current_experiences=memory.experiences,
        )
        metrics["reflector/extracted"] = len(additions)
        metrics["reflector/failed"] = int(not trainer.experience_updater.last_update_succeeded)
        audit = getattr(trainer.experience_updater, "last_classification_stats", {})
        for stat in ("submitted", "accepted", "rejected", "ambiguous",
                     "failed", "invalid_type_dropped"):
            metrics[f"reflector/classification_{stat}"] = float(audit.get(stat, 0))
        metrics.update(write_new_experiences(memory, additions, step))
    management = trainer.config.env.experience_memory.get("management", {})
    if management.get("eviction_enabled", True):
        # Single-track conservative eviction + UCB capacity management.
        eviction = memory.evict_excess_experiences(
            step,
            max_task_experiences=int(management.get("eviction_max_task_experiences", 200)),
            max_step_experiences=int(management.get("eviction_max_step_experiences", 200)),
            protect_recent_steps=int(management.get("eviction_protect_recent_steps", cfg.protect_recent_steps)),
            score_c=float(management.get("eviction_score_c", 1.0)),
            min_exposures=int(management.get("eviction_min_exposures", 20)),
            utility_threshold=float(management.get("eviction_utility_threshold", -0.05)),
            negative_windows=int(management.get("eviction_negative_windows", 2)),
        )
    else:
        eviction = {"removed": [], "warnings": []}
    negative = [x for x in eviction["removed"] if x["reason"] == "hard_negative_utility"]
    capacity_removed = [x for x in eviction["removed"] if x["reason"] == "capacity_retain_score"]
    metrics["bank/deleted_negative"] = len(negative)
    metrics["bank/deleted_capacity"] = len(capacity_removed)
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
        metrics["bank/utility_positive_rate"] = statistics.mean(
            u > 0 for u in utilities
        )
    metrics["bank/reflector_allowed"] = float(bool(contrasts))
    root = Path(trainer.config.trainer.default_local_dir)
    record = dict(
        step=step,
        metrics=metrics,
        utility_updates=updates,
        deleted_negative=negative,
        deleted_capacity=dict(eviction, removed=capacity_removed),
        evidence=evidence.state,
        reflector_source_leaf_ids=[x["source_leaf_ids"] for x in contrasts],
        reflector_failure_reason=getattr(getattr(trainer, "experience_updater", None), "last_failure_reason", None),
    )
    atomic_json(root / f"tree_management_step{step}.json", record)
    memory.save_experiences(str(root / f"experience_bank_step{step}.json"))
    trainer._sync_experiences_to_retrieval_server(
        memory, path=str(root / f"experience_bank_step{step}.json")
    )
    return metrics


def write_new_experiences(memory, additions, step):
    groups = {"task": [], "step": []}
    for index, item in enumerate(additions):
        kind = item["experience_type"]
        clean = memory._legacy_experience_to_step_experience(item, index + 1)
        clean.pop("experience_id", None)
        clean.update(
            utility=0.0,
            retrieval_count=0,
            last_retrieval_step=None,
            created_at_step=step,
        )
        clean["evidence"] = dict(
            positive_gain_count=0,
            negative_gain_count=0,
            source_traj_uids=item.get("evidence", {}).get("source_traj_uids", []),
            last_updated_step=step,
        )
        groups[kind].append(clean)
    before = {
        pool: len(memory.experiences.get(pool, []))
        for pool in ("task_experiences", "step_experiences")
    }
    added = memory.upsert_experience_groups(
        task_experiences=groups["task"], step_experiences=groups["step"]
    )
    return {
        "bank/added": added,
        "reflector/added_task": len(memory.experiences["task_experiences"])
        - before["task_experiences"],
        "reflector/added_step": len(memory.experiences["step_experiences"])
        - before["step_experiences"],
    }
