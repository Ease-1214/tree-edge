"""Reflector v3: choose informative same-state forks, then full root-to-leaf paths.

No rollouts are generated here. Selection is a sampling heuristic, NOT causal
attribution. The Actor/utility system is unchanged.
"""
from __future__ import annotations

from collections import defaultdict
from difflib import SequenceMatcher
import random
import re
import statistics

from agent_system.tree.core import _bank_lookup, _turn_records, normalize_action


def _action_difference(left: str, right: str):
    lk, lt = normalize_action(left)
    rk, rt = normalize_action(right)
    if (lk, lt) == (rk, rt):
        return None
    if lk == rk == "answer":
        # Answer-only changes are often entity/format guesses, not reusable plans.
        return 0.15
    if lk != rk:
        return 1.0
    if lk == rk == "search":
        tokens = lambda s: set(re.findall(r"[\w]+", s.casefold()))
        x, y = tokens(lt), tokens(rt)
        jaccard = len(x & y) / max(1, len(x | y))
        sequence = SequenceMatcher(None, lt.casefold(), rt.casefold()).ratio()
        # Do not turn "the", word order, or a nearly identical search into advice.
        if jaccard >= .92 or (jaccard >= .83 and sequence >= .83):
            return None
        return max(.18, 1 - .55 * jaccard - .25 * sequence)
    return 0.25


def _downstream_experience_count(tree, leaf, fork_depth):
    """Exclude task/step differences at the studied fork itself."""
    return sum(
        bool(tree.nodes[n].new_experience_ids)
        for n in leaf.node_ids
        if tree.nodes[n].depth > fork_depth
    )


def _paired_leaves(tree, audit, leaves):
    """Find one successful Left / failed Right pair whose later path has minimal confounds."""
    left_node, right_node = audit["left_node_id"], audit["right_node_id"]
    left = [leaves[k] for k in audit["left_leaf_ids"] if k in leaves]
    right = [leaves[k] for k in audit["right_leaf_ids"] if k in leaves]
    if not left or not right:
        return None
    lrate = float(audit.get("left_mean", statistics.mean(x.reward for x in left)))
    rrate = float(audit.get("right_mean", statistics.mean(x.reward for x in right)))
    # Only extract new experiences when the without-experience Left branch
    # succeeded and the with-experience Right branch failed (delta < 0).
    if rrate >= lrate:
        return None
    if max(audit["left_rewards"]) != 1 or max(audit["right_rewards"]) != 0:
        return None
    wins = [x for x in left if x.reward == 1]
    losses = [x for x in right if x.reward == 0]
    if not wins or not losses:
        return None
    depth = tree.nodes[left_node].depth
    pairs = [(w, l) for w in wins for l in losses]
    # Within the chosen fork, prefer the leaf pair with minimal subsequent
    # experience injections so the trajectory contrast stays clean.
    pairs.sort(key=lambda pair: (
        _downstream_experience_count(tree, pair[0], depth)
        + _downstream_experience_count(tree, pair[1], depth),
        abs(len(pair[0].node_ids) - len(pair[1].node_ids)),
        len(pair[0].node_ids) + len(pair[1].node_ids),
        pair[0].leaf_id, pair[1].leaf_id,
    ))
    return pairs[0], lrate, rrate


def select_reflector_forks(tree, *, k=10, seed=0, zero_success_quota=0, sampling="ranked"):
    """Select distinct-question harmful forks (delta < 0) via stratified depth bucketing.

    Eligible forks must have delta = rrate - lrate < 0 with a successful Left
    leaf and failed Right subtree. Candidates are deduplicated per question and
    stratified across three depth buckets (depth == 1, depth == 2, depth >= 3),
    up to 4 per bucket, without padding from delta >= 0 forks.
    """
    if zero_success_quota:
        raise ValueError("v3 fork contrast does not support zero-success-only cases")
    if sampling not in ("ranked", "random"):
        raise ValueError(f"Unknown fork sampling mode: {sampling}")
    rng = random.Random(seed)
    lookup = _bank_lookup(tree.bank_snapshot)
    leaves = {leaf.leaf_id: leaf for leaf in tree.leaves}
    rates_by_uid = defaultdict(list)
    for leaf in tree.leaves:
        rates_by_uid[leaf.uid].append(leaf.reward)
    raw_candidates = []
    for audit in tree.audits:
        lnode = tree.nodes[audit["left_node_id"]]
        rnode = tree.nodes[audit["right_node_id"]]
        if lnode.depth != rnode.depth or lnode.uid != rnode.uid:
            continue
        difference = _action_difference(lnode.action, rnode.action)
        if difference is None:
            continue
        pair = _paired_leaves(tree, audit, leaves)
        if pair is None:
            continue
        (success, failure), lrate, rrate = pair
        strict = True
        diff = lrate - rrate
        action_family = "/".join(sorted([
            normalize_action(lnode.action)[0],
            normalize_action(rnode.action)[0],
        ]))
        # Rank within each depth bucket by search/search preference, magnitude
        # of harmful delta (lrate - rrate), and action distinction. Do NOT
        # penalize earlier depths for having more downstream turns.
        score = (
            int(action_family == "search/search"),
            round(diff, 5),
            round(difference, 5),
            rng.random(),
        )
        raw_candidates.append(dict(
            score=score, uid=lnode.uid, audit=audit,
            action_distinction=difference,
            action_family=action_family,
            success=success, failure=failure, root=lnode.depth == 1,
            lrate=lrate, rrate=rrate, strict=strict,
            fork_depth=lnode.depth, left=lnode, right=rnode,
        ))

    # Deduplicate by question (keeping the highest-scoring fork per question)
    raw_candidates.sort(key=lambda c: c["score"], reverse=True)
    deduped = []
    seen_uids = set()
    seen_questions = set()
    for c in raw_candidates:
        question_key = (
            c["left"].state_before.data_source,
            c["left"].state_before.question.strip().casefold(),
        )
        if c["uid"] in seen_uids or question_key in seen_questions:
            continue
        seen_uids.add(c["uid"])
        seen_questions.add(question_key)
        deduped.append(c)

    # Stratified depth bucketing: Bucket 1 (depth==1), Bucket 2 (depth==2), Bucket 3 (depth>=3)
    buckets = {1: [], 2: [], 3: []}
    for c in deduped:
        b_idx = 1 if c["fork_depth"] == 1 else (2 if c["fork_depth"] == 2 else 3)
        buckets[b_idx].append(c)
    for b_idx in (1, 2, 3):
        buckets[b_idx].sort(key=lambda c: c["score"], reverse=True)

    per_bucket = 4
    selected = []
    for round_idx in range(per_bucket):
        for b_idx in (1, 2, 3):
            if round_idx < len(buckets[b_idx]) and len(selected) < k:
                selected.append(buckets[b_idx][round_idx])
    if len(selected) < k:
        selected_ids = {c["audit"]["fork_id"] for c in selected}
        for c in deduped:
            if len(selected) >= k:
                break
            if c["audit"]["fork_id"] not in selected_ids:
                selected.append(c)
                selected_ids.add(c["audit"]["fork_id"])

    result = []
    for c in selected:
        audit, left, right = c["audit"], c["left"], c["right"]
        common_state = left.state_before
        task_ids = list((common_state.task_memory or {}).get("experience_ids", []) or [])
        common_task_exps = [
            {
                "experience_id": eid,
                "trigger": lookup[eid].get("trigger") or lookup[eid].get("when_to_apply") or lookup[eid].get("title") or "",
                "recommended_action": lookup[eid].get("recommended_action") or lookup[eid].get("principle") or "",
                "avoid_action": lookup[eid].get("avoid_action") or "",
            }
            for eid in task_ids if eid in lookup
        ] if c["fork_depth"] > 1 else []
        added_exps = [
            {
                "experience_id": eid,
                "trigger": lookup[eid].get("trigger") or lookup[eid].get("when_to_apply") or lookup[eid].get("title") or "",
                "recommended_action": lookup[eid].get("recommended_action") or lookup[eid].get("principle") or "",
                "avoid_action": lookup[eid].get("avoid_action") or "",
            }
            for eid in right.new_experience_ids if eid in lookup
        ]
        result.append({
            "mode": "trajectory_pair",
            "reflector_version": "v3",
            "task": common_state.question,
            "data_source": common_state.data_source,
            "fork_id": audit["fork_id"],
            "fork_depth": c["fork_depth"],
            "divergence_depth": c["fork_depth"],
            "fork_tier": "strict",
            "leaf_success_rate": statistics.mean(rates_by_uid[c["uid"]]),
            "n_leaves": len(rates_by_uid[c["uid"]]),
            "common_task_experience": common_task_exps,
            "added_experiences": added_exps,
            "common_search_history": list(common_state.history),
            "fork_observation": common_state.last_observation,
            "left_action": left.action,
            "right_action": right.action,
            "left_experience_ids": list(left.new_experience_ids),
            "right_experience_ids": list(right.new_experience_ids),
            "left_subtree_rate": c["lrate"],
            "right_subtree_rate": c["rrate"],
            "left_subtree_leaves": len(audit["left_rewards"]),
            "right_subtree_leaves": len(audit["right_rewards"]),
            "success_turns": _turn_records(tree, c["success"], lookup),
            "failed_turns": _turn_records(tree, c["failure"], lookup),
            "source_leaf_ids": [c["failure"].leaf_id, c["success"].leaf_id],
            "success_post_fork_experience_count":
                _downstream_experience_count(tree, c["success"], c["fork_depth"]),
            "failed_post_fork_experience_count":
                _downstream_experience_count(tree, c["failure"], c["fork_depth"]),
        })
    return result
