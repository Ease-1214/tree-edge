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
    """Find one successful/failed pair whose later path has minimal confounds."""
    left_node, right_node = audit["left_node_id"], audit["right_node_id"]
    left = [leaves[k] for k in audit["left_leaf_ids"] if k in leaves]
    right = [leaves[k] for k in audit["right_leaf_ids"] if k in leaves]
    if not left or not right:
        return None
    lrate = statistics.mean(x.reward for x in left)
    rrate = statistics.mean(x.reward for x in right)
    # Positive and negative interventions are equally useful.
    if lrate == rrate:
        return None
    good, bad = (left, right) if lrate > rrate else (right, left)
    wins = [x for x in good if x.reward == 1]
    losses = [x for x in bad if x.reward == 0]
    if not wins or not losses:
        return None
    depth = tree.nodes[left_node].depth
    pairs = [(w, l) for w in wins for l in losses]
    # The root intervention (T) is an intended difference; only count
    # *subsequent* extra interventions as possible confounders.
    pairs.sort(key=lambda pair: (
        _downstream_experience_count(tree, pair[0], depth)
        + _downstream_experience_count(tree, pair[1], depth),
        abs(len(pair[0].node_ids) - len(pair[1].node_ids)),
        len(pair[0].node_ids) + len(pair[1].node_ids),
        pair[0].leaf_id, pair[1].leaf_id,
    ))
    return pairs[0], lrate, rrate


def select_reflector_forks(tree, *, k=10, seed=0, zero_success_quota=0, sampling="ranked"):
    """Select at most k distinct questions with informative branch contrasts.

    Strong forks: one subtree has no success, the other has >=1 success.
    Weaker forks with unequal mean outcomes are fallbacks.
    A small root representation preference prevents many internal forks
    crowding out potential multi-turn task strategies. It does NOT require
    the model to produce T experiences.
    """
    if zero_success_quota:
        # All-failed questions cannot provide the requested successful/failure pair.
        raise ValueError("v3 fork contrast does not support zero-success-only cases")
    if sampling not in ("ranked", "random"):
        raise ValueError(f"Unknown fork sampling mode: {sampling}")
    rng = random.Random(seed)
    lookup = _bank_lookup(tree.bank_snapshot)
    leaves = {leaf.leaf_id: leaf for leaf in tree.leaves}
    rates_by_uid = defaultdict(list)
    for leaf in tree.leaves:
        rates_by_uid[leaf.uid].append(leaf.reward)
    candidates = []
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
        strict = max(audit["left_rewards"]) != max(audit["right_rewards"])
        diff = abs(lrate - rrate)
        # Subtree score difference is evidence of a useful contrast, not an
        # estimate of the causal effect of the experience.
        score = (
            int(strict), round(diff, 5), round(difference, 5),
            -(_downstream_experience_count(tree, success, lnode.depth)
              + _downstream_experience_count(tree, failure, lnode.depth)),
            -len(success.node_ids) - len(failure.node_ids),
            rng.random(),
        )
        candidates.append(dict(
            score=score, uid=lnode.uid, audit=audit,
            action_distinction=difference,
            action_family="/".join(sorted([
                normalize_action(lnode.action)[0],
                normalize_action(rnode.action)[0],
            ])),
            success=success, failure=failure, root=lnode.depth == 1,
            lrate=lrate, rrate=rrate, strict=strict,
            fork_depth=lnode.depth, left=lnode, right=rnode,
        ))

    roots = sorted((c for c in candidates if c["root"]),
                   key=lambda c: c["score"], reverse=True)
    selected = []
    seen = set()
    seen_questions = set()

    def take(c):
        question_key = (
            c["left"].state_before.data_source,
            c["left"].state_before.question.strip().casefold(),
        )
        if (c["uid"] in seen or question_key in seen_questions
                or len(selected) >= k):
            return
        seen.add(c["uid"])
        seen_questions.add(question_key)
        selected.append(c)

    # Reserve visibility for early planning without forcing a quota of T
    # experiences. Then select by informativeness + diversity, so dozens of
    # near-identical root queries cannot crowd out all internal decisions.
    root_preference = 0 if sampling == "random" else min(k, 2)
    planning_roots = [
        c for c in roots if c["action_family"] == "search/search"
    ]
    for c in (planning_roots or roots)[:root_preference]:
        take(c)

    while len(selected) < k:
        remaining = [
            c for c in candidates
            if c["uid"] not in seen
            and (
                c["left"].state_before.data_source,
                c["left"].state_before.question.strip().casefold(),
            ) not in seen_questions
        ]
        if not remaining:
            break

        def diverse_score(c):
            depth_count = sum(x["root"] == c["root"] for x in selected)
            family_count = sum(
                x["action_family"] == c["action_family"] for x in selected
            )
            left = c["audit"]["left_rewards"]
            right = c["audit"]["right_rewards"]
            diff = abs(c["lrate"] - c["rrate"])
            confounds = (
                _downstream_experience_count(tree, c["success"], c["fork_depth"])
                + _downstream_experience_count(tree, c["failure"], c["fork_depth"])
            )
            # Previous ranking selected mostly "guess an answer vs search",
            # which yields generic already-banked "do not guess" advice.
            # Prefer differences in search plans and evidence interpretation,
            # while retaining a minority of answer/search contrasts.
            strategy_bonus = {
                "search/search": 2.0,
                "answer/answer": 0.6,
                "answer/search": 0.15,
            }.get(c["action_family"], 0.0)
            return (
                2 * int(c["strict"])
                + 1.4 * diff
                + 0.7 * c["action_distinction"]
                + strategy_bonus
                - 0.38 * depth_count
                - 0.52 * family_count
                - 0.12 * confounds,
                c["score"],
            )
        if sampling == "random":
            # Uniformly pick eligible forks after the SAME validity filters.
            # Still keep distinct questions, as in the ranked implementation.
            remaining.sort(key=lambda c: c["audit"]["fork_id"])
            take(rng.choice(remaining))
        else:
            take(max(remaining, key=diverse_score))
    result = []
    for c in selected:
        audit, left, right = c["audit"], c["left"], c["right"]
        common_state = left.state_before
        result.append({
            "mode": "trajectory_pair",
            "reflector_version": "v3",
            "task": common_state.question,
            "data_source": common_state.data_source,
            "fork_id": audit["fork_id"],
            "fork_depth": c["fork_depth"],
            "divergence_depth": c["fork_depth"],
            "fork_tier": "strict" if c["strict"] else "relaxed",
            "leaf_success_rate": statistics.mean(rates_by_uid[c["uid"]]),
            "n_leaves": len(rates_by_uid[c["uid"]]),
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
