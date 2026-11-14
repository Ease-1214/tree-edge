#!/usr/bin/env python3
"""T71 read-only probe: structural Fork filter -> delta<0 -> L-success/R-failure pair."""
import argparse
import json
from collections import Counter, defaultdict
from statistics import mean

import torch

from agent_system.tree.core import normalize_action
from agent_system.tree.reflector_selection_v3 import (
    _action_difference,
    _downstream_experience_count,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot", required=True)
    parser.add_argument("--details", action="store_true")
    args = parser.parse_args()
    tree = torch.load(args.snapshot, map_location="cpu", weights_only=False)

    descendants = defaultdict(list)
    for leaf in tree.leaves:
        if leaf.source != "tree":
            continue
        for node_id in leaf.node_ids:
            descendants[node_id].append(leaf)

    counts = Counter()
    failures = Counter()
    pairs = []
    negative_forks = []
    for fork in tree.forks.values():
        counts["all_forks"] += 1
        left_node = tree.nodes.get(fork.left_node_id)
        right_node = tree.nodes.get(fork.right_node_id)
        if left_node is None or right_node is None:
            failures["missing_node"] += 1
            continue
        if not fork.new_experience_ids:
            failures["no_new_experience"] += 1
            continue
        if (
            left_node.uid != right_node.uid
            or left_node.depth != right_node.depth
            or left_node.state_before.history != right_node.state_before.history
            or left_node.state_before.last_observation != right_node.state_before.last_observation
        ):
            failures["not_same_state"] += 1
            continue
        left = descendants[fork.left_node_id]
        right = descendants[fork.right_node_id]
        if not left or not right:
            failures["missing_terminal_side"] += 1
            continue
        counts["same_state_terminal_with_injection"] += 1
        left_action = normalize_action(left_node.action)
        right_action = normalize_action(right_node.action)
        if left_action[0] == "invalid" or right_action[0] == "invalid":
            failures["invalid_action"] += 1
            continue
        if left_action == right_action:
            failures["same_normalized_action"] += 1
            continue
        counts["structural_distinct_action"] += 1
        lrate, rrate = mean(x.reward for x in left), mean(x.reward for x in right)
        if rrate < lrate:
            counts["negative_distinct_action"] += 1
        elif rrate == lrate:
            counts["zero_distinct_action"] += 1
        else:
            counts["positive_distinct_action"] += 1
        if _action_difference(left_node.action, right_node.action) is None:
            failures["similar_search_query"] += 1
            continue
        counts["structural_meaningful_action"] += 1
        if rrate < lrate:
            counts["negative_meaningful_action"] += 1
        elif rrate == lrate:
            counts["zero_meaningful_action"] += 1
        else:
            counts["positive_meaningful_action"] += 1
        if not (rrate < lrate):
            continue
        positive_left = [x for x in left if x.reward == 1]
        negative_right = [x for x in right if x.reward == 0]
        if not positive_left or not negative_right:
            failures["negative_no_valid_pair"] += 1
            continue
        counts["negative_forks_with_pair"] += 1
        counts["all_possible_l_success_r_failure_pairs"] += (
            len(positive_left) * len(negative_right)
        )
        pair = min(
            ((l, r) for l in positive_left for r in negative_right),
            key=lambda lr: (
                _downstream_experience_count(tree, lr[0], fork.depth)
                + _downstream_experience_count(tree, lr[1], fork.depth),
                lr[0].leaf_id,
                lr[1].leaf_id,
            ),
        )
        negative_forks.append({
            "fork_id": fork.fork_id,
            "uid": fork.uid,
            "depth": fork.depth,
            "delta": round(rrate - lrate, 6),
            "left_rewards": [x.reward for x in left],
            "right_rewards": [x.reward for x in right],
            "new_experience_ids": list(fork.new_experience_ids),
            "left_success_leaf": pair[0].leaf_id,
            "right_failure_leaf": pair[1].leaf_id,
            "later_experience_interventions": (
                _downstream_experience_count(tree, pair[0], fork.depth)
                + _downstream_experience_count(tree, pair[1], fork.depth)
            ),
        })
        pairs.append(pair)

    result = {
        "snapshot": args.snapshot,
        "counts": dict(sorted(counts.items())),
        "structural_rejections": dict(sorted(failures.items())),
        "selected_pairs": len(pairs),
        "unique_left_success_leaves": len({l.leaf_id for l, _ in pairs}),
        "unique_right_failure_leaves": len({r.leaf_id for _, r in pairs}),
        "unique_selected_leaf_ids": len({
            leaf.leaf_id for pair in pairs for leaf in pair
        }),
        "unique_questions_with_selected_pairs": len({
            item["uid"] for item in negative_forks
        }),
        "selected_task_level_forks": sum(x["depth"] == 1 for x in negative_forks),
        "selected_step_level_forks": sum(x["depth"] > 1 for x in negative_forks),
    }
    if args.details:
        result["selected_forks"] = negative_forks
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
