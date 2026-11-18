"""Map selected T71 Fork contrasts into the EDGE one-pass search updater.

Selection remains entirely in trajectory_reflector_examples().
"""
from typing import Dict, List


def edge_search_examples_from_forks(selected_forks: List[Dict]) -> List[Dict]:
    """Preserve fork provenance and both branch outcomes in EDGE input schema.

    Deliberately omit mode=trajectory_pair, which activates a T71-only
    specialized prompt. No new candidate filtering happens here.
    """
    examples = []
    for fork in selected_forks:
        left_leaves = int(fork.get("left_subtree_leaves") or 0)
        right_leaves = int(fork.get("right_subtree_leaves") or 0)
        left_rate = float(fork.get("left_subtree_rate") or 0)
        right_rate = float(fork.get("right_subtree_rate") or 0)
        delta = right_rate - left_rate
        item_dict = {
            "task": fork.get("task", ""),
            "task_type": "unknown",
            "fork_id": fork["fork_id"],
            "fork_depth": int(fork.get("fork_depth") or 1),
            "anchor_obs": fork.get("fork_observation") or "",
            "search_history": fork.get("common_search_history") or [],
            "left_action": fork.get("left_action") or "",
            "right_action": fork.get("right_action") or "",
            "left_experience_ids": fork.get("left_experience_ids") or [],
            "right_experience_ids": fork.get("right_experience_ids") or [],
            "common_task_experience": fork.get("common_task_experience") or {},
            "added_experiences": fork.get("added_experiences") or fork.get("right_experience_ids") or [],
            "left_subtree_rate": left_rate,
            "right_subtree_rate": right_rate,
            "delta": delta,
            "effect": "helped" if delta > 0 else "harmed" if delta < 0 else "neutral",
            "source_leaf_ids": fork.get("source_leaf_ids") or [],
            "success_turns": fork.get("success_turns") or [],
            "failed_turns": fork.get("failed_turns") or [],
            "paired_rollout": {
                "without_experience": {
                    "action": fork.get("left_action") or "",
                    "score": left_rate,
                    "successes": round(left_rate * left_leaves),
                    "total": left_leaves,
                    "experience_ids": fork.get("left_experience_ids") or [],
                },
                "with_experience": {
                    "action": fork.get("right_action") or "",
                    "score": right_rate,
                    "successes": round(right_rate * right_leaves),
                    "total": right_leaves,
                    "experience_ids": fork.get("right_experience_ids") or [],
                },
            },
        }
        if fork.get("reflector_version"):
            item_dict["reflector_version"] = fork["reflector_version"]
        examples.append(item_dict)
    return examples
