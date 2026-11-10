"""Tree identities, leaf statistics, conservative gates, and question-level evidence."""

from dataclasses import dataclass, field
from collections import defaultdict
from copy import deepcopy
import math
import hashlib
import re
import statistics


def tree_enabled(config, step):
    cfg = config.env.get("tree_experience", {})
    return bool(cfg.get("enable", False))


@dataclass
class SearchBranchState:
    uid: str
    question: str
    ground_truth: object
    data_source: str
    history: list = field(default_factory=list)
    env_history: list = field(default_factory=list)
    turns: int = 0
    done: bool = False
    task_memory: dict = field(default_factory=dict)
    last_observation: str = ""
    parent_id: str | None = None
    path: tuple = ()

    def clone(self):
        return deepcopy(self)


@dataclass
class DecisionNode:
    uid: str
    node_id: str
    parent_id: str | None
    fork_id: str
    depth: int
    branch: str
    state_before: SearchBranchState
    new_experience_ids: tuple
    actual_prompt_text: str
    noexp_prompt_text: str
    actual_prompt_ids: list = field(default_factory=list)
    response_ids: list = field(default_factory=list)
    response_mask: list = field(default_factory=list)
    action: str = ""
    observation: str = ""
    done: bool = False
    terminal_reward: float | None = None
    path_has_T: bool = False
    tensors: dict = field(default_factory=dict)
    valid: bool = True


@dataclass
class ForkRecord:
    fork_id: str
    uid: str
    depth: int
    left_node_id: str
    right_node_id: str
    new_experience_ids: tuple
    utility_snapshot: tuple


@dataclass
class Leaf:
    uid: str
    leaf_id: str
    node_ids: tuple
    reward: float
    data_source: str


@dataclass
class TreeRollout:
    nodes: dict = field(default_factory=dict)
    forks: dict = field(default_factory=dict)
    leaves: list = field(default_factory=list)
    audits: list = field(default_factory=list)
    metrics: dict = field(default_factory=dict)
    bank_snapshot: dict = field(default_factory=dict)

    def evaluate_forks(self):
        self.audits = []
        for fork in self.forks.values():
            left = [leaf for leaf in self.leaves if fork.left_node_id in leaf.node_ids]
            right = [
                leaf for leaf in self.leaves if fork.right_node_id in leaf.node_ids
            ]
            if not left or not right:
                raise ValueError("Fork lacks terminal leaf evidence: " + fork.fork_id)
            same = normalize_action(
                self.nodes[fork.left_node_id].action
            ) == normalize_action(self.nodes[fork.right_node_id].action)
            l_rewards = [leaf.reward for leaf in left]
            r_rewards = [leaf.reward for leaf in right]
            utility = (
                statistics.mean(fork.utility_snapshot)
                if fork.utility_snapshot
                else float("-inf")
            )
            delta = statistics.mean(r_rewards) - statistics.mean(l_rewards)
            informative = len(set(l_rewards + r_rewards)) > 1
            gate = (
                bool(fork.new_experience_ids)
                and not same
                and max(l_rewards) == 0
                and max(r_rewards) == 1
                and utility >= 0
            )
            self.audits.append(
                dict(
                    fork_id=fork.fork_id,
                    uid=fork.uid,
                    left_node_id=fork.left_node_id,
                    right_node_id=fork.right_node_id,
                    new_experience_ids=list(fork.new_experience_ids),
                    utility_snapshot=list(fork.utility_snapshot),
                    utility_mean=utility,
                    left_leaf_ids=[x.leaf_id for x in left],
                    right_leaf_ids=[x.leaf_id for x in right],
                    left_rewards=l_rewards,
                    right_rewards=r_rewards,
                    delta=delta,
                    same_action=same,
                    informative=informative,
                    gate=gate,
                    weight=sum(r_rewards) / (len(right) + 1) if gate else 0.0,
                    path_has_T=self.nodes[fork.left_node_id].path_has_T,
                )
            )
        return self.audits

    def expanded_rows(self):
        advantages = leaf_advantages(self.leaves)
        rows = []
        for leaf in sorted(self.leaves, key=lambda x: x.leaf_id):
            for node_id in leaf.node_ids:
                rows.append(
                    dict(
                        uid=leaf.uid,
                        traj_uid=leaf.leaf_id,
                        leaf_id=leaf.leaf_id,
                        node_id=node_id,
                        fork_id=self.nodes[node_id].fork_id,
                        row_id=leaf.leaf_id + ":" + node_id,
                        reward=leaf.reward,
                        advantage=advantages[leaf.leaf_id],
                        kd_weight=0.0,
                        kd_canonical=False,
                        is_padding_copy=False,
                    )
                )
        for fork in self.audits:
            if fork["gate"]:
                row = next(x for x in rows if x["node_id"] == fork["left_node_id"])
                row.update(kd_weight=fork["weight"], kd_canonical=True)
        return rows

    def question_credits(self):
        grouped = defaultdict(list)
        for fork in self.audits:
            if fork["same_action"] or not fork["informative"]:
                continue
            for eid in fork["new_experience_ids"]:
                state = self.nodes[fork["left_node_id"]].state_before
                question_id = hashlib.sha256(
                    (state.data_source + "\0" + state.question).encode()
                ).hexdigest()
                grouped[(eid, question_id)].append(fork["delta"])
        return {key: statistics.mean(values) for key, values in grouped.items()}


def normalize_action(action):
    for kind in ("search", "answer"):
        match = re.search(
            r"<" + kind + r">(.*?)</" + kind + ">", action, flags=re.I | re.S
        )
        if match:
            return kind, match.group(1).strip().lower()
    return "invalid", ""


def leaf_advantages(leaves, epsilon=1e-6):
    groups = defaultdict(dict)
    for leaf in leaves:
        previous = groups[leaf.uid].setdefault(leaf.leaf_id, leaf.reward)
        if previous != leaf.reward:
            raise ValueError("A leaf has inconsistent rewards")
    result = {}
    for rewards in groups.values():
        values = list(rewards.values())
        mean = statistics.mean(values)
        std = statistics.stdev(values) if len(values) > 1 else 0.0
        for lid, reward in rewards.items():
            result[lid] = (reward - mean) / (std + epsilon) if std > 0 else 0.0
    return result


class QuestionEvidence:
    """Persist distinct-question observations and a separate pending EMA window."""

    def __init__(self, state=None):
        self.state = state or {"observations": {}, "pending": {}, "contrasts": []}

    def record(self, credits):
        for (eid, uid), delta in credits.items():
            for pool in ("observations", "pending"):
                values = self.state[pool].setdefault(eid, {}).setdefault(uid, [])
                values.append(float(delta))

    def statistics(self, eid):
        values = [
            statistics.mean(x) for x in self.state["observations"].get(eid, {}).values()
        ]
        n = len(values)
        mean = statistics.mean(values) if n else 0.0
        sd = statistics.stdev(values) if n > 1 else 0.0
        se = sd / math.sqrt(n) if n > 1 else None
        return dict(
            n=n,
            mean=mean,
            sample_std=sd,
            standard_error=se,
            upper_2se=mean + 2 * se if se is not None else None,
        )

    def merge_window(
        self,
        bank,
        step,
        beta=0.1,
        min_questions=20,
        protect_recent_steps=20,
        max_delete_per_window=0,
    ):
        updates, removed = [], []
        for pool in ("task_experiences", "step_experiences"):
            kept = []
            for item in bank.get(pool, []):
                eid = item["experience_id"]
                pending = self.state["pending"].get(eid, {})
                stats = self.statistics(eid)
                if pending:
                    delta = statistics.mean(
                        statistics.mean(x) for x in pending.values()
                    )
                    old = float(item.get("utility", 0.0))
                    item["utility"] = (1 - beta) * old + beta * delta
                    item.setdefault("evidence", {})["last_updated_step"] = step
                    updates.append(
                        dict(
                            experience_id=eid,
                            old_utility=old,
                            utility=item["utility"],
                            delta_window=delta,
                        )
                    )
                item.setdefault("evidence", {})["tree_question_statistics"] = stats
                old_enough = (
                    step - int(item.get("created_at_step", 0)) >= protect_recent_steps
                )
                if (
                    stats["n"] >= max(2, min_questions)
                    and stats["upper_2se"] < 0
                    and old_enough
                ):
                    removed.append(
                        dict(
                            experience_id=eid,
                            pool=pool,
                            reason="statistical_negative",
                            **stats,
                        )
                    )
                kept.append(item)
            bank[pool] = kept
        if max_delete_per_window > 0:
            removed = sorted(
                removed,
                key=lambda item: (
                    item["upper_2se"],
                    item["pool"],
                    item["experience_id"],
                ),
            )[:max_delete_per_window]
        selected = {(item["pool"], item["experience_id"]) for item in removed}
        for pool in ("task_experiences", "step_experiences"):
            bank[pool] = [
                item
                for item in bank[pool]
                if (pool, item["experience_id"]) not in selected
            ]
        self.state["pending"] = {}
        return updates, removed


def fork_reflector_examples(tree):
    examples = []
    for audit in tree.audits:
        if audit["same_action"] or not (
            max(audit["left_rewards"]) == 0
            and max(audit["right_rewards"]) == 1
            or max(audit["right_rewards"]) == 0
            and max(audit["left_rewards"]) == 1
        ):
            continue
        left = tree.nodes[audit["left_node_id"]]
        right = tree.nodes[audit["right_node_id"]]
        state = left.state_before
        examples.append(
            dict(
                task=state.question,
                question_id=hashlib.sha256(
                    (state.data_source + "\0" + state.question).encode()
                ).hexdigest(),
                fork_depth=left.depth,
                common_task_experience=(
                    deepcopy(state.task_memory) if left.depth > 1 else {}
                ),
                task_type="unknown",
                anchor_obs=state.last_observation,
                search_history=deepcopy(state.history),
                fork_id=audit["fork_id"],
                delta=audit["delta"],
                effect=(
                    "helped"
                    if audit["delta"] > 0
                    else "harmed" if audit["delta"] < 0 else "neutral"
                ),
                traj_uid=audit["right_leaf_ids"][0],
                no_exp_traj_uid=audit["left_leaf_ids"][0],
                source_leaf_ids=audit["left_leaf_ids"] + audit["right_leaf_ids"],
                added_experiences=[
                    deepcopy(x)
                    for pool in ("task_experiences", "step_experiences")
                    for x in tree.bank_snapshot.get(pool, [])
                    if x["experience_id"] in audit["new_experience_ids"]
                ],
                paired_rollout={
                    "with_experience": {
                        "action": right.action,
                        "score": statistics.mean(audit["right_rewards"]),
                        "successes": sum(audit["right_rewards"]),
                        "total": len(audit["right_rewards"]),
                        "experience_ids": audit["new_experience_ids"],
                    },
                    "without_experience": {
                        "action": left.action,
                        "score": statistics.mean(audit["left_rewards"]),
                        "successes": sum(audit["left_rewards"]),
                        "total": len(audit["left_rewards"]),
                    },
                },
            )
        )
    return examples


def select_reflector_examples(examples):
    selected = {}
    for item in sorted(examples, key=lambda x: (-abs(x["delta"]), x["fork_id"])):
        selected.setdefault(item["question_id"], item)
    return list(selected.values())[:10]
