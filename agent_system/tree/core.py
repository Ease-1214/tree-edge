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
                and max(l_rewards) == 0
                and max(r_rewards) == 1
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
        self, bank, step, beta=0.1, min_questions=20, protect_recent_steps=20
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
                else:
                    kept.append(item)
            bank[pool] = kept
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


# --------------------------------------------------------------------------- #
# T069: reflector material = hardest solvable questions of the current batch   #
# --------------------------------------------------------------------------- #

import random as _random


def _bank_lookup(bank):
    table = {}
    for pool in ("task_experiences", "step_experiences"):
        for item in bank.get(pool, []) or []:
            table[str(item.get("experience_id"))] = item
    return table


def _experience_line(item):
    head = item.get("title") or item.get("trigger") or item.get("when_to_apply") or ""
    body = item.get("principle") or item.get("recommended_action") or ""
    return (head + (": " if head and body else "") + body).strip()


def _turn_records(tree, leaf, lookup):
    turns = []
    for node_id in leaf.node_ids:
        node = tree.nodes[node_id]
        task_ids = list((node.state_before.task_memory or {}).get("experience_ids", []) or [])
        if node.depth == 1 and node.branch == "R":
            task_ids = list(node.new_experience_ids)
        step_ids = list(node.new_experience_ids) if node.depth > 1 else []
        turns.append(dict(
            node_id=node_id,
            depth=node.depth,
            task_experiences=[_experience_line(lookup[e]) for e in task_ids if e in lookup],
            step_experiences=[_experience_line(lookup[e]) for e in step_ids if e in lookup],
            experience_ids=task_ids + step_ids,
            action=node.action,
            observation=node.observation,
        ))
    return turns


def trajectory_reflector_examples(tree, k=10, seed=0, zero_success_quota=0):
    """Pick the k questions with the LOWEST leaf success rate that still have at least one
    successful and one failed leaf (ties broken at random), and return for each one failed
    and one successful root-to-leaf trajectory with the experiences injected at every turn.

    zero_success_quota > 0 additionally admits that many all-failed questions (failed
    trajectory only); default 0.
    """
    rng = _random.Random(seed)
    lookup = _bank_lookup(tree.bank_snapshot)
    by_q = defaultdict(list)
    for leaf in tree.leaves:
        by_q[leaf.uid].append(leaf)

    def has_exp(leaf):
        return any(tree.nodes[n].new_experience_ids or tree.nodes[n].path_has_T for n in leaf.node_ids)

    solvable, zero = [], []
    for uid, leaves in by_q.items():
        rate = statistics.mean(l.reward for l in leaves)
        if 0 < rate < 1:
            solvable.append((rate, rng.random(), uid))
        elif rate == 0:
            zero.append((rng.random(), uid))
    solvable.sort()
    zero.sort()
    picked = [(uid, True) for _, _, uid in solvable[: max(0, k - zero_success_quota)]]
    picked += [(uid, False) for _, uid in zero[:zero_success_quota]]
    picked = picked[:k]

    examples = []
    for uid, with_success in picked:
        leaves = by_q[uid]
        failed = [l for l in leaves if l.reward == 0]
        failed_exp = [l for l in failed if has_exp(l)]
        fail_leaf = rng.choice(failed_exp or failed)
        succ_leaf = rng.choice([l for l in leaves if l.reward > 0]) if with_success else None
        state = tree.nodes[fail_leaf.node_ids[0]].state_before
        diverge = 1
        if succ_leaf is not None:
            for i, (a, b) in enumerate(zip(fail_leaf.node_ids, succ_leaf.node_ids)):
                if a != b:
                    diverge = tree.nodes[a].depth
                    break
        examples.append(dict(
            mode="trajectory_pair",
            task=state.question,
            data_source=state.data_source,
            leaf_success_rate=statistics.mean(l.reward for l in leaves),
            n_leaves=len(leaves),
            failed_turns=_turn_records(tree, fail_leaf, lookup),
            success_turns=_turn_records(tree, succ_leaf, lookup) if succ_leaf else None,
            divergence_depth=diverge,
            source_leaf_ids=[fail_leaf.leaf_id] + ([succ_leaf.leaf_id] if succ_leaf else []),
        ))
    return examples

