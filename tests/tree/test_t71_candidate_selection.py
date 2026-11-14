"""T71: semantic Task-4 / Step-6 shortlist -> Utility/UCB best-1 injection."""
import math
from copy import deepcopy

import numpy as np
import pytest

from agent_system.memory.experience_memory import ExperienceMemory
from agent_system.tree.runner import _choose_single_fork_experience


def _item(idx, pool, utility=0.0, retrieval_count=15):
    kind = "task" if pool == "task_experiences" else "step"
    return {
        "experience_id":f"{kind}_{idx:06d}",
        "task_type":"direct_retrieval",
        "title":f"strategy {idx}",
        "principle":f"do action {idx}",
        "when_to_apply":f"condition {idx}",
        "retrieval_obs":f"strategy {idx}",
        "retrieval_count":retrieval_count,
        "utility":utility,
    }


def test_local_embedding_semantic_shortlist_before_utility_ucb_ranking(monkeypatch):
    bank = ExperienceMemory(
        load_initial_experiences=False, retrieval_mode="embedding",
        retrieval_alpha=0.5, retrieval_ucb_c=0.5,
        retrieval_top_2k=2000, task_specific_top_k=4,
    )
    pool = "task_experiences"
    sims = [0.99, 0.98, 0.97, 0.96, 0.95, 0.90]
    items = [_item(i, pool, utility=(0.8 if i == 4 else 5.0 if i == 5 else 0))
             for i in range(1, 7)]
    bank.experiences[pool] = items
    bank._normalize_all_experience_meta()
    vectors = np.asarray([[sim, math.sqrt(1-sim*sim)] for sim in sims], dtype=np.float32)
    monkeypatch.setattr(bank, "_compute_pool_embeddings",
                        lambda kind: {"embeddings":vectors, "items":items})
    monkeypatch.setattr(bank, "_encode_texts",
                        lambda texts, normalize_embeddings=True: np.array([[1.,0.]],dtype=np.float32))

    batch = bank.retrieve_task_experiences_batch(
        ["Find a specific entity relation"], top_k=4, semantic_candidate_k=4
    )
    candidates = batch[0]["task_experiences"]
    assert len(candidates) == 4
    assert {x["experience_id"] for x in candidates} == {
        f"task_{i:06d}" for i in range(1,5)
    }
    # Among the nearest four, strategy 4 is best on blended score.
    # Strategy 5 has huge Utility but is NOT in the similarity shortlist.
    selected = _choose_single_fork_experience(batch[0], pool, 4)
    assert selected["experience_ids"] == ["task_000004"]
    assert selected["task_experiences"][0]["experience_id"] == "task_000004"
    assert selected["step_experiences"] == []
    assert len(selected["experiences"]) == 1


@pytest.mark.parametrize("pool,k", [("task_experiences",4), ("step_experiences",6)])
def test_remote_retrieval_candidate_k_exactly_semantic_topk(monkeypatch,pool,k):
    memory = ExperienceMemory(
        load_initial_experiences=False, retrieval_mode="embedding",
        experience_retrieval_service_url="http://retrieval.local:29700",
        retrieval_alpha=0.5, retrieval_ucb_c=0.5,
        retrieval_top_2k=2000, task_specific_top_k=4,
    )
    memory.experiences[pool] = [_item(i,pool,utility=0.4 if i == k else 0)
                                for i in range(1,k+1)]
    memory._normalize_all_experience_meta()
    posted=[]
    class Response:
        def __init__(self, payload):
            self.payload=payload
        def raise_for_status(self):
            return None
        def json(self):
            items=[dict(memory.experiences[pool][i],
                        similarity=0.99-i*0.01)
                   for i in range(k)]
            return {"result":[{
                "query_text":self.payload["queries"][0],
                pool:items,
            }]}
    class Session:
        def post(self,url,json,timeout):
            posted.append((url,json))
            return Response(json)
    monkeypatch.setattr(memory,"_get_http_session",lambda url:Session())
    if pool == "task_experiences":
        result=memory.retrieve_task_experiences_batch(
            ["test relation"],top_k=k,semantic_candidate_k=k)
    else:
        result=memory.retrieve_step_experiences_batch(
            ["test relation Current observation: one result"],top_k=k,
            semantic_candidate_k=k)
    assert len(posted) == 1
    _,sent = posted[0]
    assert sent["candidate_k"] == k  # not global retrieval_top_2k=2000
    assert sent["top_k"] == k
    assert sent["pool"] == pool
    assert len(result[0][pool]) == k
    chosen=_choose_single_fork_experience(result[0],pool,k)
    assert chosen["experience_ids"] == [f"{'task' if k == 4 else 'step'}_{k:06d}"]


def test_selection_disallows_missing_rank_score_or_excess_candidates():
    pool="step_experiences"
    two=[_item(i,pool) for i in (1,2)]
    with pytest.raises(ValueError, match="retrieval_score"):
        _choose_single_fork_experience(
            {pool:two,"experience_ids":[x["experience_id"] for x in two]},pool,6)
    four=[dict(_item(i,pool),retrieval_score=float(7-i)) for i in range(1,5)]
    with pytest.raises(ValueError, match="expected at most"):
        _choose_single_fork_experience(
            {pool:four,"experience_ids":[x["experience_id"] for x in four]},pool,3)


def test_tree_integrated_top4_top6_best_one_and_inherited_task(monkeypatch,tmp_path):
    import test_t069_conditional_natural as synthetic

    task_pool=[_item(i,"task_experiences") for i in range(1,5)]
    step_pool=[_item(i,"step_experiences") for i in range(1,7)]
    monkeypatch.setattr(synthetic.Memory, "experiences", {
        "task_experiences":task_pool,
        "step_experiences":step_pool,
    })

    query_log=[]
    def payload(pool, best):
        rows=[]
        for i,item in enumerate(task_pool if pool=="task_experiences" else step_pool):
            x=dict(item,similarity=0.99-i*.01,
                   retrieval_score=0.9 if i == best-1 else 0.3-i*.01)
            rows.append(x)
        return {
            pool:rows,
            "task_experiences":rows if pool=="task_experiences" else [],
            "step_experiences":rows if pool=="step_experiences" else [],
            "experiences":rows,
            "experience_ids":[x["experience_id"] for x in rows],
        }
    def task(self,texts,**kwargs):
        query_log.append(("task",kwargs))
        return [deepcopy(payload("task_experiences",3)) for _ in texts]
    def step(self,texts,**kwargs):
        query_log.append(("step",kwargs))
        return [deepcopy(payload("step_experiences",5)) for _ in texts]
    monkeypatch.setattr(synthetic.Memory,"retrieve_task_experiences_batch",task)
    monkeypatch.setattr(synthetic.Memory,"retrieve_step_experiences_batch",step)

    from types import SimpleNamespace
    from agent_system.tree import runner

    def execute(config, backend, nodes, actions):
        results=[]
        for node in nodes:
            if node.source == "natural":
                results.append((("natural observation", 0., True, {"tool_calling":1}), []))
            else:
                done = node.depth == 2
                reward = float(done and node.branch == "R")
                results.append((("tree observation",reward,done,{"tool_calling":1}), []))
        return results

    monkeypatch.setattr(runner, "execute_decisions", execute)
    collector = synthetic.Collector(tmp_path)
    memory = synthetic.Memory()
    manager = SimpleNamespace(
        retrieval_memory=memory,envs=None,
        projection_f=lambda responses:(
            [f"<search>query-{i%2}</search>" for i in range(len(responses))],
            [True]*len(responses),
        ),
    )
    batch = runner.tree_multi_turn_loop(
        collector, synthetic.case_batch([0]), synthetic.Actor(), manager
    )
    tree = collector.last_tree
    assert len(tree.forks) == 3  # root, depth-2 L and R
    assert all(len(fork.new_experience_ids) == 1 for fork in tree.forks.values())
    assert [fork.new_experience_ids for fork in tree.forks.values()][0] == ("task_000003",)
    assert all(fork.new_experience_ids == ("step_000005",)
               for fork in tree.forks.values() if fork.depth==2)
    assert query_log[0] == ("task",{"top_k":4,"semantic_candidate_k":4})
    assert all(kind=="step" and opts=={"top_k":6,"semantic_candidate_k":6}
               for kind,opts in query_log[1:])
    deep_right=[node for node in tree.nodes.values()
                if node.depth==2 and node.branch=="R"]
    assert deep_right
    assert any("task_000003" in n.actual_prompt_text and
               "step_000005" in n.actual_prompt_text
               for n in deep_right)
    assert all("task_000004" not in n.actual_prompt_text and
               "step_000006" not in n.actual_prompt_text
               for n in tree.nodes.values())
    assert all("EXPERIENCE" not in n.noexp_prompt_text for n in tree.nodes.values())
