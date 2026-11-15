"""T71 uses EDGE original exact normalized fingerprint dedup + eviction."""
from agent_system.memory.experience_memory import ExperienceMemory
from agent_system.tree.training import write_new_experiences


def _entry(eid, retrieval, utility=0.0, retrieval_count=0, created=0, neg_windows=0):
    return {
        "experience_id":eid, "task_type":"direct_retrieval",
        "title":retrieval, "principle":"Follow exact relation from results",
        "when_to_apply":retrieval, "retrieval_obs":retrieval,
        "trigger":retrieval, "recommended_action":"Search for matching relation",
        "avoid_action":"Do not guess", "utility":utility,
        "retrieval_count":retrieval_count,"created_at_step":created,
        "evidence":{"consecutive_negative_windows":neg_windows,"source_traj_uids":[]},
    }


def test_edge_fingerprint_normalizes_whitespace_and_case_preserving_lifecycle():
    m=ExperienceMemory(load_initial_experiences=False)
    base=_entry("step_000001","Search the exact relation",utility=0.35,retrieval_count=22,created=30)
    m.experiences["step_experiences"]=[base]
    m._normalize_all_experience_meta()
    duplicate=_entry(None,"  SEARCH   THE EXACT RELATION  ",utility=0,retrieval_count=0,created=90)
    count=m.upsert_experience_groups(step_experiences=[duplicate])
    assert count==0
    assert len(m.experiences["step_experiences"])==1
    kept=m.experiences["step_experiences"][0]
    assert kept["experience_id"]=="step_000001"
    assert kept["utility"]==0.35 and kept["retrieval_count"]==22
    assert kept["created_at_step"]==30


def test_edge_no_semantic_paraphrase_dedup():
    m=ExperienceMemory(load_initial_experiences=False)
    m.experiences["step_experiences"]=[_entry("step_000001","Search the exact relation")]
    m._normalize_all_experience_meta()
    similar_different_text=_entry(None,"Verify the precise relation")
    assert m.upsert_experience_groups(step_experiences=[similar_different_text])==1
    assert len(m.experiences["step_experiences"])==2


def test_t71_write_path_uses_exact_edge_upsert_without_embedding(monkeypatch):
    m=ExperienceMemory(load_initial_experiences=False)
    # If embedding dedup were still called, this test would fail.
    def no_embed(*args,**kwargs):
        raise AssertionError("embedding must not run on admission")
    monkeypatch.setattr(m,"_encode_texts",no_embed)
    exp={"experience_type":"step","task_type":"direct_retrieval",
         "trigger":"A result has an ambiguous date",
         "recommended_action":"Search for the exact date and check event relation",
         "avoid_action":"Do not give a nearby unrelated date",
         "retrieval_text":"An ambiguous date in retrieved results",
         "state_cues":["event date"],"evidence":{"source_traj_uids":["a"]}}
    first=write_new_experiences(m,[exp],step=20)
    second=write_new_experiences(m,[exp],step=25)
    assert first["bank/added"]==1
    assert second["bank/added"]==0
    assert len(m.experiences["step_experiences"])==1
    assert m.experiences["step_experiences"][0]["created_at_step"]==20


def test_edge_eviction_requires_negative_windows_exposures_and_age():
    m=ExperienceMemory(load_initial_experiences=False)
    m.experiences["task_experiences"]=[
        _entry("task_000001","hard_bad",utility=-.30,retrieval_count=10,created=20,neg_windows=2),
        _entry("task_000002","not_enough_bad_windows",utility=-.40,retrieval_count=10,created=20,neg_windows=1),
        _entry("task_000003","young_bad",utility=-.45,retrieval_count=10,created=95,neg_windows=3),
        _entry("task_000004","few_exposures_bad",utility=-.45,retrieval_count=2,created=20,neg_windows=3)]
    m._normalize_all_experience_meta()
    result=m.evict_excess_experiences(
        current_step=100,max_task_experiences=200,max_step_experiences=200,
        protect_recent_steps=20,min_exposures=5,utility_threshold=-.25,
        negative_windows=2)
    assert {x["experience_id"] for x in result["removed"]}=={"task_000001"}
    assert result["removed"][0]["reason"]=="hard_negative_utility"
    assert len(m.experiences["task_experiences"])==3


def test_edge_capacity_independent_pools_and_recent_protection():
    m=ExperienceMemory(load_initial_experiences=False)
    m.experiences["task_experiences"]=[
        _entry("task_000001","lower",utility=-.10,retrieval_count=40,created=0),
        _entry("task_000002","higher",utility=.50,retrieval_count=40,created=0),
        _entry("task_000003","recent",utility=-.90,retrieval_count=40,created=97)]
    m.experiences["step_experiences"]=[
        _entry("step_000001","solo_step",utility=.1,retrieval_count=40,created=0)]
    m._normalize_all_experience_meta()
    result=m.evict_excess_experiences(current_step=100,
        max_task_experiences=2,max_step_experiences=2,protect_recent_steps=20,
        utility_threshold=-.25,min_exposures=5,negative_windows=2)
    assert len(m.experiences["task_experiences"])==2
    assert len(m.experiences["step_experiences"])==1
    assert {x["experience_id"] for x in m.experiences["task_experiences"]}=={
        "task_000002","task_000003"}
    assert [x["reason"] for x in result["removed"]]==["capacity_retain_score"]

def test_utility_window_flattens_all_forks_and_ema_zero_observation():
    from agent_system.tree.core import QuestionEvidence
    e=QuestionEvidence()
    bank={"task_experiences":[{"experience_id":"task_000001","utility":0.0}]}
    # Same question: two Fork deltas; another question: one. All three count equally.
    e.record({("task_000001","q1"):-1.0})
    e.record({("task_000001","q1"):0.0})
    e.record({("task_000001","q2"):0.0})
    updated=e.merge_window(bank,step=5,beta=0.1,management_interval_steps=5)
    assert len(updated)==1
    assert updated[0]["group_count"]==2
    assert updated[0]["fork_count"]==3
    assert abs(updated[0]["delta_window"]-(-1/3))<1e-8
    assert abs(updated[0]["utility"]-(-1/30))<1e-8
    assert updated[0]["consecutive_negative_windows"]==1
    assert e.merge_window(bank,step=10,beta=0.1,management_interval_steps=5)==[]
    assert abs(bank["task_experiences"][0]["utility"]-(-1/30))<1e-8
    # Unobserved and zero-delta windows preserve the negative streak; only delta > 0 resets it.
    assert bank["task_experiences"][0]["evidence"]["consecutive_negative_windows"]==1
    e.record({("task_000001","q3"):0.0})
    e.merge_window(bank,step=15,beta=0.1,management_interval_steps=5)
    assert bank["task_experiences"][0]["evidence"]["consecutive_negative_windows"]==1
    e.record({("task_000001","q4"):-1.0})
    e.merge_window(bank,step=20,beta=0.1,management_interval_steps=5)
    assert bank["task_experiences"][0]["evidence"]["consecutive_negative_windows"]==2
    e.record({("task_000001","q5"):1.0})
    e.merge_window(bank,step=25,beta=0.1,management_interval_steps=5)
    assert bank["task_experiences"][0]["evidence"]["consecutive_negative_windows"]==0


def test_t75_vector_dedup_on_upsert_in_embedding_mode(monkeypatch):
    import numpy as np
    m = ExperienceMemory(load_initial_experiences=False, retrieval_mode="embedding")
    base_pos = _entry("step_000001", "Stop searching when snippet answers question", utility=0.25, retrieval_count=10, created=5)
    base_neg = _entry("step_000002", "Refine query when results are ambiguous", utility=-0.15, retrieval_count=8, created=5)
    m.experiences["step_experiences"] = [base_pos, base_neg]
    m._normalize_all_experience_meta()

    def fake_encode(texts, normalize_embeddings=True):
        out = []
        for t in texts:
            if "Stop searching" in t or "Answer immediately" in t:
                out.append([0.98, 0.198997])
            elif "Refine query" in t or "Add qualifier" in t:
                out.append([0.1, 0.994987])
            else:
                out.append([-1.0, 0.0])
        return np.asarray(out, dtype=np.float32)

    monkeypatch.setattr(m, "_encode_texts", fake_encode)

    # 1. Paraphrase of positive-utility experience (sim >= 0.85): keep old text, do not add new
    dup_pos = _entry(None, "Answer immediately once snippet states the answer")
    dup_pos["principle"] = "New principle that should not overwrite positive experience"
    assert m.upsert_experience_groups(step_experiences=[dup_pos]) == 0
    assert len(m.experiences["step_experiences"]) == 2
    assert m.experiences["step_experiences"][0]["principle"] == "Follow exact relation from results"

    # 2. Paraphrase of negative-utility experience (sim >= 0.85): update old text in-place, keep ID & utility
    dup_neg = _entry(None, "Add qualifier when results are ambiguous")
    dup_neg["principle"] = "Improved principle for negative experience"
    assert m.upsert_experience_groups(step_experiences=[dup_neg]) == 0
    assert len(m.experiences["step_experiences"]) == 2
    assert m.experiences["step_experiences"][1]["experience_id"] == "step_000002"
    assert m.experiences["step_experiences"][1]["principle"] == "Improved principle for negative experience"
    assert m.experiences["step_experiences"][1]["utility"] == -0.15


def test_fork_credits_keep_multiple_forks_from_same_question():
    from types import SimpleNamespace
    from agent_system.tree.core import TreeRollout, SearchBranchState, QuestionEvidence

    tree = TreeRollout()
    cases = [
        ("a", "same question", -1.0, False),
        ("b", "same question", -1.0, False),
        ("c", "different question", 1.0, False),
        ("d", "same question", -1.0, True),
    ]
    for fork_id, question, delta, same_action in cases:
        state = SearchBranchState(
            uid=fork_id, question=question, ground_truth=None, data_source="nq",
        )
        tree.nodes[f"L-{fork_id}"] = SimpleNamespace(state_before=state)
        tree.audits.append({
            "fork_id":fork_id, "left_node_id":f"L-{fork_id}",
            "same_action":same_action, "new_experience_ids":["step_000001"],
            "delta":delta,
        })

    credits = tree.fork_credits()
    assert credits == tree.question_credits()  # Old API stays compatible.
    assert len(credits) == 3  # Same-action fork d excluded.
    assert len({key[1] for key in credits}) == 2
    assert {key[2] for key in credits} == {"a","b","c"}

    evidence = QuestionEvidence()
    evidence.record(credits)
    bank = {"step_experiences":[{"experience_id":"step_000001","utility":0.0}]}
    updates = evidence.merge_window(bank,step=5,beta=0.1,management_interval_steps=5)
    assert len(updates) == 1
    assert updates[0]["group_count"] == 2
    assert updates[0]["fork_count"] == 3
    assert abs(updates[0]["delta_window"]-(-1/3)) < 1e-8
    assert abs(updates[0]["utility"]-(-1/30)) < 1e-8
    assert evidence.statistics("step_000001")["n"] == 2  # Diagnostics only.


def test_t71_negative_cutoff_no_age_guard_min_20_exposures():
    m=ExperienceMemory(load_initial_experiences=False)
    m.experiences["task_experiences"]=[
        _entry("task_000001","eligible_at_threshold",
               utility=-.01,retrieval_count=20,created=100,neg_windows=2),
        _entry("task_000002","not_20_exposures",
               utility=-.9,retrieval_count=19,created=10,neg_windows=4),
        _entry("task_000003","not_negative_enough",
               utility=-.009,retrieval_count=30,created=5,neg_windows=3),
        _entry("task_000004","not_two_negative_windows",
               utility=-.1,retrieval_count=30,created=5,neg_windows=1),
        _entry("task_000005","positive_utility",
               utility=.008,retrieval_count=20,created=100,neg_windows=3),
    ]
    m._normalize_all_experience_meta()
    out=m.evict_excess_experiences(
        current_step=100,max_task_experiences=200,max_step_experiences=200,
        protect_recent_steps=0,min_exposures=20,
        utility_threshold=-.01,negative_windows=2)
    assert [x["experience_id"] for x in out["removed"]]==["task_000001"]
    assert out["removed"][0]["reason"]=="hard_negative_utility"


def test_t71_launch_options_only_no_global_default_change():
    from pathlib import Path
    root=Path(__file__).resolve().parents[1]
    launch=(root/"examples/grpo_trainer/run_t71.sh").read_text()
    assert "env.experience_memory.management.eviction_min_exposures=20" in launch
    assert "env.experience_memory.management.eviction_utility_threshold=-0.01" in launch
    assert "env.experience_memory.management.eviction_protect_recent_steps=0" in launch
    assert "env.tree_experience.protect_recent_steps=0" in launch
    assert "env.tree_experience.task_top_k=4" in launch
    assert "env.tree_experience.step_top_k=6" in launch
    assert "env.experience_memory.top_k=6" in launch
    # Task retrieval has an independent override; must retain four candidates.
    assert "env.experience_memory.task_specific_top_k=4" in launch
    assert "env.experience_memory.management.retrieval_alpha=0.5" in launch
    assert "env.experience_memory.management.retrieval_ucb_c=0.5" in launch
    assert "env.tree_experience.utility_ema_beta=0.1" in launch
    assert "env.tree_experience.utility_update_interval=5" in launch
    assert "env.experience_memory.management.eviction_max_task_experiences=200" in launch
    assert "env.experience_memory.management.eviction_max_step_experiences=200" in launch
    shared=(root/"verl/trainer/config/ppo_trainer.yaml").read_text()
    assert "eviction_utility_threshold: -0.25" in shared
