"""End-to-end CPU unit & mathematical equivalence tests for T75 v2 optimizations:
M2 (21-dim [Top-K, Rest] KL), M3 (skip all-correct saturated SFT),
E1 (per-node invalid penalty), E2 (max_mini_batches=6),
E3 (lossless dead-row skipping & entropy bypass), and S1-S4 (prompt & reflector fixes).
"""
from copy import deepcopy
from pathlib import Path
from types import MethodType
import json
import torch
from omegaconf import OmegaConf

from verl import DataProto
from agent_system.memory.experience_memory import ExperienceMemory
from agent_system.memory.experience_updater import ExperienceUpdater
from agent_system.tree.core import (
    DecisionNode,
    ForkRecord,
    Leaf,
    SearchBranchState,
    TreeRollout,
    leaf_advantages,
)
from agent_system.tree.edge_fork_adapter import edge_search_examples_from_forks
from agent_system.tree.loss import tree_topk_loss
from agent_system.tree.question_batches import (
    node_mixed_plan,
    prepare_dedup_batch,
    question_batches,
)
from agent_system.tree.reflector_selection_v3 import select_reflector_forks
from agent_system.tree.runner import build_step_retrieval_query


ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# M2: 21-dim [Top-K, Rest] Exact Partitioned Reverse KL & Memory-Efficient LSE
# ---------------------------------------------------------------------------
def test_m2_tail_bucket_reverse_kl_catches_out_of_topk_peak_and_matches_exact():
    from verl.workers.actor.dp_actor import _memory_efficient_logsumexp

    torch.manual_seed(42)
    B, R, V, K = 2, 3, 50, 5
    full_teacher = torch.randn(B, R, V)
    # Teacher puts strong mass on tokens 0..4
    full_teacher[..., :K] += 4.0
    t_topk_logits, t_topk_ids = torch.topk(full_teacher, k=K, dim=-1)
    t_lse = torch.logsumexp(full_teacher.float(), dim=-1)

    # Construct Student where within-top-K relative logits match Teacher up to a constant shift,
    # BUT Student puts a massive wrong peak on token K+3 (outside Teacher Top-K)!
    full_student = full_teacher.clone() - 3.0
    full_student[..., K + 3] = 8.0
    full_student = full_student.detach().requires_grad_(True)

    s_topk_logits = full_student.gather(dim=-1, index=t_topk_ids)
    s_lse = _memory_efficient_logsumexp(full_student.to(torch.bfloat16))
    mask = torch.ones(B, R)
    weights = torch.tensor([1.0, 0.5])

    # Legacy 20-dim truncated KL is blind to the constant shift & out-of-top-K wrong peak:
    legacy_loss, legacy_kl, _ = tree_topk_loss(
        s_topk_logits.detach().clone().requires_grad_(True),
        t_topk_logits,
        mask,
        weights,
        mask,
    )
    assert float(legacy_kl.detach().max()) < 1e-6

    # New (K+1)-bucket KL strongly detects the out-of-top-K wrong peak and pushes it down!
    loss_21, kl_21, closed_grad = tree_topk_loss(
        s_topk_logits,
        t_topk_logits,
        mask,
        weights,
        mask,
        student_lse=s_lse,
        teacher_lse=t_lse,
    )
    assert float(kl_21.detach().min()) > 1.5
    loss_21.backward()
    # Wrong peak token K+3 outside top-K must get a positive gradient (pushed DOWN by SGD/Adam)
    assert (full_student.grad[..., K + 3] > 0.002).all()
    # Teacher top-K tokens 0..K-1 must get negative gradient (pulled UP)
    assert (full_student.grad[..., :K].sum(-1) < -0.002).all()

    # Verify closed-form top-K gradient matches autograd wrt independent top-K logits + exact full-vocab partition KL
    full_s_fp32 = torch.randn(B, R, V, requires_grad=True)
    p_s = torch.softmax(full_s_fp32, dim=-1)
    p_t = torch.softmax(full_teacher, dim=-1)
    p_s_top = p_s.gather(-1, t_topk_ids)
    p_t_top = p_t.gather(-1, t_topk_ids)
    p_s_rest = 1.0 - p_s_top.sum(-1)
    p_t_rest = 1.0 - p_t_top.sum(-1)
    exact_part_kl = (
        (p_s_top * (p_s_top.log() - p_t_top.log())).sum(-1)
        + p_s_rest * (p_s_rest.log() - p_t_rest.log())
    )
    _, our_kl, our_grad = tree_topk_loss(
        full_s_fp32.gather(-1, t_topk_ids),
        t_topk_logits,
        mask,
        weights,
        mask,
        student_lse=torch.logsumexp(full_s_fp32, dim=-1),
        teacher_lse=t_lse,
    )
    torch.testing.assert_close(our_kl, exact_part_kl, rtol=1e-5, atol=1e-6)
    auto_g = torch.autograd.grad(our_kl.sum(), full_s_fp32)[0].gather(-1, t_topk_ids)
    # Note: full-vocab autograd wrt z_i (i in K) equals our_grad because d(p_b)/d(z_i) = p_i*(delta_{ib} - p_b)
    torch.testing.assert_close(our_grad, auto_g, rtol=1e-5, atol=1e-6)


# ---------------------------------------------------------------------------
# M3 + E1: Skip SFT Only on All-Correct Questions & Per-Node Invalid Penalty
# ---------------------------------------------------------------------------
def _mk_node(nid, uid, depth, branch, state, action, obs="", done=True, valid=True, source="tree", new_exp=()):
    return DecisionNode(
        uid=uid,
        node_id=nid,
        parent_id=None,
        fork_id=f"fork_{nid}",
        depth=depth,
        branch=branch,
        state_before=state,
        new_experience_ids=tuple(new_exp),
        actual_prompt_text="prompt",
        noexp_prompt_text="prompt",
        action=action,
        observation=obs,
        done=done,
        valid=valid,
        source=source,
    )


def test_m3_and_e1_sft_filters_only_all_correct_and_penalty_is_per_node():
    tree = TreeRollout(invalid_action_penalty_coef=0.5)
    # Question q_sat: 4/4 all-correct natural trajectories -> SFT must be 0.0
    s_sat = SearchBranchState("q_sat", "Easy Q?", ["A"], "nq")
    for idx in range(4):
        nid = f"sat_n{idx}"
        tree.nodes[nid] = _mk_node(nid, "q_sat", 1, "L", s_sat, "<answer>A</answer>", done=True, valid=True, source="natural")
        tree.leaves.append(Leaf("q_sat", f"sat_l{idx}", (nid,), 1.0, "nq", source="natural"))

    # Question q_partial (2/4 natural) + multi-turn invalid final node on failing trajectory
    s_part = SearchBranchState("q_part", "Medium Q?", ["B"], "nq")
    tree.nodes["p_ok1"] = _mk_node("p_ok1", "q_part", 1, "L", s_part, "<search>q</search>", "obs", False, True, source="natural")
    tree.nodes["p_ok2"] = _mk_node("p_ok2", "q_part", 2, "L", s_part, "<answer>B</answer>", "", True, True, source="natural")
    tree.leaves.append(Leaf("q_part", "part_win_0", ("p_ok1", "p_ok2"), 1.0, "nq", source="natural"))
    tree.leaves.append(Leaf("q_part", "part_win_1", ("p_ok1", "p_ok2"), 1.0, "nq", source="natural"))

    # Failing trajectory where Turn 1 <search> is VALID, Turn 2 is INVALID (truncated)
    tree.nodes["p_fail_t1_valid"] = _mk_node("p_fail_t1_valid", "q_part", 1, "L", s_part, "<search>good</search>", "obs", False, True, source="natural")
    tree.nodes["p_fail_t2_invalid"] = _mk_node("p_fail_t2_invalid", "q_part", 2, "L", s_part, "<think>cutoff", "", True, False, source="natural")
    tree.leaves.append(Leaf("q_part", "part_lose_0", ("p_fail_t1_valid", "p_fail_t2_invalid"), 0.0, "nq", source="natural"))
    tree.leaves.append(Leaf("q_part", "part_lose_1", ("p_fail_t1_valid",), 0.0, "nq", source="natural"))

    # Question q_rescue (0/4 natural all 0, rescued by 1 tree leaf = 1)
    s_res = SearchBranchState("q_res", "Hard Q?", ["C"], "hotpotqa")
    for idx in range(4):
        nid = f"res_nat_{idx}"
        tree.nodes[nid] = _mk_node(nid, "q_res", 1, "L", s_res, "<answer>wrong</answer>", "", True, True, source="natural")
        tree.leaves.append(Leaf("q_res", f"res_nat_l{idx}", (nid,), 0.0, "hotpotqa", source="natural"))
    tree.nodes["res_tree_win"] = _mk_node("res_tree_win", "q_res", 1, "R", s_res, "<answer>C</answer>", "", True, True, source="tree")
    tree.leaves.append(Leaf("q_res", "res_tree_l0", ("res_tree_win",), 1.0, "hotpotqa", source="tree"))

    base_adv = leaf_advantages(tree.leaves)
    rows = tree.expanded_rows()
    by_row = {(r["leaf_id"], r["node_id"]): r for r in rows}

    # M3 checks:
    # 1) 4/4 all-correct saturated question gets sft_weight == 0.0
    for idx in range(4):
        assert by_row[(f"sat_l{idx}", f"sat_n{idx}")]["sft_weight"] == 0.0
    # 2) 2/4 natural winner nodes KEEP sft_weight == 1.0
    assert by_row[("part_win_0", "p_ok1")]["sft_weight"] == 1.0
    assert by_row[("part_win_0", "p_ok2")]["sft_weight"] == 1.0
    # 3) 0/4 tree-rescued winner node KEEPS sft_weight == 1.0
    assert by_row[("res_tree_l0", "res_tree_win")]["sft_weight"] == 1.0

    # E1 checks:
    # Valid prefix turn 1 is NOT penalized, invalid turn 2 IS penalized by -0.5
    assert abs(by_row[("part_lose_0", "p_fail_t1_valid")]["advantage"] - base_adv["part_lose_0"]) < 1e-7
    assert abs(by_row[("part_lose_0", "p_fail_t2_invalid")]["advantage"] - (base_adv["part_lose_0"] - 0.5)) < 1e-7


# ---------------------------------------------------------------------------
# E2: Converge 6,000+ Expanded Rows into 6 Minibatches (Not 24)
# ---------------------------------------------------------------------------
def test_e2_question_batches_and_node_mixed_cap_minibatches():
    rows = []
    for q in range(256):
        for t in range(24):
            rows.append(dict(
                uid=f"q{q}",
                node_id=f"q{q}:n{t % 6}",
                advantage=1.0 if t % 2 == 0 else -1.0,
                kd_weight=0.5 if t == 0 and q < 50 else 0.0,
                sft_weight=1.0 if t % 2 == 0 else 0.0,
            ))
    assert len(rows) == 6144
    # Legacy default (max_mini_batches=None) preserves 24 bins
    assert len(question_batches(rows, batch_size=256)) == 24
    # Capped mode produces 6 balanced minibatches
    bins_6 = question_batches(rows, batch_size=256, max_mini_batches=6)
    assert len(bins_6) == 6
    assert sum(len(b) for b in bins_6) == len(rows)

    plan_6 = node_mixed_plan(rows, [512] * len(rows), world_size=2, batch_size=256, max_mini_batches=6)
    assert len(plan_6) == 6
    # Tiny 5-row input still produces 1 minibatch (not 6 empty ones)
    tiny_bins = question_batches(rows[:5], batch_size=256, max_mini_batches=6)
    assert len(tiny_bins) == 1


# ---------------------------------------------------------------------------
# E3: 100% Gradient & Metric Equivalence Under Dead-Row Skipping
# ---------------------------------------------------------------------------
def test_e3_dead_row_skipping_exact_gradient_and_metric_equivalence(monkeypatch):
    from types import SimpleNamespace
    import verl.workers.actor.dp_actor as dp_mod
    import verl.utils.torch_functional as tf_mod
    import verl.utils.debug.performance as performance

    monkeypatch.setattr(tf_mod, "FLAH_ATTN_CROSS_ENTROPY_LOSS_AVAILABLE", False)
    monkeypatch.setattr(performance, "_get_current_mem_info", lambda: (0, 0, 0, 0))
    monkeypatch.setattr(dp_mod, "get_torch_device", lambda: SimpleNamespace(current_device=lambda: "cpu"))
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 1)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 0)
    monkeypatch.setattr(
        torch.distributed,
        "all_reduce",
        lambda tensor, op=None: None,
    )
    monkeypatch.setattr(
        torch.distributed,
        "all_gather",
        lambda outputs, value: outputs[0].copy_(value),
    )
    monkeypatch.setattr(
        torch.distributed,
        "all_gather_object",
        lambda outputs, obj: outputs.__setitem__(0, obj),
    )

    from contextlib import nullcontext

    class TinyCausalModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.emb = torch.nn.Embedding(16, 8)
            self.head = torch.nn.Linear(8, 16, bias=False)

        def no_sync(self):
            return nullcontext()

        def forward(self, input_ids, attention_mask=None, position_ids=None, use_cache=False, **kwargs):
            h = self.emb(input_ids)
            logits = self.head(h)
            return type("Out", (), {"logits": logits})()

    torch.manual_seed(2026)
    base_model = TinyCausalModel()

    # Build a 10-row dedup tree batch where 6 rows are all-zero dead rows (60% dead!)
    # and 4 rows have active PG, OPSD, and/or SFT signals.
    B, P, R = 10, 3, 4
    input_ids = torch.randint(1, 15, (B, P + R))
    sft_inp = input_ids.clone()
    sft_inp[1, 0] = 2  # changed prompt for SFT
    teacher_inp = input_ids.clone()
    teacher_inp[2, 0] = 3
    kd_stu_inp = input_ids.clone()
    kd_stu_inp[2, 0] = 4

    batch_tensors = dict(
        responses=input_ids[:, -R:],
        input_ids=input_ids,
        attention_mask=torch.ones(B, P + R, dtype=torch.long),
        position_ids=torch.arange(P + R).expand(B, -1),
        w_tok=torch.tensor([2., 1., 1., 1., 2., 3., 1., 2., 1., 0.]),
        positive_adv_count=torch.tensor([2., 1., 0., 0., 0., 0., 0., 0., 0., 0.]),
        negative_adv_count=torch.tensor([0., 0., 1., 1., 0., 0., 0., 0., 0., 0.]),
        A_pos=torch.tensor([1.5, 0.8, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        A_neg=torch.tensor([0.0, 0.0, -1.2, -0.7, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        advantages=torch.zeros(B, R),
        edge_teacher_input_ids=teacher_inp,
        edge_teacher_attention_mask=torch.ones(B, P + R, dtype=torch.long),
        edge_teacher_position_ids=torch.arange(P + R).expand(B, -1),
        edge_distill_weight=torch.tensor([0., 0., 0.6, 0., 0., 0., 0., 0., 0., 0.]),
        kd_weight=torch.tensor([0., 0., 0.6, 0., 0., 0., 0., 0., 0., 0.]),
        kd_student_input_ids=kd_stu_inp,
        kd_student_attention_mask=torch.ones(B, P + R, dtype=torch.long),
        kd_student_position_ids=torch.arange(P + R).expand(B, -1),
        kd_canonical=torch.tensor([False, False, True, False, False, False, False, False, False, False]),
        kd_path_has_T=torch.tensor([False, False, True, False, False, False, False, False, False, False]),
        sft_weight=torch.tensor([1., 1., 0., 0., 0., 0., 0., 0., 0., 0.]),
        sft_student_input_ids=sft_inp,
        sft_student_attention_mask=torch.ones(B, P + R, dtype=torch.long),
        sft_student_position_ids=torch.arange(P + R).expand(B, -1),
        is_padding_copy=torch.tensor([False] * 9 + [True]),
        dedup_mini=torch.zeros(B, dtype=torch.long),
        dedup_mini_weight=torch.full((B,), 14.0),
    )

    def run_mode(skip_dead: str):
        monkeypatch.setenv("T75_SKIP_DEAD_ROWS", skip_dead)
        model = deepcopy(base_model)
        actor = object.__new__(dp_mod.DataParallelPPOActor)
        actor.actor_module = model
        actor.actor_optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        actor._explicit_replica_sync = True
        actor._fsdp_sync_optional_forwards = False
        actor.use_remove_padding = False
        actor.response_only_logits = False
        actor.use_fused_kernels = False
        actor.use_ulysses_sp = False
        actor.ulysses_sequence_parallel_size = 1
        actor.device_name = "cpu"
        actor.compute_entropy_from_logits = dp_mod.verl_F.entropy_from_logits
        actor.config = OmegaConf.create(dict(
            edge_distillation=dict(enable=True, coef=0.05, token_selection="teacher_topk", top_k=4, path_tier=2),
            success_sft=dict(enable=True, coef=0.05),
            ppo_mini_batch_size=10,
            ppo_micro_batch_size_per_gpu=2,
            use_dynamic_bsz=False,
            use_kl_loss=False,
            ppo_epochs=1,
            clip_ratio=0.2,
            clip_ratio_low=0.2,
            clip_ratio_high=0.2,
            clip_ratio_c=3.0,
            grad_clip=1.0,
            entropy_coeff=0.0,
            loss_agg_mode="token-mean",
            policy_loss=dict(loss_mode="vanilla"),
        ))

        forward_rows = []
        orig_fwd = actor._forward_micro_batch

        def wrapped_fwd(self, micro_batch, *args, **kwargs):
            forward_rows.append(int(micro_batch["responses"].shape[0]))
            return orig_fwd(micro_batch, *args, **kwargs)

        actor._forward_micro_batch = MethodType(wrapped_fwd, actor)

        lp_proto = DataProto.from_dict(
            tensors=deepcopy(batch_tensors),
            meta_info=dict(
                micro_batch_size=2,
                temperature=1.0,
                use_dynamic_bsz=False,
                tree_mode=True,
                dedup_shared_prefix=True,
            ),
        )
        old_lp, ent = actor.compute_log_prob(lp_proto, calculate_entropy=False)
        assert ent is None

        upd_tensors = deepcopy(batch_tensors)
        upd_tensors["old_log_probs"] = old_lp
        upd_proto = DataProto.from_dict(
            tensors=upd_tensors,
            meta_info=dict(temperature=1.0, tree_mode=True, dedup_shared_prefix=True),
        )
        metrics = actor.update_policy(upd_proto)
        weights_after = torch.cat([p.detach().flatten() for p in model.parameters()])
        return weights_after, metrics, sum(forward_rows), old_lp

    w_full, m_full, rows_full, lp_full = run_mode("0")
    w_skip, m_skip, rows_skip, lp_skip = run_mode("1")

    # Active rows (0..3) must have identical old_log_probs
    torch.testing.assert_close(lp_skip[:4], lp_full[:4], rtol=1e-6, atol=1e-6)
    # Forwarded rows must drop substantially (from 10+10+3 = 23 down to 4+4+3 = 11!)
    assert rows_skip < rows_full * 0.55
    # Model parameters after AdamW step must be 100% identical!
    torch.testing.assert_close(w_skip, w_full, rtol=1e-5, atol=1e-6)
    # Key aggregated metrics must match identically!
    for key in (
        "actor/expanded_tokens",
        "actor/pg_loss_expanded",
        "kd/active_tokens",
        "kd/weighted_loss_active_tokens",
        "kd/weighted_loss_all_tokens",
        "kd/student_mass",
        "sft/active_tokens",
        "sft/loss_token_mean",
    ):
        assert abs(float(m_skip[key]) - float(m_full[key])) < 1e-6, f"Mismatch on {key}: {m_skip[key]} vs {m_full[key]}"


# ---------------------------------------------------------------------------
# S1-S4: Experience Formatting, Multi-Doc Excerpt, & Reflector Leak Gate
# ---------------------------------------------------------------------------
def test_s1_s2_s3_s4_prompt_slimming_multidoc_query_and_v3_leak_gate():
    # S1: No "Avoid Do not", no "..", no duplicated trigger, compact <90 words
    mem = ExperienceMemory(load_initial_experiences=False, retrieval_mode="template")
    raw_reflector_exp = {
        "experience_id": "exp_000100",
        "experience_type": "step",
        "task_type": "entity_attribute_lookup",
        "trigger": "When multiple retrieved snippets share the same film or book title across different release years.",
        "recommended_action": "Include both the exact release year and director relation in the next search query.",
        "avoid_action": "Do not issue broad title-only searches or guess from the wrong adaptation.",
        "expected_effect": "This isolates the exact adaptation and prevents entity drift across remakes.",
        "state_cues": ["same title across years.", "missing target year."],
        "retrieval_text": "Multiple candidate films share a title across different release years.",
    }
    converted = mem._legacy_experience_to_step_experience(raw_reflector_exp, 1)
    assert "Avoid Do not" not in converted["principle"]
    assert ".." not in converted["principle"]
    assert "; " in converted["when_to_apply"]

    rendered = mem.format_for_prompt({"task_experiences": [], "step_experiences": [converted]})
    assert "Avoid Do not" not in rendered
    assert ".." not in rendered
    assert rendered.count("same film or book title") == 1
    assert "prevents entity drift" not in rendered  # verbose rationale stripped from 3B prompt
    assert len(rendered.split()) < 90

    # S2: Doc 3 answer is preserved & 9-word focused query is NOT mislabeled as unfocused
    state = SearchBranchState(
        "q_mummy",
        'Who directed the 1932 film "The Mummy" starring Boris Karloff?',
        ["Karl Freund"],
        "nq",
    )
    state.turns = 1
    state.history = [(
        '<search>director of "The Mummy" 1932 film Boris Karloff</search>',  # 9 words!
        "",
    )]
    doc1 = "The Mummy is a 1999 American action-adventure film written and directed by Stephen Sommers starring Brendan Fraser and Rachel Weisz in Egypt."
    doc2 = "The Mummy franchise includes multiple universal monsters adaptations, novelizations, spin-off sequels, and modern reboots produced over decades."
    doc3 = "The Mummy (1932) is an American pre-Code horror film directed by Karl Freund and starring Boris Karloff as Imhotep."
    state.last_observation = (
        f"<information>Doc 1(Title: The Mummy (1999 film)) {doc1}\n"
        f"Doc 2(Title: The Mummy (franchise)) {doc2}\n"
        f"Doc 3(Title: The Mummy (1932 film)) {doc3}</information>"
    )
    q_summary = build_step_retrieval_query(state, max_steps=4)
    assert "Karl Freund" in q_summary
    assert "overly long multi-clause query" not in q_summary

    # S3 + S4: reflector_version="v3" preserved through edge_fork_adapter -> blocks leaked answer & strips imperative prefix
    tree = TreeRollout()
    s0 = SearchBranchState("q_leak", "Who directed The Mummy (1932)?", ["Karl Freund"], "nq")
    tree.nodes["L"] = _mk_node("L", "q_leak", 1, "L", s0, "<answer>Karl Freund</answer>", "", True, True)
    tree.nodes["R"] = _mk_node("R", "q_leak", 1, "R", s0, "<answer>Stephen Sommers</answer>", "", True, True, new_exp=["task_000002"])
    tree.forks["F"] = ForkRecord("F", "q_leak", 1, "L", "R", ("task_000002",), (0.0,))
    tree.leaves = [
        Leaf("q_leak", "L0", ("L",), 1.0, "nq"),
        Leaf("q_leak", "R0", ("R",), 0.0, "nq"),
    ]
    tree.evaluate_forks()
    selected = select_reflector_forks(tree, k=4, seed=0)
    adapted = edge_search_examples_from_forks(selected)
    assert adapted[0].get("reflector_version") == "v3"

    updater = object.__new__(ExperienceUpdater)
    leaked_item = {
        "experience_type": "T",
        "task_type": "direct_retrieval",
        "trigger": "When asked about Karl Freund films",
        "recommended_action": "Search exact year",
        "avoid_action": "Guessing",
        "retrieval_text": "Use this strategy when asked about a film director.",
        "state_cues": ["film"],
        "source_examples": [1],
    }
    clean_item = {
        "experience_type": "T",  # alias normalized to "task" under v3!
        "task_type": "direct_retrieval",
        "trigger": "When asked for the director of a classic film with later remakes",
        "recommended_action": "Include the release year alongside the title",
        "avoid_action": "Searching the bare title alone",
        "retrieval_text": "Use this strategy when a question asks for the director of a film with remakes.",
        "state_cues": ["film remake"],
        "source_examples": [1],
    }
    kept = updater._postprocess_search_experiences([leaked_item, clean_item], adapted)
    assert updater.last_answer_leak_dropped == 1
    assert len(kept) == 1
    assert kept[0]["experience_type"] == "task"
    assert not kept[0]["retrieval_text"].lower().startswith("use this strategy")

    # S4 seed bank check: task_000001 removed
    bank = json.loads((ROOT / "memory_data/search/state_aware_experiences_v2.json").read_text(encoding="utf-8"))
    ids = {x["experience_id"] for x in bank["task_experiences"]}
    assert "task_000001" not in ids
    assert len(ids) == 10

