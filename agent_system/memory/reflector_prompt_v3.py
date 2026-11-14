"""Build an evidence-complete, bank-aware search Reflector prompt.

No arbitrary character cutoff: the API/model's real context limit decides
whether an input is too long. All source actions/observations are preserved.
"""
from __future__ import annotations
import json


def _json(obj):
    return json.dumps(obj, ensure_ascii=False, indent=2, default=str)


def _catalog(current_experiences, examples):
    from agent_system.memory.experience_updater import ExperienceUpdater
    present = {}
    for pool, exp in ExperienceUpdater._iter_existing_experiences(current_experiences):
        eid = str(exp.get("experience_id", ""))
        # Exclude only runtime counters, never action conditions or content.
        fields = ("title", "task_type", "when_to_apply", "trigger",
                  "principle", "recommended_action", "avoid_action",
                  "expected_effect", "state_cues", "retrieval_text", "retrieval_obs")
        present[eid] = {"experience_id": eid, "pool": pool,
                        **{k: exp[k] for k in fields if exp.get(k)}}
    removed = {}
    for ex in examples:
        for role in ("success_turns", "failed_turns"):
            for turn in ex.get(role) or []:
                for kind in ("task", "step"):
                    ids = turn.get(kind + "_experience_ids") or []
                    descriptions = turn.get(kind + "_experiences") or []
                    for i, eid in enumerate(ids):
                        if eid not in present and eid not in removed:
                            removed[eid] = {
                                "experience_id": eid, "pool": "removed_" + kind,
                                "description_from_saved_tree":
                                    descriptions[i] if i < len(descriptions) else "(not available)",
                            }
    return list(present.values()) + list(removed.values())


def _turns(turns):
    return [
        {
            "turn": turn.get("depth"),
            "task_experience_ids": turn.get("task_experience_ids") or [],
            "step_experience_ids": turn.get("step_experience_ids") or [],
            # Full raw text includes complete <think>, <search>/<answer>.
            "action_raw": turn.get("action") or "",
            "tool_observation_full": turn.get("observation") or "",
        }
        for turn in turns or []
    ]


def _example(i, ex):
    return {
        "example_number": i,
        "question": ex.get("task"),
        "batch_question_leaf_success_fraction": ex.get("leaf_success_rate"),
        "fork": {
            "fork_id": ex.get("fork_id"),
            "depth": ex.get("fork_depth"),
            "type": ex.get("fork_tier"),
            "left_subtree_success_fraction": ex.get("left_subtree_rate"),
            "right_subtree_success_fraction": ex.get("right_subtree_rate"),
            "left_subtree_leaf_count": ex.get("left_subtree_leaves"),
            "right_subtree_leaf_count": ex.get("right_subtree_leaves"),
            "left_action_at_same_state": ex.get("left_action"),
            "right_action_at_same_state": ex.get("right_action"),
            "left_new_experience_ids_at_fork": ex.get("left_experience_ids") or [],
            "right_new_experience_ids_at_fork": ex.get("right_experience_ids") or [],
            "observation_before_both_actions": ex.get("fork_observation", ""),
            "shared_history_before_fork": ex.get("common_search_history") or [],
        },
        "successful_full_trajectory": _turns(ex.get("success_turns")),
        "failed_full_trajectory": _turns(ex.get("failed_turns")),
        "additional_experience_interventions_after_fork": {
            "success": ex.get("success_post_fork_experience_count", 0),
            "failure": ex.get("failed_post_fork_experience_count", 0),
        },
    }


def build_reflector_v3_prompt(examples, current_experiences, next_exp_idx, max_new=5):
    """Generate one prompt for 10 full same-state fork cases and the bank."""
    if not examples:
        raise ValueError("no examples")
    if len(examples) > 10:
        raise ValueError("v3 expects at most ten examples")
    records = [_example(i, ex) for i, ex in enumerate(examples, 1)]
    catalog = _catalog(current_experiences, examples)
    instructions = """You write new, practical search experiences for a four-action
question-answering agent. You receive 10 comparisons from a real branching
search tree, plus the entire currently active experience bank.

YOUR TASK
Read EVERY comparison. For each, look for a concrete mistake or a better
searching/answering decision that the visible history could help avoid.
Propose up to 5 DISTINCT, usable experiences across the cases; 0 is okay
if no case supports a useful insight. Do not stop after finding the first
plausible rule. An experience is a proposal to be tested by future training,
not a claim already proven effective. You do not need causal certainty.

EVIDENCE
Each case contains two complete root-to-leaf paths for the same question.
They share EXACTLY the same state at the recorded fork. The fork branches
differ in the experience injected there. Afterward, search results and later
experiences can differ, so the final outcome alone never proves that an
action or injected experience caused success. The subtree success rates are
a rough contrast signal, not independent trials. Read both paths, the
fork state, full observations and actual answers. If a raw tool result
itself reports that material was omitted, treat it as unknown, not absent.
Look for an observable error or a useful decision difference. If an outcome is just query wording,
random retrieval differences, or an arbitrary alternative answer with no
reusable diagnostic lesson, skip that case.

EXPERIENCE TYPE = WHEN IT IS RETRIEVED AND USED
First identify an evidence-supported, reusable piece of advice from
the successful and failed trajectories. Then choose the pool in which
the advice could actually be useful:

- task (T): Retrieved using ONLY the original question, injected from
  the first turn and kept for the whole question. Its trigger,
  state_cues and retrieval_text should describe a pattern recognizable
  from the original question. Give an actionable way to approach that
  question. It CAN be a multi-step plan, but need not be multi-hop.
  You MAY explain how to handle possible future observations in the
  recommended action; do not require them to be already observed.
- step (S): Retrieved using the original question PLUS the latest
  search observation and used for this turn. Its trigger and state_cues
  may describe a concrete situation that has occurred in the search;
  specify a useful next action or when to answer.

Classification is NOT a quality ranking. Do not prefer task over step,
do not require either type, do not impose a T/S ratio, and do not turn
a step-level correction into a generic task plan to produce more T.
A root fork does not automatically imply T, and a multi-step sentence
does not automatically make an experience T. A specific task pattern
can coexist with broader advice; similar advice may still be useful
at different retrieval stages.
Do not require both types. Either type is optional.

EXISTING BANK AND NATURAL REDUNDANCY
Read the bank to avoid merely copying previous experience text. A concrete
T can coexist with a generic T even when they share a two-hop skeleton;
its checkable trigger, named relation to resolve, and failure to avoid
can be useful for retrieval. A task T and a state S may overlap because
they apply at different stages. Do NOT disqualify or change an
experience's T/S type just because another entry is broadly similar.
Exact duplicates are already handled by the existing bank upsert;
no separate semantic-duplicate-review API will be called. Do not force
five experiences and do not deliberately copy an old entry word-for-word.

QUALITY FILTER
- Each rule needs a checkable applicability condition and a concrete
  action that the agent could follow. Avoid empty maxims or merely
  restating the current question as "first find X, then find Y".
- Support the advice with the visible actions and observations; do
  not treat one success/failure contrast as causal proof.
- Use placeholder entities rather than leaking the example's specific
  people, works, places, dates, or final answer.
- Entries must stand alone and never refer to forks or other bank items.
- The generated T/S type determines retrieval timing, not novelty.

OUTPUT
Examine all 10 cases, then return ONLY a JSON array containing 0 to
MAX_NEW_ITEMS entries. Prefer multiple independent, grounded insights
when they exist; do not pad with shallow or speculative rules.
Each item must have:
  experience_id: one supplied candidate ID
  experience_type: "task" or "step"
  task_type: one of general, direct_retrieval, multi_hop_reasoning,
    entity_attribute_lookup, comparison, numeric, temporal,
    source_identification, list_intersection, location_chain, other
  trigger: applicable question pattern (T) or observed situation (S)
  recommended_action: specific plan/action, query pattern and stopping rule
  avoid_action: the concrete mistake to avoid
  expected_effect: intended result, not a proven reward claim
  state_cues: short array of question cues (T) or question+observation cues (S)
  retrieval_text: 1-3 sentences useful for retrieving this exact rule
  source_examples: one-based example numbers that support this rule

Return nothing besides the JSON array.
"""
    allowed_ids = [f"exp_{next_exp_idx + i:06d}" for i in range(max_new)]
    return (
        instructions
        + "\nMAX_NEW_ITEMS: " + str(max_new)
        + "\nCANDIDATE_IDS: " + _json(allowed_ids)
        + "\n\nCOMPLETE EXAMPLES (not truncated by the prompt builder):\n"
        + _json(records)
        + "\n\nEXPERIENCE CATALOG (complete semantic content of current bank; removed"
          " entries are marked):\n"
        + _json(catalog)
        + "\n\nOutput ONLY the JSON array.\n"
    )
