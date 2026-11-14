"""T71 two-stage experience reflection: unchanged eligible fork filter -> reflect -> distill.

Stage-one compares complete observed trajectories in pairs of two to avoid
large prompt timeouts. Stage-two produces up to 5 distinct, reusable items.
Production code uses the exact stage-two prompt validated offline with gpt-6.1-sol.
"""
import json
import re
from typing import Dict, List

# Injected verbatim from the offline tested tools/t71_stage2_max5_probe.py
STAGE2 = """You are the experience distillation stage of a multi-turn search agent.

INPUT: Stage-one reflections from a set of real search-trajectory Forks.
Stage one has already compared the traces; DO NOT re-analyze whole traces.
Your job is to retain a SMALL SET of distinct, high-value, actionable
SEARCH BEHAVIORS from the collection, not to write one rule per example.

First mentally collect all concrete transferable strategies from records
with status == 'valid'. All other records are INELIGIBLE; never cite them.
Next identify overlapping insights: if two reflections recommend the same
WHEN-to-apply and DO-next-action, they represent ONE strategy, even if
their example entities or wordings differ. Consolidate those insights
without losing a useful trigger or concrete next action. Do not combine
different search actions into a vague generality just to save entries.
Finally select at most FIVE independently useful strategies across the
WHOLE batch. Prioritize by (a) recognizable trigger, (b) concrete action
which can change agent decisions, (c) cross-question usefulness,
(d) non-redundancy among selected strategies.
A case does NOT deserve an experience just because it is marked valid.
Some reflections may be thin, overlapping, or offer no transferable
decision. Skip them. ZERO to FIVE is allowed; FIVE IS A HARD MAXIMUM,
NOT A TARGET. You are never required to cover every valid Fork.
Prefer fewer strong strategies over a padded list of shallow ones.

EVERY output experience must include:
- WHEN: a checkable applicability condition, not the source question.
- DO: a specific search / evidence validation / answer decision.
- AVOID: the concrete failure mode.
- An effective retrieval_text of 1-3 standalone sentences.
- source_fork_ids: the valid Fork IDs truly supporting that strategy.
If multiple Forks support the SAME merged strategy, list all their IDs.
Do not attach an unrelated ID just to improve apparent support.

Experience types describe RETRIEVAL TIMING, not quality:
- 'task': recognized from the original question alone, useful at start.
- 'step': requires the original question AND current tool observation,
  useful as the next action. Do not use future knowledge as a step cue.
Neither pool is mandatory. Do not force a T/S balance.

Never write a generic rule like 'search carefully', 'verify your answer',
or 'make more queries'. Never copy source-specific named answers, dates,
or people. Never invent guidance unsupported by the given reflections.

SELF-CHECK BEFORE RETURNING:
1. Every item has an actionable WHEN/DO/AVOID and is not redundant.
2. Each cited source_fork_id belongs to an input with status='valid'.
3. Number of entries is between 0 and 5 inclusive.
4. If there are zero valid insights, output [] exactly.
Return ONLY a JSON ARRAY, no markdown and no explanatory prose.
Each object must have EXACTLY these fields:
{"experience_type":"task|step",
"trigger":"recognizable question pattern (task) or situation (step)",
"recommended_action":"specific behavior to do",
"avoid_action":"specific behavior to avoid",
"state_cues":["observable conditions"],
"retrieval_text":"independent retrieval-focused instructions",
"source_fork_ids":["exact valid input fork IDs"]}."""

NOVELTY_SECTION = """NOVELTY AGAINST THE EXISTING BANK (REQUIRED):
The provided EXISTING_EXPERIENCE_BANK is already installed. Your task is to
propose genuinely ADDITIONAL search decision behaviors, not paraphrases or
narrower examples of strategies already covered. Before outputting EACH item:
compare it against all existing experiences based on observable WHEN, executable
DO, and concrete AVOID, not lexical or topic similarity.
If an old experience already instructs the agent to make essentially the same
decision in the same recognized situation, OMIT the candidate even when the
source question, entity type, date, or wording is different.
Do not split a general old rule into multiple mere examples (e.g. entity->attribute,
movie->release date), and do not repeat an injected experience merely because
one Fork succeeded and the other failed.
A new item is permitted when it adds a distinct, checkable decision condition,
query transformation, evidence-disambiguation criterion, or stop/answer action
not already operationally specified in the old bank. Different decisions in
overlapping topics are allowed. Do not reject solely because topics match.
Only return new additive items, do NOT edit or reword existing experiences.
Keep the exact original JSON output schema and cite only valid input Fork IDs.
ZERO is preferred to generic or duplicate content.
"""

STAGE1 = """You are an evidence-oriented analyst of a multi-turn search QA agent.
Input: exactly TWO real SAME-STATE branching search trajectory contrasts.
At each fork, the agent tried two different actions with different experience
injections; one shown continuation succeeded and one failed.

Your sole goal is REFLECTION: find a specific observable behavior from the
successful path that offers a potentially useful decision strategy, IF one
exists. This is NOT a causal evaluation of injected old experiences.
Do NOT write final generalized experience entries or edit old experiences.

For EACH Fork:
1. Explain which information was available and still missing at the fork.
2. Contrast what the successful and failed paths actually did, citing the
   observed queries, answers, search snippets and depth.
3. Describe a concrete, plausible success strategy or a failure mode.
4. Write a candidate reusable_insight only when the trajectories suggest a
   specific search decision, entity relation, evidence check, stopping rule,
   or answer-formatting behavior that can transfer to new questions.
5. If no actionable insight is visible, mark status='insufficient' and set
   reusable_insight=null. Skipping cases is expected; do not invent lessons.

Do not mechanically accept reward as proof of a causal strategy. Search
results and later injected experiences may differ: treat them as uncertainty
rather than assuming one earlier action necessarily caused the outcome.
This stage does not decide how many final experiences to store.

Return ONLY valid JSON: exactly TWO objects, matching the input fork_ids.
Each object must have this schema:
{"fork_id":"...","status":"valid|insufficient",
"decision_state":"...","observed_difference":"...",
"success_mechanism":"...","failure_pattern":"...",
"reusable_insight":"... or null",
"evidence":[{"side":"success|failure","depth":1,"action":"faithful excerpt",
"observation":"faithful observation snippet or summary"}],
"uncertainty":"..."}
Do not omit either Fork. No explanations outside JSON."""

def _json_array(raw):
    s=(raw or "").strip()
    if s.startswith("```"):
        s=s.split("\n",1)[1].rsplit("```",1)[0].strip()
    try:
        value=json.loads(s)
    except json.JSONDecodeError:
        start,end=s.find("["),s.rfind("]")
        if start<0 or end<start:
            raise
        value=json.loads(s[start:end+1])
    if not isinstance(value,list):
        raise ValueError("Expected JSON array")
    return value

def _record(e):
    def turns(name):
        return [
            {"depth":x.get("depth"),
             "action":x.get("action") or "",
             "tool_observation_full":x.get("observation") or "",
             "task_experience_ids":x.get("task_experience_ids") or [],
             "step_experience_ids":x.get("step_experience_ids") or [],
             "task_experience_content":x.get("task_experiences") or [],
             "step_experience_content":x.get("step_experiences") or []}
            for x in e.get(name,[])
        ]
    return {
        "fork_id":e["fork_id"],
        "question":e.get("task",""),
        "ground_truth":e.get("ground_truth"),
        "fork_depth":e.get("fork_depth"),
        "common_search_history_before_fork":e.get("common_search_history"),
        "observation_before_fork":e.get("fork_observation"),
        "left_action_at_fork":e.get("left_action"),
        "right_action_at_fork":e.get("right_action"),
        "left_experience_ids_at_fork":e.get("left_experience_ids",[]),
        "right_experience_ids_at_fork":e.get("right_experience_ids",[]),
        "left_subtree_success_rate":e.get("left_subtree_rate"),
        "right_subtree_success_rate":e.get("right_subtree_rate"),
        "success_full_path":turns("success_turns"),
        "failure_full_path":turns("failed_turns"),
        "later_experience_injections":{
            "success":e.get("success_post_fork_experience_count",0),
            "failure":e.get("failed_post_fork_experience_count",0)},
    }

def _existing_bank_action_index(bank: Dict) -> List[Dict]:
    """Compact action index sent only to stage two for novelty evaluation.

    Preserve the full active bank, including legacy entries whose action is
    stored inside the principle field instead of recommended_action.
    """
    index = []
    for pool in ("task_experiences", "step_experiences"):
        for experience in (bank or {}).get(pool, []):
            index.append({
                "id": experience.get("experience_id"),
                "experience_type": "task" if pool == "task_experiences" else "step",
                "when": (
                    experience.get("trigger")
                    or experience.get("when_to_apply")
                    or experience.get("title")
                ),
                "action": (
                    experience.get("recommended_action")
                    or experience.get("principle")
                ),
                "avoid": experience.get("avoid_action") or "See principle",
                "retrieval_hint": experience.get("retrieval_obs"),
            })
    return index


def _request(updater,prompt,max_tokens):
    kwargs={"model":updater.model,
            "messages":[{"role":"user","content":prompt}],
            updater.token_limit_param:max_tokens}
    response=updater._completion_with_deadline(kwargs)
    return _json_array(response.choices[0].message.content or "")

def analyze_forks_two_stage(updater, examples:List[Dict], bank:Dict) -> List[Dict]:
    """Return bank-ready experiences. Zero on any incomplete or malformed API result.

    No experiences are inserted here; the training loop handles persistence.
    """
    updater.last_update_succeeded=False
    updater.last_failure_reason=None
    updater.last_two_stage_counts={"reflections":0,"valid":0,"extracted":0}
    examples=list(examples or [])[:10]
    if not examples:
        updater.last_update_succeeded=True
        return []
    try:
        reflections=[]
        seen=set()
        for start in range(0,len(examples),2):
            group=examples[start:start+2]
            # Stage 1 must process the last odd Fork too.
            instruction=STAGE1
            if len(group)==1:
                instruction=instruction.replace("exactly TWO", "exactly ONE").replace(
                    "exactly TWO objects", "exactly ONE object").replace(
                    "Do not omit either Fork.", "Do not omit the Fork.")
            prompt=instruction+"\n\nCASES:\n"+json.dumps([_record(e) for e in group],ensure_ascii=False)
            output=_request(updater,prompt,max(6144,int(updater.max_completion_tokens)))
            expected=[str(e["fork_id"]) for e in group]
            if [str(x.get("fork_id")) for x in output]!=expected:
                raise ValueError("Stage-one response does not match input Fork IDs")
            for item in output:
                if item.get("status") not in ("valid","insufficient"):
                    raise ValueError("Stage-one invalid status")
                if item["fork_id"] in seen:
                    raise ValueError("Stage-one duplicate fork")
                seen.add(item["fork_id"])
            reflections.extend(output)
        updater.last_two_stage_counts["reflections"]=len(reflections)
        updater.last_two_stage_counts["valid"]=sum(x.get("status")=="valid" for x in reflections)
        if not updater.last_two_stage_counts["valid"]:
            updater.last_update_succeeded=True
            return []

        stage2 = (
            STAGE2.replace(
                "SELF-CHECK BEFORE RETURNING:",
                NOVELTY_SECTION + "\nSELF-CHECK BEFORE RETURNING:",
            )
            + "\n\nEXISTING_EXPERIENCE_BANK:\n"
            + json.dumps(_existing_bank_action_index(bank), ensure_ascii=False)
            + "\n\nSTAGE_ONE_REFLECTIONS:\n"
            + json.dumps(reflections, ensure_ascii=False)
        )
        raw=_request(updater,stage2,max(7500,int(updater.max_completion_tokens)))
        if len(raw)>updater.max_new_experiences_per_update:
            raise ValueError("Stage two exceeded maximum experience count")
        valid_ids={str(r["fork_id"]) for r in reflections if r.get("status")=="valid"}
        index_by_id={str(e["fork_id"]):i+1 for i,e in enumerate(examples)}
        intermediate=[]
        for item in raw:
            if not isinstance(item,dict):
                raise ValueError("Stage two non-object item")
            cited=item.get("source_fork_ids")
            if not isinstance(cited,list) or not cited or any(str(x) not in valid_ids for x in cited):
                raise ValueError("Stage two cited missing or invalid Fork")
            if item.get("experience_type") not in ("task","step"):
                raise ValueError("Stage two invalid experience type")
            for field in ("trigger","recommended_action","avoid_action","retrieval_text"):
                if not isinstance(item.get(field),str) or not item[field].strip():
                    raise ValueError("Stage two blank required field: "+field)
            exp={
                "experience_type":item["experience_type"],
                "task_type":"unknown",  # let existing search postprocessor infer the task type
                "trigger":item["trigger"],
                "recommended_action":item["recommended_action"],
                "avoid_action":item["avoid_action"],
                "state_cues":item.get("state_cues",[]),
                "retrieval_text":item["retrieval_text"],
                "source_examples":[index_by_id[str(x)] for x in cited],
                "expected_effect":"Improve search decisions and evidence-supported answers.",
            }
            intermediate.append(exp)
        cleaned=updater._postprocess_search_experiences(intermediate,examples)
        next_idx=updater._next_exp_index(bank)
        items=updater._reassign_exp_ids(cleaned,next_idx)
        updater.last_two_stage_counts["extracted"]=len(items)
        updater.last_update_succeeded=True
        updater.update_history.append({
            "num_failures_analyzed":len(examples),
            "num_experiences_generated":len(items),
            "experience_ids":[x.get("experience_id") for x in items],
            "mode":"two_stage",
        })
        return items
    except Exception as error:
        updater.last_failure_reason="two_stage_"+type(error).__name__
        # Keep successful training safe when the reflector fails.
        print("[T71TwoStage] "+type(error).__name__+": "+str(error)[:350],flush=True)
        return []
