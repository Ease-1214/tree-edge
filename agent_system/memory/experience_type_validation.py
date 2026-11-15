"""Semantic timing check for Task/Step search experiences.

This module runs only on newly produced Fork-derived search experiences,
not the live bank. The LLM does not edit content or override the generator:
anything inconsistent or ambiguous is rejected before the bank upsert.
"""
import json
from typing import Any, Callable, Dict, List, Tuple

CLASSIFICATION_GUIDANCE = """
EXPERIENCE TYPE SELF-CHECK (MANDATORY):
Choose task/step by what information is required to recognize the WHEN
(trigger and state_cues), NOT by the depth of the source Fork.
- TASK: Its applicability is recognizable from the original question ALONE,
  before a search has been issued. The recommended_action may prescribe
  searching and verifying results later; this does NOT make it Step.
- STEP: Its applicability needs the actual current search state: retrieved
  snippets, a resolved intermediate entity, missing/conflicting evidence,
  a previous failed query, or the agent's progress after a search.
- For every proposed experience, mentally hide all search results and
  search history. Can you still tell whether its trigger applies, using only
  the original question? Yes => task; no => step.
- Classify the ACTUAL trigger written, not a different hypothetical rewrite.
- For mixed or unclear triggers, rewrite to one precise, defensible
  applicability condition before deciding the type; if no valid rule can
  be stated, omit it. Do not add answer-specific facts.
- For TASK, EVERY item in state_cues must itself be checkable from the
  ORIGINAL QUESTION alone. Never put what a future search passage says,
  which candidate a tool found, or what evidence is missing in task state_cues.
  Put future checks in recommended_action or avoid_action instead.
  If an observed search condition is genuinely required to trigger the
  strategy, classify it as STEP and write that condition explicitly.
- Before outputting, check EACH proposed TASK state_cue separately using
  only the original question. Remove or rewrite any future-dependent cue
  as a question-only cue without altering the supported decision.
- Fork depth, task complexity, and number of hops are NOT classification
  criteria. Neither type is required; do NOT force any type ratio.
"""

VALIDATION_INSTRUCTION = """You audit the applicability timing of search-agent experiences.
You are NOT the experience writer; do not rewrite, add, delete, or change
the advice, and do not use any source Fork depth or reward.

Each experience has an author's declared type. Independently infer:
- task: its actual trigger can be recognized from the ORIGINAL QUESTION ALONE.
- step: recognizing its actual trigger REQUIRES an observation, search result,
  established intermediate answer, search history, or already-completed action
  (including triggers that combine a question pattern with a search-result
  condition, e.g., "A question asks X and search results mention Y" — that is
  a valid STEP experience, NOT ambiguous).
- ambiguous: contradictory, empty, or non-actionable conditions that cannot
  be used reliably at either stage.

CRITICAL RULES:
1. Evaluate TRIGGER, STATE_CUES, and RETRIEVAL_TEXT together as applicability conditions.
2. A task strategy may tell the agent to check future search results; an
   action mentioning results is NOT sufficient to classify it as step.
3. If the trigger, state_cues, or retrieval_text say results already mention
   people, conflicting dates, missing information, etc., that is a STEP
   condition, even if its action includes an initial search instruction.
4. Do NOT infer type from multi-hop reasoning, source Fork depth, whether
   it was successful, or the author's declared label.
5. Do NOT imagine an improved trigger: classify what is actually supplied.
6. Return one result for EVERY index, in the same order. Be conservative.

Output ONLY a JSON array:
[{"index":1,"inferred_type":"task|step|ambiguous","reason":"short concrete reason"}]

EXPERIENCES:
"""


def validate_experience_types(
    experiences: List[Dict[str, Any]],
    complete: Callable[[Dict[str, Any]], Any],
    *,
    model: str,
    token_limit_param: str = "max_tokens",
    max_completion_tokens: int = 2048,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """One batch API request. Invalid/incomplete responses raise, fail-closed.

    Returns accepted objects (with experience_type aligned to inferred_type
    when unambiguous task/step) and per-item classification decisions.
    Ambiguous items are rejected.
    """
    if not experiences:
        return [], []
    if len(experiences) > 10:
        raise ValueError("Classification batch exceeds maximum 10 experiences")
    payload = []
    for index, item in enumerate(experiences, start=1):
        declared = item.get("experience_type")
        if declared not in ("task", "step"):
            raise ValueError(f"Invalid declared experience type at index {index}")
        payload.append({
            "index": index,
            "declared_type": declared,
            "trigger": item.get("trigger", ""),
            "state_cues": item.get("state_cues", []),
            "retrieval_text": item.get("retrieval_text", ""),
            "recommended_action": item.get("recommended_action", ""),
            "avoid_action": item.get("avoid_action", ""),
        })
    instruction = VALIDATION_INSTRUCTION + json.dumps(payload, ensure_ascii=False)
    response = complete({
        "model": model,
        "messages": [{"role": "user", "content": instruction}],
        token_limit_param: min(3072, max(1024, int(max_completion_tokens))),
    })
    raw = response.choices[0].message.content or ""
    start, end = raw.find("["), raw.rfind("]")
    if start < 0 or end <= start:
        raise ValueError("Classification response is not a JSON array")
    result = json.loads(raw[start:end + 1])
    if not isinstance(result, list) or len(result) != len(experiences):
        raise ValueError("Classification response count differs from input count")
    decisions = []
    accepted = []
    seen = set()
    for expected, (item, decision) in enumerate(zip(experiences, result), start=1):
        if not isinstance(decision, dict):
            raise ValueError("Classification decision is not an object")
        index = decision.get("index")
        if type(index) is not int or index != expected or index in seen:
            raise ValueError("Classification response indexes are invalid")
        seen.add(index)
        inferred = decision.get("inferred_type")
        reason = decision.get("reason")
        if inferred not in ("task", "step", "ambiguous"):
            raise ValueError("Classification response inferred_type is invalid")
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("Classification response is missing rationale")
        is_valid = inferred in ("task", "step")
        relabeled = is_valid and inferred != item["experience_type"]
        decisions.append({
            "index": index,
            "declared_type": item["experience_type"],
            "inferred_type": inferred,
            "accepted": is_valid,
            "relabeled": relabeled,
            "reason": reason.strip()[:500],
        })
        if is_valid:
            if relabeled:
                updated = dict(item)
                updated["experience_type"] = inferred
                accepted.append(updated)
            else:
                accepted.append(item)
    return accepted, decisions
