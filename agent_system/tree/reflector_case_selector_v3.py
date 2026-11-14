"""Select 10 useful full-trajectory contrasts from a 40-fork pool.

One short model-selection call; follow it with the full evidence prompt.
The selector sees previews only for ranking; no shortened evidence is
used when generating final experiences.
"""
import json

def selection_prompt(examples, bank):
    indexed = []
    for pool in ("task_experiences", "step_experiences"):
        for x in bank.get(pool,[]):
            indexed.append({
                "id":x.get("experience_id"),"pool":pool,
                "trigger":x.get("trigger") or x.get("when_to_apply") or x.get("title"),
                "action":x.get("recommended_action") or x.get("principle"),
            })
    previews=[]
    for i,e in enumerate(examples,1):
        def short_path(key):
            return [
                {"turn": t.get("depth"),"action": t.get("action"),
                 "search_observation": t.get("observation") if t.get("depth")==e["fork_depth"] else ""}
                for t in e[key] or []
            ]
        previews.append({
            "index":i, "question":e["task"],
            "fork_depth":e["fork_depth"], "strict":e["fork_tier"]=="strict",
            "subtree_rates": [e["left_subtree_rate"],e["right_subtree_rate"]],
            "action_left":e["left_action"],
            "action_right":e["right_action"],
            "shared_observation_before_fork":e["fork_observation"],
            "successful_path":short_path("success_turns"),
            "failed_path":short_path("failed_turns"),
        })
    return """Choose EXACTLY 10 high-quality contrasts for an experience writer
from the numbered 40 SAME-STATE branching search examples.

Your selection is the main bottleneck: do not select merely the largest
terminal-success contrast. Especially avoid trivial query rewordings,
random alternative named answers, or a failure that only supports
a generic existing rule such as "do not guess" or "find evidence".

PRIORITIZE cases likely to yield *new executable decision rules*:
- T potential: a multi-turn plan identifiable from the ORIGINAL QUESTION,
  such as which constraint/entity/relation to solve first, then how to
  use its result to resolve the final relation. No generic "decompose".
- S potential: the common search observation plus question reveals a
  concrete actionable error: wrong role of a named person, distinct
  requested relation vs tempting nearby relation, misplaced intermediate,
  incorrect stopping decision, specific event-vs-event timeframe, etc.
- A narrower, distinct decision cue IS allowed even if the bank already
  has a broad umbrella rule. Reject only operationally redundant advice.
- Prioritize diverse failure mechanisms over 10 variations of the same
  problem (e.g. "guessing without evidence").
- Cases from root or internal forks are both allowed; 10 cases can mix
  depths. Do not assign the final T/S classification yourself.
- Only candidate comparisons for which there is enough visible
  evidence to diagnose a plausible corrective action should be chosen.
  Success rates aren't causal proof.
- These are PREVIEWS for SELECTION; the writer will receive complete
  original trajectories and observations of all 10 selected cases.

Return ONLY JSON object {"selected_indices":[10 distinct one-based integers]}.
No other text.

EXISTING BANK:
""" + json.dumps(indexed,ensure_ascii=False) + "\n\nCANDIDATE PREVIEWS:\n" + json.dumps(previews,ensure_ascii=False) + "\n"



def select_reflector_examples_with_api(updater, pool, bank, *, k=10):
    """Choose diverse learnable cases; safely fall back to heuristic ranking."""
    if len(pool) <= k:
        updater.last_selector_status = {
            "used_llm": False, "reason": "pool_small", "pool": len(pool),
        }
        return pool[:k]
    prompt = selection_prompt(pool, bank)
    ids = []
    try:
        resp = updater._completion_with_deadline({
            "model": updater.model,
            "messages": [{"role": "user", "content": prompt}],
            updater.token_limit_param: 2048,
        })
        raw = resp.choices[0].message.content or ""
        obj = json.loads(raw[raw.index("{"):raw.rindex("}") + 1])
        for val in obj.get("selected_indices", []):
            try:
                i = int(val)
            except (TypeError, ValueError):
                continue
            if 1 <= i <= len(pool) and i not in ids:
                ids.append(i)
        updater.last_selector_status = {
            "used_llm": True, "reason": None, "pool": len(pool),
            "prompt_chars": len(prompt), "ids": ids[:k],
        }
    except Exception as error:
        updater.last_selector_status = {
            "used_llm": False, "reason": type(error).__name__,
            "pool": len(pool),
        }
        print("[ReflectorSelector] fallback:", type(error).__name__, flush=True)
    # Never reduce the number of full contrast examples due to a malformed
    # selection response. The first pool items are the deterministic fallback.
    for i in range(1, len(pool) + 1):
        if len(ids) >= k:
            break
        if i not in ids:
            ids.append(i)
    return [pool[i - 1] for i in ids[:k]]
