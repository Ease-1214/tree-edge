#!/usr/bin/env python3
"""T71 third controlled arm: answer-verdict-first reflection; same 10 forks and stage2."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from tools import t71_reflection_labelcheck_ab as ab

ab.OUT=Path("/home/zehao/echo-collab/T071/reports/reflection-step42-seed71-verdictfirst")
ab.STAGE1_REVISED="""You analyze multistep search-agent trajectory Forks to discover useful,
GENERALIZABLE NEW SEARCH STRATEGIES, not to correct old experiences.
Each input contains 2 independently evaluated cases. At each fork, L does
not inject the newly selected experiences, while R injects them. L has
reward 1 and R reward 0 in the representative full trajectories.
These reward numbers are fallible, not gold semantic judgments.

FOR EACH FORK perform THREE SUBSTAGES, IN ORDER:

I. CHECK ANSWERS FIRST (REQUIRED BEFORE ANY REFLECTION).
Using the actual question wording, provided reference answer, both final
answers, and visible evidence, fill TWO INDEPENDENT SEMANTIC judgments:
"L_answer_acceptable": true / false / null (uncertain)
"R_answer_acceptable": true / false / null (uncertain).
Interpret 'acceptable' as an answer that a knowledgeable human would
reasonably accept as answering the question, NOT exact-match reward:
- Synonyms, abbreviations, and paraphrases of the same entity are acceptable.
- A more specific but factually valid subcategory may also be acceptable
  when it still correctly answers a question for a category.
- A more precise correct date is not worse merely because the reward says 0.
- An incomplete answer to a question requiring a multi-item list is false.
- A response matching the *explicit* broader/narrower geographic unit
  requested can be acceptable even if the reference assumes another scope.
- A clue not fully proven by retrieved snippets does not AUTOMATICALLY make
  a ground-truth-matching final answer unacceptable.
- If the question is semantically ambiguous and R is one reasonable reading,
  count R as acceptable.
Do not invent contradictory evidence or fake ambiguity.

THE DECISION RULE IS NOT OPTIONAL:
IF L_answer_acceptable == false OR R_answer_acceptable == true:
  status MUST BE "label_unreliable".
  Give a concise explicit reason, no reusable insight. STOP this case.
ELSE if L_answer_acceptable is null OR R_answer_acceptable is null:
  When there is a concrete reason the target or reference is unreliable,
  reject; otherwise continue but explicitly note uncertainty.
ELSE continue with the behavioral analysis.

The label check tests whether one answer is better, NOT whether the
freshly injected experience CAUSED the improvement. Later injections,
trajectory differences, noise, or imperfect causal isolation NEVER
by themselves trigger "label_unreliable"; they belong under uncertainty.

II. REFLECT ON THE VISIBLE BEHAVIOR for non-rejected cases.
Identify the shared state, the precise L/R fork actions, follow-up searches,
observations and final answers. Explain what L visibly did differently
that plausibly helped, and where R made a concrete, actionable misstep.
Support by actual turn depth and cited observation. Avoid unsupported
causal claims about the old experience. If there was no real behavioral
difference with a meaningful lesson, do not invent one.

III. REUSABILITY CHECK.
If the observation supports a specific action the agent could take in a
recognizable future state, mark "valid" and write a concise provisional
reusable_insight (not the final structured experience). One good case can
support a provisional insight without causal proof.
Otherwise mark "no_reusable_insight". Do not force 1 insight per fork.
It is invalid to return status="valid" with reusable_insight empty/null.
Carefully verify your final status is consistent with the answer judgments.

RETURN ONLY JSON: an array of exactly 2 objects in input order.
{"fork_id":"...","status":"valid|label_unreliable|no_reusable_insight",
"label_check":{"question_target":"...","ground_truth":"...",
"L_final_answer":"...","R_final_answer":"...",
"L_answer_acceptable":true,"R_answer_acceptable":false,
"verdict":"credible|unreliable","reason":"specific semantic rationale"},
"decision_state":"... or empty",
"observed_difference":"... or empty",
"success_mechanism":"... or empty",
"failure_pattern":"... or empty",
"reusable_insight":"... or null",
"evidence":[{"side":"L|R","depth":1,"action":"actual faithful excerpt",
"observation":"short faithful observation summary"}],
"uncertainty":"..."}
Note: true/false/null are JSON values, not text.
Do not write final experiences: stage two handles that."""
if __name__=="__main__":
    ab.main()
