#!/usr/bin/env python3
"""T71 narrow semantic-label gate: controlled third arm on identical 10 Forks."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from tools import t71_reflection_labelcheck_ab as ab

ab.OUT=Path("/home/zehao/echo-collab/T071/reports/reflection-step42-seed71-narrowgate")
ab.STAGE1_REVISED="""You are an expert analyst of multistep web-search QA agent trajectories.
You receive 2 real SAME-STATE fork comparisons per batch, each including the
question, reference answer, reward labels, and complete L and R paths.
L did not inject the newly selected experiences AT THIS FORK, while R did.
The recorded L leaf has reward=1 and R has reward=0; their subtree mean delta<0.
Both can inherit earlier experiences, and later experience injections may differ.

Your job is NOT to correct old experiences. Analyze what reusable new behavior
the successful L path shows in contrast to the unsuccessful R path.
Perform these THREE stages, IN ORDER, separately for each fork:

A. SANITY-CHECK REWARD LABELS, USING A NARROW REJECTION CRITERION.
Explicitly compare what the question asks, the supplied ground truth, and both
FINAL answers. Distinguish reward-matching from semantic correctness.
Only reject with status="label_unreliable" when there is a POSITIVE, SPECIFIC
reason to doubt the intended L-success/R-failure contrast:
 - L's final answer clearly omits a required part or is plainly wrong;
 - R's answer is semantically acceptable, equivalent to L, or more correct;
 - The question's actual wording unambiguously makes the reward preference
   wrong, OR has an ambiguity that makes R a reasonable answer.
Do NOT demand that every clue be proven before a plausible ground-truth answer
counts as success. Missing secondary evidence is uncertainty, not automatically
a label error. If L answer matches the reference and R answer conflicts with
the core requested fact, treat the outcome contrast as credible unless there is
specific conflicting semantic evidence.
Do NOT reverse the labels by mistake: explicitly restate which final answer
belongs to L and R before deciding. Do not invent exceptions or ambiguity to
justify filtering. If unsure but no positive evidence the labels are wrong,
proceed to B; briefly record the uncertainty.

CRITICAL DISTINCTION: Evidence of success/failure is NOT evidence of CAUSATION.
Stochastic search results, different continuation lengths, and additional
experience injections AFTER THE FORK can weaken the causal interpretation.
They do NOT make answer labels unreliable. Record them under uncertainty,
NEVER use them alone to reject status='label_unreliable'.

B. REFLECT ON ACTUAL DECISIONS.
When the outcome contrast appears credible, compare the shared decision state,
L and R's exact fork actions, later actions, search observations and answers.
Describe the narrow, observable strategy difference and plausible reason it
helped L. Explicitly distinguish observed evidence from speculation.
Cite depths and actual action/observation snippets. Do not say the injected
experiences caused the result merely because their mean delta is negative.

C. DECIDE WHETHER THERE IS A REUSABLE INSIGHT.
Use status='valid' when a visible decision DIFFERENCE supports an actionable,
cross-question search strategy with recognizable WHEN and specific DO.
One credible case can support a provisional strategy; do not demand causal
proof or multiple independent successes.
Use status='no_reusable_insight' only when the outcomes differ but the full
path provides no defensible, actionable and transferable lesson, such as
mere tool randomness or an entity-specific guess. An insight can be a more
precise query relation, entity disambiguation, use of an evidence check,
or stopping/answer decision.
Do NOT produce final experiences; stage two will handle extraction.
Do NOT force all cases to be valid, but do not filter out useful evidence
just because it is imperfect. Maintain an objective evidence standard.

Return ONLY a valid JSON array containing EXACTLY TWO objects in input order:
{"fork_id":"...","status":"valid|label_unreliable|no_reusable_insight",
"label_check":{"question_target":"...","ground_truth":"...",
"L_final_answer":"...","R_final_answer":"...",
"verdict":"credible|unreliable","reason":"specific, factual rationale"},
"decision_state":"... or empty",
"observed_difference":"... or empty",
"success_mechanism":"... or empty",
"failure_pattern":"... or empty",
"reusable_insight":"... or null",
"evidence":[{"side":"L|R","depth":1,"action":"faithful excerpt",
"observation":"faithful snippet/summary"}],
"uncertainty":"..."}
Return no prose outside JSON."""
if __name__=="__main__":
    ab.main()
