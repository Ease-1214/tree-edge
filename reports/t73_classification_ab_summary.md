# T73 Task/Step classification: real Fork A/B replay (2026-11-15)

## Exact historic inputs
Restored T73 tree-step5.pt, tree-step10.pt, tree-step15.pt, ran the unmodified Fork selector with k=100000, sampling=random, seed=step and the original EDGE one-call adapter. Compared the 10 selected source_leaf_id pairs per window with the saved tree_management_step{5,10,15}.json; all matched in order. Used initial bank for Step 5, Step 5 bank for Step 10, and Step 10 bank for Step 15. These tests do not alter the running T73.

## Test
- Each window used the same 10 selected Fork examples, bank context and model gpt-6.1-sol.
- A: original generation prompt.
- B: strengthened Task/Step classification instructions.
- C: B plus an explicit constraint that EVERY Task state_cues item must be visible from original question only.
- Actual live model generation, not supplying any previously generated experience as a forced output. The generation is nondeterministic, so prompts can yield DIFFERENT experiences.
- Each set was then checked via a separate batch semantic timing audit; reported counts represent this independent semantic audit, NOT an externally verified ground truth.

| Version | Generated | Declared Task | Declared Step | Independent timing audit accepted |
| --- | ---: | ---: | ---: | ---: |
| A original | 7 | 7 | 0 | 5 / 7 |
| B first strengthened | 7 | 5 | 2 | 6 / 7 |
| C final strengthened | 7 | 6 | 1 | 7 / 7 |

The B exception: a Task about total television-series episode count included a state_cue requiring that a passage already gave a season-level count. C resolves this by moving future-evidence checks into recommended_action and requiring Task state_cues to be question-only.

A Step in C is triggered by retrieved passages containing multiple numbers for different scopes (individual installment vs entire series); it is correctly Step. Task strategies in C have question-only triggers and question-only state_cues.

## Files
- report t73_ab_generation_audit.json contains original A and B independent check results.
- report t73_ab_generation_audit_v2.json contains C check results.
- ab_prompts/ contains exact reconstructed generation prompts.
- ab_generations/ contains raw model output per generation.
- tools/t73_reconstruct_reflector_inputs.py, tools/t73_ab_generate_experiences.py,
  tools/t73_ab_generate_revised2.py, tools/t73_ab_audit_generated.py,
  tools/t73_ab_audit_revised2.py reproduce these offline runs.

## Runtime fix
- Strong Task/Step timing rules in experience_type_validation.py and generation prompt.
- Independent one-call batch semantic validation after first-pass extraction. Reject mismatched/ambiguous output; never silently relabel. Fail closed if validation API/schema fails.
- Invalid/missing model-provided Fork experience_type no longer inferred from Fork depth.
- Metrics: reflector/classification_submitted, accepted, rejected, ambiguous, failed, invalid_type_dropped.
- Zero changes to Fork selection, training algorithm or currently running T73.

## Limitations
This is 3 window samples with one generation per prompt variant. Different generations across variants prevent paired accuracy claims on identical *experiences*. Independent classifier is also an LLM. Correctness over all future generations is not guaranteed. The second validator remains necessary for defense-in-depth.
