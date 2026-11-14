#!/usr/bin/env python3
"""T71 controlled A/B: same 10 Step42 forks, revised stage-one label gate only.

No change to samples, stage two, experience bank, actor, or training.
"""
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import tools.t71_reflection_two_stage_probe as baseline
from openai import OpenAI
import httpx

OUT = Path("/home/zehao/echo-collab/T071/reports/reflection-step42-seed71-labelcheck")
ORIGINAL_CASES = Path("/home/zehao/echo-collab/T071/reports/reflection-step42-seed71/01_selected_cases.json")

STAGE1_REVISED = """You are an evidence-oriented analyst of a multistep web-search QA agent.
You receive 2 real SAME-STATE tree forks. At the fork, L did NOT inject the
listed fresh experience(s), while R DID. The tree's numeric reward gives L
higher success. One representative L reward=1 path and R reward=0 path
are shown in full, alongside the original question and the reference answer.

You must perform THREE steps IN ORDER for EACH fork:

STEP 1: VALIDATE THE SUCCESS/FAILURE CONTRAST.
The numeric reward is only a training label, NOT guaranteed semantic accuracy.
Explicitly compare (a) the question's exact requested scope and answer granularity,
(b) ground_truth, (c) L's final answer and its supporting observation,
(d) R's final answer and its supporting observation.
Pay particular attention to semantic equivalence (e.g. U.S. vs American),
different date precisions, lists that are incomplete, reference-answer ambiguity,
overlapping categories, and variants that are more accurate despite reward=0.
If L is not actually a correct/complete answer, OR R is an acceptable answer,
OR a semantic success/failure comparison cannot be trusted, mark
status='label_unreliable' and give concise reason. STOP FOR THAT FORK.
Do NOT generate a strategy even if a plausible general lesson is visible.

STEP 2: REFLECT ON BEHAVIOR.
Only for a credible L-success/R-failure contrast: describe the fork decision
state; the concrete difference between their searches, answers and observed
tool results; and what L visibly did that plausibly helped while R did not.
Cite specific action depth and relevant observation (no invented observations).
Do not assume the injected experience caused the outcome; other later
experience injections and stochastic search are confounders.

STEP 3: REUSABILITY GATE.
If this is a valid contrast but the evidence contains no concrete, actionable,
transferable search/answer strategy (e.g. just different keyword luck, chance
retrieval, or a single answer guess), mark status='no_reusable_insight'.
Only mark status='valid' when BOTH the contrast is semantically credible AND
the full trajectory supports a concrete reusable search strategy.
Do NOT write final experience entries: this is reflection only.
No quota on valid cases. Reject ambiguous cases rather than force lessons.

Return ONLY a valid JSON array of exactly 2 objects, one for each fork, using:
{"fork_id":"...","status":"valid|label_unreliable|no_reusable_insight",
"label_check":{"question_target":"...","ground_truth":"...",
"L_final_answer":"...","R_final_answer":"...",
"verdict":"credible|unreliable","reason":"..."},
"decision_state":"... or empty",
"observed_difference":"... or empty",
"success_mechanism":"... or empty",
"failure_pattern":"... or empty",
"reusable_insight":"... or null",
"evidence":[{"side":"L|R","depth":1,"action":"faithful excerpt",
"observation":"faithful excerpt/summary"}],
"uncertainty":"..."}
Do not omit any fork. Do not add explanatory text outside JSON."""

def save(name,data):
    OUT.mkdir(parents=True,exist_ok=True, mode=0o700)
    p=OUT/name
    p.write_text(data if isinstance(data,str) else json.dumps(data,ensure_ascii=False,indent=2,default=str),encoding="utf-8")
    p.chmod(0o600)

def run_batch(n, group,auth):
    prompt=STAGE1_REVISED+"\n\nCASES:\n"+json.dumps(group,ensure_ascii=False,default=str)
    save(f"batch{n:02d}_prompt.txt",prompt)
    with httpx.Client(trust_env=False,timeout=360) as h:
        c=OpenAI(api_key=auth["api_key"],base_url=auth["base_url"],max_retries=0,http_client=h)
        raw,usage=baseline.call(c,auth["model"],prompt,6000)
    save(f"batch{n:02d}_raw.txt",raw)
    reflections=baseline.parse_json(raw)
    if not isinstance(reflections,list) or len(reflections)!=2:
        raise ValueError(f"Batch {n} expected 2 reflections, got {len(reflections) if isinstance(reflections,list) else type(reflections)}")
    if [x.get("fork_id") for x in reflections]!=[x["fork_id"] for x in group]:
        raise ValueError(f"Batch {n} fork identity mismatch")
    save(f"batch{n:02d}_reflections.json",reflections)
    return n,reflections,usage,len(prompt)

def main():
    cases=json.loads(ORIGINAL_CASES.read_text())
    assert len(cases)==10
    save("01_selected_fork_ids.json",[c["fork_id"] for c in cases])
    auth=baseline.load_auth()
    print("STAGE1_REVISED",len(cases),"FORKS",auth["model"],flush=True)
    batches=[(i//2+1,cases[i:i+2]) for i in range(0,10,2)]
    outcomes={}
    with ThreadPoolExecutor(max_workers=2) as pool:
        futs=[pool.submit(run_batch,n,g,auth) for n,g in batches]
        for fut in as_completed(futs):
            n,refl,usage,chars=fut.result()
            outcomes[n]=(refl,usage)
            print("BATCH",n,"CHARS",chars,"STATUSES",[r.get("status") for r in refl],flush=True)
    reflections=[r for n in sorted(outcomes) for r in outcomes[n][0]]
    save("02_stage1_reflections.json",reflections)
    counts={k:sum(x.get("status")==k for x in reflections) for k in ["valid","label_unreliable","no_reusable_insight"]}
    print("GATE_COUNTS",counts,flush=True)

    # Keep stage 2 prompt EXACTLY the same as baseline; change only input reflection content.
    prompt2=baseline.STAGE2+"\n\nREFLECTIONS:\n"+json.dumps(reflections,ensure_ascii=False)
    save("03_stage2_prompt.txt",prompt2)
    with httpx.Client(trust_env=False,timeout=360) as h:
        c=OpenAI(api_key=auth["api_key"],base_url=auth["base_url"],max_retries=0,http_client=h)
        raw2,usage2=baseline.call(c,auth["model"],prompt2,7500)
    save("04_stage2_raw.txt",raw2)
    exp=baseline.parse_json(raw2)
    if not isinstance(exp,list):
        raise ValueError("Stage2 output must be a JSON list")
    save("05_stage2_experiences.json",exp)
    save("06_summary.json",{
        "seed":71,"fixed_sample_count":10,"batch_size":2,
        "model":auth["model"],"stage1_gate_counts":counts,
        "stage2_count":len(exp),
        "stage1_status_by_fork":{r["fork_id"]:r["status"] for r in reflections},
        "stage2_sources":[e.get("source_fork_ids") for e in exp],
        "stage1_usages":[outcomes[n][1] for n in sorted(outcomes)],
        "stage2_usage":usage2,
    })
    print("STAGE2_EXPERIENCES",len(exp),flush=True)

if __name__=="__main__":
    main()
