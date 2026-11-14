#!/usr/bin/env python3
"""Offline T71 stage-2 prompt-only probe. DO NOT write to trainer/bank."""
import json, sys, time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools import t71_reflection_two_stage_probe as base
from openai import OpenAI
import httpx

ROOT=Path("/home/zehao/echo-collab/T071/reports/reflection-stage2-max5-probe-20261114")
ORIGINAL=Path("/home/zehao/echo-collab/T071/reports/reflection-step42-seed71")
STAGE2="""You are the experience distillation stage of a multi-turn search agent.

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

def save(f, obj):
    ROOT.mkdir(parents=True,exist_ok=True,mode=0o700)
    p=ROOT/f
    p.write_text(obj if isinstance(obj,str) else json.dumps(obj,indent=2,ensure_ascii=False),encoding="utf-8")
    p.chmod(0o600)

def evaluate(exp, reflections):
    valid={x["fork_id"] for x in reflections if x.get("status")=="valid"}
    ids=list(exp) if isinstance(exp,list) else []
    issue=[]
    if not isinstance(exp,list): issue.append("output_not_list")
    if len(ids)>5: issue.append("exceed_max_5")
    if len(ids)>0 and not valid:issue.append("nonempty_on_no_valid")
    pairs=[]
    for i,x in enumerate(ids):
        if not isinstance(x,dict): issue.append(f"item{i}_not_dict"); continue
        required={"experience_type","trigger","recommended_action","avoid_action","state_cues","retrieval_text","source_fork_ids"}
        missing=required-set(x)
        if missing:issue.append(f"item{i}_missing_{missing}")
        if x.get("experience_type") not in ("task","step"):issue.append(f"item{i}_bad_type")
        cited=x.get("source_fork_ids",[])
        if not cited or not isinstance(cited,list) or any(s not in valid for s in cited):
            issue.append(f"item{i}_invalid_source")
        if not str(x.get("trigger","")).strip() or not str(x.get("recommended_action","")).strip() or not str(x.get("avoid_action","")).strip():
            issue.append(f"item{i}_empty_required_behavior")
        pairs.append((i, cited))
    return {"count":len(ids),"valid_input":len(valid),"cited_sources":pairs,"violations":issue}

def run(n, name, records, auth):
    prompt=STAGE2+"\n\nSTAGE_ONE_REFLECTIONS:\n"+json.dumps(records,ensure_ascii=False)
    save(f"{n:02d}_{name}_prompt.txt",prompt)
    t=time.monotonic()
    with httpx.Client(trust_env=False,timeout=360) as h:
        client=OpenAI(api_key=auth["api_key"],base_url=auth["base_url"],max_retries=0,http_client=h)
        raw,usage=base.call(client,auth["model"],prompt,7500)
    elapsed=round(time.monotonic()-t,2)
    save(f"{n:02d}_{name}_raw.txt",raw)
    exp=base.parse_json(raw)
    save(f"{n:02d}_{name}_experiences.json",exp)
    return {"case":name,"repeat":n,"seconds":elapsed,**evaluate(exp,records),"usage":usage}

def main():
    reflections=json.loads((ORIGINAL/"04_stage1_reflections.json").read_text(encoding="utf-8"))
    assert len(reflections)==10 and len({x['fork_id'] for x in reflections})==10
    auth=base.load_auth()
    save("prompt_template.txt",STAGE2)
    print("MODEL",auth["model"],"STAGE2 MAX5 PROBE",flush=True)
    # Repeat full 10-fork prompt twice; one short 3-case set and one no-valid test.
    no_valid=[dict(x,status="insufficient",reusable_insight=None) for x in reflections[:2]]
    experiments=[(1,"full_10",reflections),(2,"full_10",reflections),
                 (3,"three_valid",reflections[:3]),(4,"no_valid",no_valid)]
    results=[]
    with ThreadPoolExecutor(max_workers=2) as pool:
        fs=[pool.submit(run,n,name,items,auth) for n,name,items in experiments]
        for f in as_completed(fs):
            v=f.result()
            print("RESULT",v['repeat'],v['case'],"count",v['count'],
                  "violations",v['violations'],"seconds",v['seconds'],flush=True)
            results.append(v)
    save("summary.json",sorted(results,key=lambda x:x['repeat']))
    print("COMPLETE",flush=True)

if __name__=="__main__":
    main()
