#!/usr/bin/env python3
"""T71 isolated offline test: random 10 negative Forks -> reflection -> extraction.

Does not modify actor, tree, training checkpoints, or experience bank.
"""
import argparse
import json
import os
import random
from collections import defaultdict
from pathlib import Path
from statistics import mean

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["OMP_NUM_THREADS"] = "1"
for k in ("ALL_PROXY","all_proxy","HTTP_PROXY","http_proxy","HTTPS_PROXY","https_proxy"):
    os.environ.pop(k, None)

import torch
from agent_system.tree.core import _turn_records, _bank_lookup, normalize_action
from agent_system.tree.reflector_selection_v3 import _action_difference, _downstream_experience_count

ROOT = Path("/home/zehao/echo-collab/T071/reports/reflection-step42-seed71")
SNAPSHOT = Path("/home/zehao/echo-collab/T070/conditional-natural4-sft/v2-opsd-active-005/tree-private/tree-step42.pt")

def save(name, content):
    ROOT.mkdir(parents=True, exist_ok=True, mode=0o700)
    target = ROOT / name
    if not isinstance(content, str):
        content = json.dumps(content, ensure_ascii=False, indent=2, default=str)
    fd = os.open(str(target), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(content)

def load_auth():
    # Use credentials already authorized for the ongoing T70 local model gateway.
    # Never print or save tokens.
    env = dict(os.environ)
    if not env.get("EXPERIENCE_UPDATE_API_KEY"):
        p = Path("/proc/1446193/environ")
        if p.exists():
            for part in p.read_bytes().split(b"\0"):
                if b"=" in part:
                    k, v = part.split(b"=", 1)
                    if k.decode() in ("EXPERIENCE_UPDATE_API_KEY",
                                      "EXPERIENCE_UPDATE_BASE_URL",
                                      "EXPERIENCE_UPDATE_MODEL"):
                        env[k.decode()] = v.decode()
    key = env.get("EXPERIENCE_UPDATE_API_KEY")
    if not key:
        raise RuntimeError("Live T70 model credentials not accessible.")
    return dict(api_key=key,
                base_url=env.get("EXPERIENCE_UPDATE_BASE_URL","http://127.0.0.1:29663/v1"),
                model=env.get("EXPERIENCE_UPDATE_MODEL","gpt-6.1-sol"))

def choose_forks(tree, seed=71, k=10):
    descendants = defaultdict(list)
    for leaf in tree.leaves:
        if leaf.source != "tree":
            continue
        for nid in leaf.node_ids:
            descendants[nid].append(leaf)
    selected = []
    for fork in tree.forks.values():
        a = tree.nodes.get(fork.left_node_id)
        b = tree.nodes.get(fork.right_node_id)
        if not a or not b or not fork.new_experience_ids:
            continue
        if a.uid != b.uid or a.depth != b.depth or (
            a.state_before.history != b.state_before.history or
            a.state_before.last_observation != b.state_before.last_observation
        ):
            continue
        l = descendants[fork.left_node_id]
        r = descendants[fork.right_node_id]
        if not l or not r:
            continue
        action_a = normalize_action(a.action)
        action_b = normalize_action(b.action)
        if action_a[0]=="invalid" or action_b[0]=="invalid" or action_a==action_b:
            continue
        if _action_difference(a.action,b.action) is None:
            continue
        pL,pR = mean(x.reward for x in l), mean(x.reward for x in r)
        if not (pR < pL):
            continue
        wins = [x for x in l if x.reward == 1]
        losses = [x for x in r if x.reward == 0]
        if not wins or not losses:
            continue
        pair = min(((w,f) for w in wins for f in losses),
                   key=lambda p: (_downstream_experience_count(tree,p[0],fork.depth)
                                  +_downstream_experience_count(tree,p[1],fork.depth),
                                  p[0].leaf_id,p[1].leaf_id))
        selected.append((fork,a,b,pair[0],pair[1],l,r,pL,pR))
    assert len(selected)==51, f"Eligibility changed: {len(selected)} != 51"
    random_sample = random.Random(seed).sample(sorted(selected,key=lambda x:x[0].fork_id), k)
    bank = _bank_lookup(tree.bank_snapshot)
    cases = []
    for fork,a,b,success,failure,l,r,pL,pR in random_sample:
        s=a.state_before
        cases.append({
            "fork_id":fork.fork_id,
            "question":s.question,
            "ground_truth":s.ground_truth,
            "fork_depth":fork.depth,
            "question_information_available_before_fork":s.history,
            "latest_observation_before_fork":s.last_observation,
            "left_action_at_fork":a.action,
            "right_action_at_fork":b.action,
            "right_injected_old_experiences":[
                {"experience_id":eid,"trigger":bank.get(eid,{}).get("trigger"),
                 "recommended_action":bank.get(eid,{}).get("recommended_action"),
                 "avoid_action":bank.get(eid,{}).get("avoid_action"),
                 "principle":bank.get(eid,{}).get("principle")}
                for eid in fork.new_experience_ids],
            "left_leaf_rewards":[x.reward for x in l],
            "right_leaf_rewards":[x.reward for x in r],
            "pL":pL,"pR":pR,"delta":pR-pL,
            "left_success_leaf_id":success.leaf_id,
            "right_failure_leaf_id":failure.leaf_id,
            "left_success_full_path":_turn_records(tree,success,bank),
            "right_failure_full_path":_turn_records(tree,failure,bank),
            "post_fork_other_experience_injections":{
                "left":_downstream_experience_count(tree,success,fork.depth),
                "right":_downstream_experience_count(tree,failure,fork.depth)},
        })
    return cases

STAGE1 = """You are an evidence-oriented analyst of a multistep web-search QA agent.
Input: exactly 10 real SAME-STATE tree forks. At the fork, L did NOT inject the
listed new experience; R DID inject it. Average L descendant success > R.
One representative successful L path and unsuccessful R path are shown in full.

Your sole goal is REFLECTION: identify the specific observable behavior that
made the L path successful but was missing or mishandled in R, IF supported.
You are NOT evaluating whether the old experience is inherently wrong.
Do NOT propose final generalized experiences or rewrite existing experiences.

For EACH supplied fork:
1. Say what was known/unknown at the decision point.
2. Identify the concrete decision difference at the fork and relevant downstream
   actions and search observations. Use evidence (node depth, actual query, tool
   observation and final answer); do not infer unsupported missing facts.
3. Explain the plausible success mechanism and the failed path's observable gap.
4. State a candidate REUSABLE SEARCH STRATEGY only if the observed evidence supports one.
5. Mark 'insufficient' if the pair differs only through accidental retrieval,
   wording, incidental detail, ambiguous attribution or lacks an actionable lesson.
Never say the experience caused the outcome from success rate alone. Later
interventions are potential confounders. Follow the trajectories, not hindsight.
Actions may contain only the parsed action, not complete private reasoning.
Keep judgments concise but specific and cite fork IDs and exact turn depths.

Return ONLY valid JSON array of exactly 10 objects:
{"fork_id":"...","status":"valid|insufficient",
 "decision_state":"...","observed_difference":"...",
 "success_mechanism":"...","failure_pattern":"...",
 "reusable_insight":"... or null",
 "evidence":[{"side":"L|R","depth":1,"action":"short exact or faithful excerpt",
 "observation":"short faithful excerpt or observation summary"}],
 "uncertainty":"..."}.
Do not omit any fork. Do not include explanations outside JSON."""

STAGE2 = """You are a search-agent EXPERIENCE EXTRACTION expert.
You receive 10 evidence-grounded analyses of branching search trajectories.
This stage does NOT re-analyze entire trajectories; use ONLY the valid
observations and the evidence provided in stage 1.

Create practical, generalizable search experiences describing:
WHEN the agent can recognize a state; DO a concrete search/evidence/answer
action; AVOID the specific failure. Extrapolate to different entities and
questions, while retaining an operationally specific next action.

Only use observations whose status is 'valid'. Do not generate generic
maxims ('be careful', 'search more', 'make sure answer is right'), factual
one-off answers, post-hoc rules unsupported by evidence, or guesses about
why old experience failed. A single supported case can justify a hypothesis;
do not pretend causal proof. Zero experiences is completely acceptable.
No need to fill a quota; at most 10 items if genuinely different insights.

The experience type is determined by WHEN IT CAN BE RETRIEVED:
- task: trigger recognizable from original question only; useful from start.
- step: trigger depends on the current tool observation and question; useful
  as next action at a specific search state.
A task experience must not require observing future search results.

Return ONLY a JSON array. For each item:
{"experience_type":"task|step",
 "trigger":"checkable applicability condition",
 "recommended_action":"specific executable search/verification/answer behavior",
 "avoid_action":"specific decision to avoid",
 "state_cues":["observable question or observation cues"],
 "retrieval_text":"one or two standalone sentences to retrieve this experience",
 "source_fork_ids":["fork IDs from stage 1"]}.
Avoid proper names from examples in final experience. No explanations outside JSON."""

def parse_json(raw):
    s=raw.strip()
    if s.startswith("```"):
        s=s.split("\n",1)[1].rsplit("```",1)[0].strip()
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        a=s.find("["); b=s.rfind("]")
        if a>=0 and b>=a:
            return json.loads(s[a:b+1])
        raise

def call(client,model,prompt,max_tokens):
    resp=client.chat.completions.create(
        model=model,
        messages=[{"role":"user","content":prompt}],
        max_completion_tokens=max_tokens,
        timeout=480
    )
    return resp.choices[0].message.content or "", (
        resp.usage.model_dump() if resp.usage else {})

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--prepare-only",action="store_true")
    args=p.parse_args()
    print("LOAD", SNAPSHOT.name,flush=True)
    tree=torch.load(SNAPSHOT,map_location="cpu",weights_only=False)
    cases=choose_forks(tree)
    ids=[x["fork_id"] for x in cases]
    print("SAMPLE",json.dumps(ids),flush=True)
    save("01_selected_cases.json",cases)
    stage1_prompt=STAGE1+"\n\nCASES:\n"+json.dumps(cases,ensure_ascii=False,default=str)
    save("02_stage1_prompt.txt",stage1_prompt)
    print("STAGE1_PROMPT_CHARS",len(stage1_prompt),flush=True)
    if args.prepare_only:
        return
    auth=load_auth()
    from openai import OpenAI
    import httpx
    client=OpenAI(api_key=auth["api_key"],base_url=auth["base_url"],
                  http_client=httpx.Client(trust_env=False,timeout=480))
    print("MODEL",auth["model"],flush=True)
    raw1,usage1=call(client,auth["model"],stage1_prompt,15000)
    save("03_stage1_raw.txt",raw1)
    reflections=parse_json(raw1)
    save("04_stage1_reflections.json",reflections)
    print("STAGE1_REFLECTIONS",len(reflections),"VALID",
          sum(x.get("status")=="valid" for x in reflections),
          "TOKENS",json.dumps(usage1),flush=True)
    stage2_prompt=STAGE2+"\n\nREFLECTIONS:\n"+json.dumps(reflections,ensure_ascii=False)
    save("05_stage2_prompt.txt",stage2_prompt)
    raw2,usage2=call(client,auth["model"],stage2_prompt,12000)
    save("06_stage2_raw.txt",raw2)
    experiences=parse_json(raw2)
    save("07_stage2_experiences.json",experiences)
    print("STAGE2_EXPERIENCES",len(experiences),"TOKENS",json.dumps(usage2),flush=True)
    save("08_summary.json",{
        "seed":71,"candidate_forks":51,"sample_size":len(cases),
        "sample_fork_ids":ids,"stage1_valid":sum(x.get("status")=="valid" for x in reflections),
        "stage1_total":len(reflections),"experiences_count":len(experiences),
        "model":auth["model"],"usage_stage1":usage1,"usage_stage2":usage2,
    })

if __name__=="__main__":
    main()
