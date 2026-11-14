"""Read-only 5-step candidate pooling experiment on saved trees.

Compare against single-step 40-candidate preselection without changing the
running trainer or bank. Same API selector, same final complete-path Reflector.
"""
import json
import os
import sys
from pathlib import Path
from copy import deepcopy
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["OMP_NUM_THREADS"] = "1"
for key in ("ALL_PROXY","all_proxy","HTTP_PROXY","http_proxy","HTTPS_PROXY","https_proxy"):
    os.environ.pop(key, None)
import torch
from agent_system.tree.core import trajectory_reflector_examples
from agent_system.tree.reflector_case_selector_v3 import select_reflector_examples_with_api
from agent_system.memory.experience_updater import ExperienceUpdater
from tools.reflector_v3_real_test import model_config, private_save


def build_five_step_pool(end_step):
    root=Path("/home/zehao/echo-collab/T069/tv2/tree-private")
    pool=[];seen=set()
    by_step={}
    for step in range(end_step-4,end_step+1):
        tree=torch.load(root/f"tree-step{step}.pt",map_location="cpu",weights_only=False)
        candidates=trajectory_reflector_examples(tree,k=25,seed=step)
        selected=[]
        for ex in candidates:
            key=(ex.get("data_source"), ex["task"].strip().casefold())
            if key in seen: continue
            seen.add(key);selected.append(ex)
            if len(selected)>=8:break
        by_step[step]=len(selected)
        pool.extend(selected)
        del tree
    return pool,by_step


def run(step):
    out=Path("/home/zehao/echo-collab/T069/reflector-v3-results")
    bank=json.loads(Path(f"/home/zehao/echo-collab/T069/tv2/checkpoints/experience_bank_step{step}.json").read_text())
    start=time.monotonic()
    pool,by_step=build_five_step_pool(step)
    print("WINDOW_POOL",len(pool),"BY_STEP",by_step,"SECONDS",round(time.monotonic()-start,1),flush=True)
    assert len(pool)>=30
    private_save(out/f"window-pool-step{step}.private.json",json.dumps(pool,ensure_ascii=False,indent=2))
    updater=ExperienceUpdater(**model_config(),max_new_experiences_per_update=5,max_completion_tokens=8192)
    selected=select_reflector_examples_with_api(updater,pool,bank,k=10)
    selection_status=updater.last_selector_status
    print("SELECTED",len(selected),"PROVENANCE",[
        e["fork_id"].split(":")[0] for e in selected
    ],"ROOT",sum(e["fork_depth"]==1 for e in selected),flush=True)
    private_save(out/f"window-selected-step{step}.private.json",json.dumps(selected,ensure_ascii=False,indent=2))
    prompt=updater._build_trajectory_pair_prompt(selected,bank,1)
    private_save(out/f"window-prompt-step{step}.private.txt",prompt)
    original_parse=updater._parse_experiences_response
    def capture(raw):
        private_save(out/f"window-raw-step{step}.private.txt",str(raw))
        return original_parse(raw)
    updater._parse_experiences_response=capture
    start=time.monotonic()
    experiences=updater.analyze_failures(selected,bank)
    report={
        "step":step,"by_step":by_step,"pool":len(pool),
        "selector_status":selection_status,
        "prompt_chars":len(prompt),"n_experiences":len(experiences),
        "types":[x["experience_type"] for x in experiences],
        "duration_s":round(time.monotonic()-start,2),
        "success":updater.last_update_succeeded,
        "failure_reason":updater.last_failure_reason,
        "experiences":experiences,
    }
    private_save(out/f"window-result-step{step}.json",json.dumps(report,ensure_ascii=False,indent=2))
    print("WINDOW_RESULT",json.dumps(report,ensure_ascii=False),flush=True)


if __name__=="__main__":
    run(int(sys.argv[1]) if len(sys.argv)>1 else 100)
