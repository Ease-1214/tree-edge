#!/usr/bin/env python3
"""T71 Step42 fixed-seed 10-fork two-stage reflection with small API batches."""
import json, os, sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import tools.t71_reflection_two_stage_probe as base
from openai import OpenAI
import httpx

def main():
    cases=json.loads((base.ROOT/"01_selected_cases.json").read_text())
    assert len(cases)==10
    auth=base.load_auth()
    client=OpenAI(api_key=auth["api_key"],base_url=auth["base_url"],max_retries=0,
      http_client=httpx.Client(trust_env=False, timeout=360))
    all_reflections=[]
    all_usage=[]
    for idx in range(0,len(cases),2):
        number=idx//2+1
        group=cases[idx:idx+2]
        prompt=(base.STAGE1.replace("exactly 10 real","2 real")
          .replace("exactly 10 objects","exactly 2 objects")
          +"\n\nCASES:\n"+json.dumps(group,ensure_ascii=False,default=str))
        base.save(f"batch{number:02d}_prompt.txt",prompt)
        print("BATCH",number,"CHARS",len(prompt),flush=True)
        try:
            raw,usage=base.call(client,auth["model"],prompt,5500)
            base.save(f"batch{number:02d}_raw.txt",raw)
            reflections=base.parse_json(raw)
            base.save(f"batch{number:02d}_reflections.json",reflections)
            all_reflections.extend(reflections)
            all_usage.append(usage)
            print("BATCH",number,"SUCCESS",len(reflections),
                  "VALID",sum(x.get("status")=="valid" for x in reflections),flush=True)
        except Exception as exc:
            base.save(f"batch{number:02d}_error.txt",
                      type(exc).__name__+": "+str(exc)[:1000])
            print("BATCH",number,"FAILED",type(exc).__name__,str(exc)[:250],flush=True)
            return
    base.save("04_stage1_reflections.json",all_reflections)
    print("STAGE1_TOTAL",len(all_reflections),flush=True)
    stage2=(base.STAGE2+"\n\nREFLECTIONS:\n"+
            json.dumps(all_reflections,ensure_ascii=False))
    base.save("05_stage2_prompt.txt",stage2)
    raw,usage2=base.call(client,auth["model"],stage2,7500)
    base.save("06_stage2_raw.txt",raw)
    experiences=base.parse_json(raw)
    base.save("07_stage2_experiences.json",experiences)
    base.save("08_summary.json",{
        "seed":71,"sample_count":10,"batch_size":2,
        "model":auth["model"],"stage1_valid":sum(x.get("status")=="valid" for x in all_reflections),
        "experiences_count":len(experiences),"usage_stage1":all_usage,"usage_stage2":usage2})
    print("STAGE2_EXPERIENCES",len(experiences),flush=True)

if __name__=="__main__":
    main()
