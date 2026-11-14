"""Track v3 new cohort vs old experiences from snapshots, not global bank mean.

Not a causal estimate. For zero-usage entries, do not count initial U=0
as evidence of positive effect. Handle deleted members as attrition.
"""
import argparse
import json
from pathlib import Path
from statistics import mean


def load(path):
    bank=json.loads(path.read_text())
    return {str(x["experience_id"]):x
            for pool in ("task_experiences","step_experiences")
            for x in bank.get(pool,[])}


def aggregate(rows):
    vals=[float(item.get("utility",0)) for item in rows]
    return {
        "count":len(vals),
        "mean_utility":round(mean(vals),6) if vals else None,
        "positive":sum(v>0 for v in vals),
        "negative":sum(v<0 for v in vals),
        "zero":sum(v==0 for v in vals),
    }


def compare(cohort_manifest:Path, latest_bank:Path):
    manifest=json.loads(cohort_manifest.read_text())
    base=load(Path(manifest["base_bank_path"]))
    current=load(latest_bank)
    new_ids=list(manifest["new_cohort_ids"])
    if not new_ids:
        raise ValueError("empty new cohort")
    present={k:current[k] for k in new_ids if k in current}
    created=manifest["proposed_creation_step"]
    # Do not call a missing cohort "deleted" until it has actually appeared.
    activated=bool(present)
    extra={}
    new_exposed=[]
    for k,item in present.items():
        n=int(item.get("retrieval_count",0) or 0)
        if n>0:
            new_exposed.append(item)
        extra[k]={
            "utility":item.get("utility",0),
            "retrieval_count":n,
            "positive_gain_count":(item.get("evidence") or {}).get("positive_gain_count",0),
            "negative_gain_count":(item.get("evidence") or {}).get("negative_gain_count",0),
        }
    old_present={k:current[k] for k in base if k in current}
    old_exposed=[]
    for k,item in old_present.items():
        prev_n=int(base[k].get("retrieval_count",0) or 0)
        cur_n=int(item.get("retrieval_count",0) or 0)
        if cur_n>prev_n:
            old_exposed.append(item)
    return {
        "status":"ACTIVATED" if activated else "NOT_ACTIVE_IN_LATEST_SNAPSHOT",
        "proposed_creation_step":created,
        "latest_bank":str(latest_bank),
        "new_cohort_ids":new_ids,
        "new_cohort_present":len(present),
        "new_cohort_missing":len(new_ids)-len(present),
        "new_all_surviving":aggregate(list(present.values())),
        "new_with_real_exposures_only":aggregate(new_exposed),
        "new_actual_exposures_count":len(new_exposed),
        "old_with_exposures_since_baseline_only":aggregate(old_exposed),
        "old_actual_exposures_count":len(old_exposed),
        "per_new_experience":extra,
        "warning":"Only active if main training snapshot contains the cohort. This is descriptive Utility, not a controlled causal comparison."
    }


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("snapshot",type=Path)
    p.add_argument("--manifest",type=Path,default=Path("/home/zehao/echo-collab/T069/reflector-v3-results/additive-stage/new_cohort_manifest_from_step105_proposed_step110.json"))
    a=p.parse_args()
    print(json.dumps(compare(a.manifest,a.snapshot),ensure_ascii=False,indent=2))
