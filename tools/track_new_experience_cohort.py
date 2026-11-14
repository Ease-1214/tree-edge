"""Report only a newly-created experience cohort's post-creation Utility.

Uses exact snapshot identities; never infers value from all-bank mean.
Intended for repeated inspection of a real training run, not a causal
claim about Actor improvement.
"""
import argparse
import json
from pathlib import Path
from statistics import mean


def as_ids(bank):
    return {
        str(item["experience_id"]): item
        for pool in ("task_experiences", "step_experiences")
        for item in bank.get(pool, [])
    }


def report(checkpoint_dir: Path, created_step: int, observed_step: int):
    before = as_ids(json.loads((checkpoint_dir / f"experience_bank_step{created_step - 5}.json").read_text()))
    created = as_ids(json.loads((checkpoint_dir / f"experience_bank_step{created_step}.json").read_text()))
    latest = as_ids(json.loads((checkpoint_dir / f"experience_bank_step{observed_step}.json").read_text()))
    cohort_ids = sorted(set(created).difference(before))
    surviving = {eid:latest[eid] for eid in cohort_ids if eid in latest}
    utility = [float(item.get("utility", 0.0)) for item in surviving.values()]
    return {
        "created_step": created_step,
        "observed_step": observed_step,
        "created_count": len(cohort_ids),
        "surviving_count": len(surviving),
        "deleted_count": len(cohort_ids) - len(surviving),
        "mean_utility_of_surviving_new_experiences": mean(utility) if utility else None,
        "positive_utility_count": sum(x > 0 for x in utility),
        "zero_utility_count": sum(x == 0 for x in utility),
        "negative_utility_count": sum(x < 0 for x in utility),
        "experiences": [{
            "id": eid, "status": "surviving" if eid in latest else "deleted",
            "utility": latest[eid].get("utility") if eid in latest else None,
            "retrieval_count": latest[eid].get("retrieval_count") if eid in latest else None,
            "last_retrieval_step": latest[eid].get("last_retrieval_step") if eid in latest else None,
            "positive_gain_count": (latest[eid].get("evidence") or {}).get("positive_gain_count") if eid in latest else None,
            "negative_gain_count": (latest[eid].get("evidence") or {}).get("negative_gain_count") if eid in latest else None,
        } for eid in cohort_ids],
        "note": "Utility is a noisy observational signal, not causal proof; do not treat initial zeros or deletions as positive improvement."
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("creation_step", type=int)
    parser.add_argument("observed_step", type=int)
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("/home/zehao/echo-collab/T069/tv2/checkpoints"))
    a = parser.parse_args()
    print(json.dumps(report(a.checkpoint_dir, a.creation_step, a.observed_step), indent=2, ensure_ascii=False))
