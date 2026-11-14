"""Stage additive v3 experiences alongside old bank, never touch live training.

Source: exact saved 249 bank and isolated real API results.
Output: candidate bank + a cohort-ID manifest in a separate staging dir.
"""
from __future__ import annotations
import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent_system.memory.experience_memory import ExperienceMemory
from agent_system.tree.training import write_new_experiences


def snapshot_items(bank):
    return {
        str(x["experience_id"]): deepcopy(x)
        for pool in ("task_experiences", "step_experiences")
        for x in bank.get(pool, [])
    }


def stage(base_path: Path, results_path: Path, out_dir: Path, proposed_step: int):
    bank = json.loads(base_path.read_text())
    api = json.loads(results_path.read_text())
    assert api.get("request_and_parse_succeeded"), "The API request did not parse cleanly"
    new = api.get("experiences", [])
    assert 0 < len(new) <= 5, "Expect 1..5 actual API proposals"
    for e in new:
        assert e.get("experience_type") in ("task", "step")
        assert e.get("evidence", {}).get("source_traj_uids"), "No source trajectory evidence"
    m = ExperienceMemory(load_initial_experiences=False)
    m.experiences = deepcopy(bank)
    m._normalize_all_experience_meta()
    old = snapshot_items(m.experiences)
    metrics = write_new_experiences(m, new, step=proposed_step)
    both = snapshot_items(m.experiences)
    ids = sorted(set(both) - set(old))
    assert len(old) + len(ids) == len(both), "Not purely additive"
    assert all(both[k] == old[k] for k in old), "Old experiences altered"
    assert metrics["bank/added"] == len(ids)
    assert all(float(both[k]["utility"]) == 0.0 for k in ids)
    assert all(int(both[k]["created_at_step"]) == proposed_step for k in ids)
    manifest = {
        "mode": "STAGING_ONLY_NOT_LIVE",
        "requires_rebase_if_live_bank_changes": True,
        "base_bank_path": str(base_path),
        "real_api_result": str(results_path),
        "proposed_creation_step": proposed_step,
        "old_count": len(old),
        "new_count": len(ids),
        "combined_count": len(both),
        "new_cohort_ids": ids,
        "existing_ids_unchanged": True,
        "new_experiences": [
            {"experience_id": k, "utility": both[k].get("utility"),
             "retrieval_count": both[k].get("retrieval_count"),
             "trigger": both[k].get("trigger"),
             "recommended_action": both[k].get("recommended_action")}
            for k in ids
        ],
        "note": "Not active in PID 3699760. Rebase on the latest live bank before any future activation; IDs can change. Requires an authorized in-process entry point or restart to load."
    }
    out_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    source_step = base_path.stem.split("step")[-1]
    out_bank = out_dir / f"staged_additive_bank_from_step{source_step}_proposed_step{proposed_step}.json"
    out_manifest = out_dir / f"new_cohort_manifest_from_step{source_step}_proposed_step{proposed_step}.json"
    out_bank.write_text(json.dumps(m.experiences, ensure_ascii=False, indent=2))
    out_manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    out_bank.chmod(0o600)
    out_manifest.chmod(0o600)
    print(json.dumps({
        "staged_bank": str(out_bank),
        "cohort_manifest": str(out_manifest),
        "old": len(old), "new": len(ids), "total": len(both),
        "added_task": metrics["reflector/added_task"],
        "added_step": metrics["reflector/added_step"],
        "new_ids": ids, "live_mutated": False,
    }, ensure_ascii=False))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--base", type=Path, default=Path("/home/zehao/echo-collab/T069/tv2/checkpoints/experience_bank_step105.json"))
    p.add_argument("--result", type=Path, default=Path("/home/zehao/echo-collab/T069/reflector-v3-results/result-step105.json"))
    p.add_argument("--dest", type=Path, default=Path("/home/zehao/echo-collab/T069/reflector-v3-results/additive-stage"))
    p.add_argument("--step", type=int, default=110)
    a = p.parse_args()
    stage(a.base, a.result, a.dest, a.step)
