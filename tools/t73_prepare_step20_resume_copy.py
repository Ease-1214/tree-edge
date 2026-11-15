"""Prepare isolated T73 Step20 resume without changing any running process.

Only succeeds if the original checkpoint tracker AND every file are complete.
NO GPU allocation, NO source checkpoint modification and NO service stop.
"""
import json
import os
import shutil
import subprocess
from pathlib import Path

from t73_rewrite_bank_for_resume import rewrite_bank

ROOT=Path("/home/zehao/echo-collab/T073")
OLD=ROOT/"qwen25-3b-r64a128-a40x2-full11015-20261115"
NEW=ROOT/"qwen25-3b-r64a128-a40x2-classification-resume20-20261115"
STEP=20
REQUIRED=[
    "actor/lora_adapter/adapter_model.safetensors",
    "actor/lora_adapter/adapter_config.json",
    "actor/training_state_rank_0.pt",
    "actor/training_state_rank_1.pt",
    "data.pt",
    "tree_state.json",
]


def main():
    if NEW.exists():
        raise FileExistsError(f"Refusing to replace existing prepared copy: {NEW}")
    src=OLD/"checkpoints"/f"global_step_{STEP}"
    latest=OLD/"checkpoints/latest_checkpointed_iteration.txt"
    if not latest.exists() or latest.read_text().strip()!="20":
        raise RuntimeError("Original Step20 checkpoint not yet advertised as complete")
    missing=[x for x in REQUIRED if not (src/x).is_file() or (src/x).stat().st_size<128]
    if missing:raise RuntimeError("Incomplete Step20: "+repr(missing))
    tree=json.loads((src/"tree_state.json").read_text())
    if tree.get("checkpoint_step")!=20:
        raise RuntimeError("Step20 tree checkpoint identity mismatch")
    original_bank=tree.get("bank",{})
    revised_bank, changes=rewrite_bank(original_bank)
    if not any(x["experience_id"]=="task_000012" for x in changes):
        print("INFO: task_000012 absent from current bank, likely evicted")
    if not any(x["experience_id"]=="task_000016" for x in changes):
        print("INFO: task_000016 absent from current bank, likely evicted")

    # Avoid saving the same data before basic validation. No hardlinks to any
    # live checkpoint: every byte is copied independently.
    new_stage=Path(str(NEW)+".preparing")
    if new_stage.exists():
        raise FileExistsError(f"Existing incomplete staging directory: {new_stage}")
    try:
        new_step=new_stage/"checkpoints"/f"global_step_{STEP}"
        new_step.parent.mkdir(parents=True,exist_ok=False)
        shutil.copytree(src,new_step,symlinks=False)
        shutil.copy2(latest,new_stage/"checkpoints/latest_checkpointed_iteration.txt")
        tree["bank"]=revised_bank
        tpath=new_step/"tree_state.json"
        ttemp=tpath.with_suffix(".json.new")
        ttemp.write_text(json.dumps(tree,ensure_ascii=False,indent=2))
        ttemp.replace(tpath)
        bp=OLD/"checkpoints"/f"experience_bank_step{STEP}.json"
        if bp.is_file():
            old_bank=json.loads(bp.read_text())
            patched_bank, bank_changes=rewrite_bank(old_bank)
            if {x["experience_id"] for x in bank_changes}!={x["experience_id"] for x in changes}:
                raise ValueError("Bank snapshot and tree_state have different changed IDs")
            (new_stage/"checkpoints"/f"experience_bank_step{STEP}.json").write_text(
                json.dumps(patched_bank,ensure_ascii=False,indent=2)
            )
        report={
            "status":"prepared_not_started",
            "source_checkpoint":str(src),
            "prepared_checkpoint":str(NEW/"checkpoints"/f"global_step_{STEP}"),
            "step":STEP,
            "modified":changes,
            "original_unchanged":True,
            "optimizer_rank_count":2,
            "restart_requires_manual_gpu_queue_safety_check":True,
            "wandb_run_id":"t73-qwen25-3b-r64a128-a40x2-full11015-20261115",
            "wandb_resume":"must",
            "source_git_branch":"fix/t73-reflector-type-validation",
        }
        (new_stage/"resume_preparation_manifest.json").write_text(json.dumps(report,ensure_ascii=False,indent=2))
        # recheck all source and staged checkpoints agree structurally, and the
        # legacy per-rank optimizer state files still have identical sizes.
        for x in REQUIRED:
            if x=="tree_state.json":continue
            if (src/x).stat().st_size!=(new_step/x).stat().st_size:
                raise RuntimeError("Incomplete file copy: "+x)
        os.replace(new_stage,NEW)
        print("STAGED_READY",NEW,flush=True)
        print("REWRITTEN_IDS",[x["experience_id"] for x in changes],flush=True)
    except BaseException:
        # leave staging for forensic investigation, do not delete anything.
        raise

if __name__=="__main__":main()
