"""Rewrite verified Task trigger mistakes only in an isolated checkpoint copy.

Never alter a live experience bank or the original checkpoint. Preserve
experience IDs, utility, exposure counters, lifetime, evidence and metadata.
"""
import argparse
import copy
import json
from pathlib import Path


REVISIONS = {
    "task_000012": {
        "trigger": (
            "The question asks who sang or performed a named song or a specified version "
            "of a song, so the requested performer role must be distinguished from "
            "songwriting or other recordings."
        ),
        "state_cues": [
            "The question names a song or recording.",
            "The question asks who sang or performed it.",
            "The question may specify an original recording, version, or performer role.",
        ],
        "retrieval_obs": (
            "When answering who performed a song, identify the performer relationship "
            "specified by the question. Search the song title and check whether "
            "retrieved evidence identifies an original performer, cover artist, or "
            "songwriter; return only the matching performer."
        ),
    },
    "task_000016": {
        "trigger": (
            "The question asks for a biographical or historical attribute of a person "
            "identified directly or through a related work, place, or institution."
        ),
        "state_cues": [
            "The question asks for an attribute of a person, such as nationality.",
            "The question identifies that person directly or with contextual clues.",
            "Related works, locations or institutions may be mentioned in the question.",
        ],
        "retrieval_obs": (
            "For a personal-attribute question, retrieve the person and the exact "
            "requested attribute. Check that evidence assigns the attribute to "
            "that person, rather than an associated place, work or institution."
        ),
    },
}
EDITABLE = {"title", "trigger", "state_cues", "when_to_apply", "retrieval_obs"}
PROTECTED = {
    "experience_id", "utility", "retrieval_count", "last_retrieval_step",
    "created_at_step", "evidence", "task_type", "principle",
    "recommended_action", "avoid_action", "expected_effect",
}


def rewrite_bank(bank):
    bank=copy.deepcopy(bank)
    report=[]
    for pool in ("task_experiences", "step_experiences"):
        for item in bank.get(pool, []):
            identifier=item.get("experience_id")
            if identifier not in REVISIONS:
                continue
            if pool!="task_experiences":
                raise ValueError(f"{identifier} moved out of Task pool unexpectedly")
            before=copy.deepcopy(item)
            revision=REVISIONS[identifier]
            item["trigger"]=revision["trigger"]
            item["title"]=revision["trigger"]
            item["state_cues"]=revision["state_cues"]
            item["retrieval_obs"]=revision["retrieval_obs"]
            item["when_to_apply"]=(
                f"Task type: {item['task_type']}. " + revision["trigger"]
                + ". " + " ".join(revision["state_cues"])
            )
            assert set(item)==set(before),identifier
            changed={k for k in item if item[k]!=before[k]}
            assert changed.issubset(EDITABLE), (identifier,changed)
            assert all(item[k]==before[k] for k in PROTECTED),identifier
            report.append({
                "experience_id":identifier,
                "pool":pool,
                "changed_fields":sorted(changed),
                "original_trigger":before["trigger"],
                "revised_trigger":item["trigger"],
                "preserved_utility":before["utility"],
                "preserved_retrieval_count":before["retrieval_count"],
                "preserved_created_at_step":before["created_at_step"],
            })
    return bank,report


def apply(tree_state, output, experience_bank=None):
    tree_state=Path(tree_state).resolve()
    output=Path(output).resolve()
    if tree_state==output:
        raise ValueError("Never overwrite source checkpoint")
    if output.exists():
        raise FileExistsError(output)
    doc=json.loads(tree_state.read_text())
    if int(doc.get("checkpoint_step",-1)) not in (15,20):
        raise ValueError("Only offline Step15 test / Step20 resume snapshots supported")
    before_bank=doc["bank"]
    new_bank,changes=rewrite_bank(before_bank)
    doc["bank"]=new_bank
    output.parent.mkdir(parents=True,exist_ok=True)
    tmp=output.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(doc,ensure_ascii=False,indent=2))
    tmp.replace(output)
    checkpoint=doc["checkpoint_step"]
    report={"checkpoint_step":checkpoint,"modified":changes,
            "present_ids":[
                x.get("experience_id")
                for p in ("task_experiences","step_experiences")
                for x in before_bank.get(p,[])
                if x.get("experience_id") in REVISIONS
            ],
            "missing_ids":[x for x in REVISIONS if x not in {z["experience_id"] for z in changes}],
            "bank_count_before":sum(len(before_bank.get(p,[])) for p in ("task_experiences","step_experiences")),
            "bank_count_after":sum(len(new_bank.get(p,[])) for p in ("task_experiences","step_experiences")),
            "changed_only_new_copy":True}
    if experience_bank:
        bankp=Path(experience_bank)
        if bankp.exists():
            bankdata=json.loads(bankp.read_text())
            fixed,diff=rewrite_bank(bankdata)
            if len(diff)!=len(changes):
                raise ValueError("Experience-bank snapshot content differs from tree-state bank")
            patched=bankp.with_name(bankp.stem+"_revised"+bankp.suffix)
            if patched.exists():
                raise FileExistsError(patched)
            patched.write_text(json.dumps(fixed,ensure_ascii=False,indent=2))
            report["revised_bank_copy"]=str(patched)
    report_path=output.parent.parent.parent/"t73_rewrite_manifest.json"
    report_path.write_text(json.dumps(report,ensure_ascii=False,indent=2))
    print(json.dumps(report,ensure_ascii=False,indent=2))
    return report


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--src",required=True)
    parser.add_argument("--dst",required=True)
    parser.add_argument("--bank")
    args=parser.parse_args()
    apply(args.src,args.dst,args.bank)


if __name__=="__main__":
    main()
