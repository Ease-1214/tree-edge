"""Offline replay: audit T73 newly generated experience labels without changing training.

Usage: PYTHONPATH=<worktree> python tools/t73_offline_validate_experiences.py
Requires existing Reflector API env and a running forwarding endpoint.
"""
import json
from datetime import datetime, timezone
from pathlib import Path

from agent_system.memory.experience_type_validation import validate_experience_types
from agent_system.memory.experience_updater import ExperienceUpdater


def main():
    run = Path("/home/zehao/echo-collab/T073/qwen25-3b-r64a128-a40x2-full11015-20261115")
    items = []
    for step in (5, 10, 15):
        data = json.loads((run / "checkpoints" / f"experience_bank_step{step}.json").read_text())
        for pool in ("task_experiences", "step_experiences"):
            for experience in data.get(pool, []):
                if experience.get("created_at_step") == step:
                    item = dict(experience)
                    item["experience_type"] = "task" if pool == "task_experiences" else "step"
                    items.append(item)
    print("OFFLINE_REPLAY_INPUT", len(items), "task", sum(x["experience_type"] == "task" for x in items),
          "step", sum(x["experience_type"] == "step" for x in items), flush=True)
    updater = ExperienceUpdater()
    kept, decisions = validate_experience_types(
        items, updater._completion_with_deadline, model=updater.model,
        token_limit_param=updater.token_limit_param,
        max_completion_tokens=updater.max_completion_tokens,
    )
    by_index = {i+1: item for i, item in enumerate(items)}
    details = [{
        "experience_id": by_index[d["index"]]["experience_id"],
        "created_at_step": by_index[d["index"]]["created_at_step"],
        "trigger": by_index[d["index"]]["trigger"],
        **d,
    } for d in decisions]
    report = {
        "kind": "offline_readonly_real_reflector_api_check",
        "input_count": len(items),
        "accepted_count": len(kept),
        "rejected_count": len(items) - len(kept),
        "decisions": details,
    }
    out = Path(__file__).resolve().parents[1] / "reports/t73_reflector_type_replay.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    for item in details:
        print("REPLAY_LABEL", item["experience_id"],
              "step", item["created_at_step"],
              "declared", item["declared_type"],
              "inferred", item["inferred_type"],
              "accepted", item["accepted"],
              "reason", item["reason"], flush=True)
    print("REPLAY_REPORT", out, flush=True)


if __name__ == "__main__":
    main()
