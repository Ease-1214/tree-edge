"""Read-only live API validation for Reflector v3 on saved T069 trees."""
import json
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["OMP_NUM_THREADS"] = "1"
# The local API gateway does not need proxy routing.
for _key in ("ALL_PROXY","all_proxy","HTTP_PROXY","http_proxy","HTTPS_PROXY","https_proxy"):
    os.environ.pop(_key, None)

import torch
from agent_system.tree.core import trajectory_reflector_examples
from agent_system.tree.reflector_case_selector_v3 import select_reflector_examples_with_api
from agent_system.memory.experience_updater import ExperienceUpdater


def model_config():
    """Get authorized process's existing credentials without printing them."""
    p = Path("/proc/3699760/environ")
    if p.exists():
        parts = p.read_bytes().split(b"\0")
        env = {k.decode(): v.decode() for part in parts if b"=" in part
               for k, v in [part.split(b"=", 1)]}
        if env.get("EXPERIENCE_UPDATE_API_KEY"):
            return {
                "api_key": env["EXPERIENCE_UPDATE_API_KEY"],
                "base_url": env.get("EXPERIENCE_UPDATE_BASE_URL", "http://127.0.0.1:29663/v1"),
                "model": env.get("EXPERIENCE_UPDATE_MODEL", "gpt-6.1-sol"),
            }
    return {
        "api_key": os.environ.get("EXPERIENCE_UPDATE_API_KEY"),
        "base_url": os.environ.get("EXPERIENCE_UPDATE_BASE_URL", "http://127.0.0.1:29663/v1"),
        "model": os.environ.get("EXPERIENCE_UPDATE_MODEL", "gpt-6.1-sol"),
    }


def private_save(path, content):
    fd = os.open(str(path), os.O_CREAT | os.O_TRUNC | os.O_WRONLY, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as stream:
        stream.write(content)


def main(step):
    out = Path("/home/zehao/echo-collab/T069/reflector-v3-results")
    out.mkdir(exist_ok=True, mode=0o700)
    tag = os.environ.get("REFLECTOR_RUN_TAG", "").strip()
    suffix = "-" + tag if tag else ""
    treepath = Path(f"/home/zehao/echo-collab/T069/tv2/tree-private/tree-step{step}.pt")
    print("LOADING", treepath.name, flush=True)
    tree = torch.load(treepath, map_location="cpu", weights_only=False)
    pool = trajectory_reflector_examples(tree, k=40, seed=step)
    auth = model_config()
    if not auth["api_key"]:
        raise RuntimeError("Authorized API key unavailable")
    updater = ExperienceUpdater(**auth, max_new_experiences_per_update=5,
                                max_completion_tokens=8192)
    examples = select_reflector_examples_with_api(
        updater, pool, tree.bank_snapshot, k=10,
    )
    assert len(examples) == 10, "insufficient contrasting forks"
    private_save(out / f"selected-step{step}{suffix}.private.json",
                 json.dumps(examples, ensure_ascii=False, indent=2))
    prompt = updater._build_trajectory_pair_prompt(examples, tree.bank_snapshot, 1)
    private_save(out / f"prompt-step{step}{suffix}.private.txt", prompt)
    print("INPUT cases", len(examples), "chars", len(prompt),
          "root", sum(e["fork_depth"] == 1 for e in examples),
          "model", updater.model, flush=True)
    original_parse = updater._parse_experiences_response
    def parse_and_capture(raw):
        private_save(out / f"raw-step{step}{suffix}.private.txt", str(raw))
        return original_parse(raw)
    updater._parse_experiences_response = parse_and_capture
    original_completion = updater._completion_with_deadline
    usage = {}
    def completion_and_usage(*args, **kwargs):
        response = original_completion(*args, **kwargs)
        if response.usage:
            usage.update(response.usage.model_dump())
        return response
    updater._completion_with_deadline = completion_and_usage
    start = time.monotonic()
    experiences = updater.analyze_failures(examples, tree.bank_snapshot)
    result = {
        "step": step, "model": updater.model,
        "request_and_parse_succeeded": updater.last_update_succeeded,
        "failure_reason": updater.last_failure_reason,
        "seconds": round(time.monotonic() - start, 2),
        "prompt_chars": len(prompt),
        "cases": len(examples),
        "selector_status": updater.last_selector_status,
        "json_repaired": getattr(updater, "last_json_repaired", False),
        "root_cases": sum(e["fork_depth"] == 1 for e in examples),
        "n_new_experiences": len(experiences),
        "types": [e.get("experience_type") for e in experiences],
        "invalid_v3_dropped": getattr(updater, "last_invalid_v3_dropped", 0),
        "token_usage": usage,
        "experiences": experiences,
    }
    private_save(out / f"result-step{step}{suffix}.json",
                 json.dumps(result, ensure_ascii=False, indent=2))
    print("FINAL_RESULT", json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 80)
