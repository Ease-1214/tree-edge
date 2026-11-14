#!/usr/bin/env python3
"""Rebuild retrieval_count / last_retrieval_step from saved trees (T066 fix).

    python scripts/t066_recount_retrieval.py \
        --tree-dir <run>/tree-private --state <run>/checkpoints/global_step_20/tree_state.json

Reads tree-step{1..S}.pt for S = checkpoint step, recomputes exposure counts with the same
rule the trainer now uses (one count per RIGHT-node injection), and rewrites the bank in
tree_state.json in place (a .bak copy is kept). Prints a before/after summary.
"""
import argparse
import json
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from agent_system.tree.training import apply_exposure_counts, tree_exposure_counts  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tree-dir", required=True)
    ap.add_argument("--state", required=True)
    args = ap.parse_args()
    state_path = Path(args.state)
    state = json.loads(state_path.read_text())
    last = int(state["checkpoint_step"])
    bank = state["bank"]
    before = {
        x["experience_id"]: int(x.get("retrieval_count", 0) or 0)
        for p in ("task_experiences", "step_experiences") for x in bank.get(p, [])
    }
    for item in [x for p in ("task_experiences", "step_experiences") for x in bank.get(p, [])]:
        item["retrieval_count"] = 0
        item["last_retrieval_step"] = None
    totals = defaultdict(int)
    for step in range(1, last + 1):
        path = Path(args.tree_dir) / f"tree-step{step}.pt"
        if not path.exists():
            raise FileNotFoundError(path)
        tree = torch.load(path, weights_only=False)
        counts = tree_exposure_counts(tree)
        apply_exposure_counts(bank, counts, step)
        for k, v in counts.items():
            totals[k] += v
    shutil.copy(state_path, state_path.with_suffix(".json.bak"))
    state_path.write_text(json.dumps(state, indent=2))
    after = sorted(((int(x.get("retrieval_count", 0)), x["experience_id"])
                    for p in ("task_experiences", "step_experiences") for x in bank.get(p, [])), reverse=True)
    print(f"steps 1..{last}: total exposures {sum(totals.values())}, "
          f"sum before {sum(before.values())}, experiences {len(after)}")
    print("top 10:", after[:10])
    print("never exposed:", [eid for n, eid in after if n == 0])


if __name__ == "__main__":
    main()
