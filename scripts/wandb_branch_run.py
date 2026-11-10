#!/usr/bin/env python3
"""Branch a W&B run at a checkpoint step, so a restart from that step continues cleanly.

Why: W&B ignores logged steps <= the last step already in a run. Restarting training
from checkpoint step S after the run already logged S+1..T would silently drop the new
S+1..T. Instead, create a NEW run that contains history 1..S and let training resume it.

    # 1) try W&B's native fork (keeps history 1..S server-side)
    python scripts/wandb_branch_run.py --entity E --project P --run OLD_ID --step 20 \
        --name t066-tree-tier2-s0 --group t066 --tags tree,tier2,beta0.3,full200 --mode fork
    # 2) if the server rejects fork_from, copy history 1..S into a new run
    python ... --mode copy
    # prints NEW_ID; then start training with
    #   WANDB_RUN_ID=NEW_ID WANDB_RESUME=must   (same project / name / group / tags)

The old run is kept. By default it moves to "<group>-superseded".
With --keep-source it stays in the comparison group and gets question_grouped.
"""
import argparse

import wandb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--entity", required=True)
    ap.add_argument("--project", required=True)
    ap.add_argument("--run", required=True, help="old run id")
    ap.add_argument("--step", type=int, required=True, help="checkpoint step to branch at")
    ap.add_argument("--name", required=True)
    ap.add_argument("--group", required=True)
    ap.add_argument("--tags", default="")
    ap.add_argument("--notes", default="")
    ap.add_argument("--mode", choices=["fork", "copy"], required=True)
    ap.add_argument("--keep-source", action="store_true", help="Keep the source as a comparison run")
    args = ap.parse_args()
    tags = [t for t in args.tags.split(",") if t]
    api = wandb.Api()
    old = api.run(f"{args.entity}/{args.project}/{args.run}")
    notes = args.notes or f"branched from {args.run} at step {args.step}"

    if args.mode == "fork":
        run = wandb.init(entity=args.entity, project=args.project, name=args.name,
                         group=args.group, tags=tags, notes=notes,
                         fork_from=f"{args.run}?_step={args.step}")
    else:
        run = wandb.init(entity=args.entity, project=args.project, name=args.name,
                         group=args.group, tags=tags, notes=notes, config=dict(old.config))
        copied = 0
        for row in old.scan_history():
            step = row.get("_step")
            if step is None or step > args.step:
                continue
            data = {k: v for k, v in row.items() if not k.startswith("_") and v is not None}
            if data:
                run.log(data, step=int(step))
                copied += 1
        print(f"copied {copied} history rows (steps <= {args.step})")
    new_id = run.id
    run.finish()

    old.group = args.group if args.keep_source else args.group + "-superseded"
    old.tags = sorted(set(list(old.tags) + (["question_grouped"] if args.keep_source else ["superseded"])))
    old.update()
    print(f"NEW_RUN_ID={new_id}")


if __name__ == "__main__":
    main()
