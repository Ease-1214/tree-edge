"""T70 training-side validation on the *same four Actor GPUs* as training.

Dev1000 uses the existing trainer val_dataloader. Test6125 uses a second
dataloader and temporarily swaps it into the same _validate() call, which
already performs greedy no-experience multi-turn Search rollout and strict EM.
No external evaluator, independent GPU, or asynchronous queue is required.
"""
from pathlib import Path


def configure_t70_test_dataloader(trainer, collate_fn):
    path = trainer.config.trainer.get("t70_test6125_file")
    if not path:
        return
    if not Path(path).is_file():
        raise FileNotFoundError(f"T70 6125 validation dataset missing: {path}")
    from torchdata.stateful_dataloader import StatefulDataLoader
    from verl.trainer.main_ppo import create_rl_dataset
    dataset = create_rl_dataset(
        str(path), trainer.config.data, trainer.tokenizer, trainer.processor,
    )
    if len(dataset) != 6125:
        raise ValueError(f"T70 test6125 must have 6125 rows, got {len(dataset)}")
    trainer.t70_test6125_dataset = dataset
    size = trainer.config.data.get("val_batch_size") or 128
    trainer.t70_test6125_dataloader = StatefulDataLoader(
        dataset=dataset,
        batch_size=int(size),
        num_workers=0,
        shuffle=False,
        drop_last=False,
        collate_fn=collate_fn,
    )
    print(f"T70_INPROCESS_TEST6125_READY questions={len(dataset)} batch_size={size}",flush=True)


def run_t70_test6125(trainer):
    """Sequential and synchronous, with the training worker group reused."""
    loader = getattr(trainer, "t70_test6125_dataloader", None)
    if loader is None:
        return {}
    original = trainer.val_dataloader
    try:
        trainer.val_dataloader = loader
        print(f"T70_INPROCESS_TEST6125_BEGIN step={trainer.global_steps}",flush=True)
        metrics = trainer._validate()
    finally:
        trainer.val_dataloader = original
    renamed = {
        ("test6125/" + key[4:] if key.startswith("val/") else "test6125/" + key): value
        for key,value in metrics.items()
    }
    renamed["test6125/checkpoint_step"] = trainer.global_steps
    print(f"T70_INPROCESS_TEST6125_COMPLETE step={trainer.global_steps} EM={renamed.get('test6125/strict_em')}",flush=True)
    return renamed


def t70_test6125_due(trainer, *, is_last_step=False):
    if not getattr(trainer, "t70_test6125_dataloader", None):
        return False
    interval = int(trainer.config.trainer.get("t70_test6125_freq", 50))
    return interval > 0 and trainer.global_steps > 0 and (
        is_last_step or trainer.global_steps % interval == 0
    )
