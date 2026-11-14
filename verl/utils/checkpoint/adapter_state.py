"""Per-rank LoRA optimizer, scheduler and complete RNG continuation state."""
from pathlib import Path
import random
import numpy as np
import torch


def save_adapter_training_state(path,optimizer,scheduler):
    data=dict(optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict() if scheduler else None,
        torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        numpy_rng=np.random.get_state(),python_rng=random.getstate())
    temporary=Path(str(path)+".tmp")
    torch.save(data,temporary);temporary.replace(path)


def load_adapter_training_state(path,optimizer,scheduler):
    if not Path(path).exists():return False
    data=torch.load(path,map_location="cpu",weights_only=False)
    optimizer.load_state_dict(data["optimizer"])
    if scheduler and data["scheduler"] is not None:scheduler.load_state_dict(data["scheduler"])
    torch.set_rng_state(data["torch_rng"])
    if data["cuda_rng"] is not None and torch.cuda.is_available():torch.cuda.set_rng_state_all(data["cuda_rng"])
    np.random.set_state(data["numpy_rng"]);random.setstate(data["python_rng"])
    return True


def raise_checkpoint_errors(local_error):
    """Make every rank reject an incomplete checkpoint before advertising success."""
    errors = [local_error]
    if torch.distributed.is_initialized():
        errors = [None] * torch.distributed.get_world_size()
        torch.distributed.all_gather_object(errors, local_error)
    failures = [error for error in errors if error is not None]
    if failures:
        raise RuntimeError("Checkpoint save incomplete: " + " | ".join(failures))
