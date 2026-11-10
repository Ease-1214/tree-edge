from contextlib import nullcontext
from types import SimpleNamespace
import torch
import pytest
from safetensors.torch import save_file
import verl.utils.checkpoint.fsdp_checkpoint_manager as module


def test_adapter_only_resume_identity_and_scheduler(tmp_path,monkeypatch):
    destination=tmp_path/"global_step_20/actor/lora_adapter";destination.mkdir(parents=True)
    expected=torch.arange(4,dtype=torch.float32).reshape(2,2)
    save_file({"base_model.layer.lora_A.weight":expected},str(destination/"adapter_model.safetensors"))
    parameter=torch.nn.Parameter(torch.zeros(2,2))
    model=SimpleNamespace(named_parameters=lambda:[("_fsdp_wrapped_module.base_model.layer.lora_A.default.weight",parameter)])
    manager=object.__new__(module.FSDPCheckpointManager);manager.model=model;manager.world_size=1;manager.rank=0
    steps=[];manager.lr_scheduler=SimpleNamespace(step=lambda value:steps.append(value))
    monkeypatch.setattr(module.FSDP,"summon_full_params",lambda *args,**kwargs:nullcontext())
    monkeypatch.setattr(torch.distributed,"barrier",lambda:None)
    manager.load_checkpoint(str(destination.parent))
    torch.testing.assert_close(parameter,expected)
    assert steps==[20]
    save_file({"wrong_key":expected},str(destination/"adapter_model.safetensors"))
    with pytest.raises(ValueError):manager.load_checkpoint(str(destination.parent))
