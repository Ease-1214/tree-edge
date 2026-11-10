"""Checkpoint failures must propagate before a success message is emitted."""
from dataclasses import dataclass
from enum import Enum
from types import SimpleNamespace
import pytest
from omegaconf import OmegaConf
from verl.workers import fsdp_workers
from verl.utils.checkpoint import adapter_state

class Kind(Enum):
    VALUE="value"

@dataclass
class AdapterConfig:
    task_type:Kind=Kind.VALUE
    peft_type:Kind=Kind.VALUE
    target_modules:tuple=("q_proj",)

@pytest.mark.parametrize("failure",["state","adapter"])
def test_actual_checkpoint_entry_rejects_write_failure(monkeypatch,tmp_path,capsys,failure):
    class FakePeft:
        peft_config={"default":AdapterConfig()}
    class FakeWrapper:
        def cuda(self):return self
    monkeypatch.setattr(fsdp_workers,"PeftModel",FakePeft)
    monkeypatch.setattr(fsdp_workers,"FSDP",FakeWrapper)
    monkeypatch.setattr(fsdp_workers.dist,"barrier",lambda:None)
    monkeypatch.setattr(fsdp_workers.dist,"get_rank",lambda:0)
    monkeypatch.setattr(fsdp_workers,"layered_summon_lora_params",lambda model:{})
    def fail(*args,**kwargs):raise OSError("injected write failure")
    monkeypatch.setattr(adapter_state,"save_adapter_training_state",fail if failure=="state" else lambda *a:None)
    monkeypatch.setattr(fsdp_workers,"save_file",fail)
    worker=SimpleNamespace(_is_actor=True,_is_offload_param=False,_is_lora=True,
        actor_module=FakePeft(),actor_module_fsdp=FakeWrapper(),actor_optimizer=None,
        actor_lr_scheduler=None,rank=0,config=OmegaConf.create({"actor":{"checkpoint":{"contents":["lora_adapter"]}}}))
    with pytest.raises(RuntimeError,match="Checkpoint save incomplete"):
        fsdp_workers.ActorRolloutRefWorker.save_checkpoint.__wrapped__(worker,str(tmp_path))
    assert "Saved LoRA adapter" not in capsys.readouterr().out
