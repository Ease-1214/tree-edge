"""Late checkpoint results use their own x axis and are acknowledged once."""
import importlib.util,json,sys
from pathlib import Path
from types import SimpleNamespace
from verl.utils.eval_logging import pending_evaluations
from verl.utils.tracking import Tracking

def test_late_fake_checkpoint_keeps_checkpoint_axis(tmp_path,monkeypatch):
    result=tmp_path/"evaluation-results";result.mkdir()
    (result/"eval_dev1000-step40.json").write_text(json.dumps(dict(complete=True,checkpoint_step=40,metrics={"EM":.3})))
    calls=[];axes=[]
    fake=SimpleNamespace(init=lambda **kwargs:None,define_metric=lambda *a,**kw:axes.append((a,kw)),
        log=lambda **kw:calls.append(kw),finish=lambda **kw:None)
    monkeypatch.setitem(sys.modules,"wandb",fake)
    monkeypatch.setenv("T069_RUN_DIR",str(tmp_path))
    logger=Tracking("synthetic","fake",default_backend="wandb")
    logger.log({"training/global_step":100},step=100)
    assert calls[0]["data"]["eval_dev1000/checkpoint_step"]==40
    assert calls[0]["step"]==100
    assert any(a==("eval_dev1000/*",) and k["step_metric"]=="eval_dev1000/checkpoint_step" for a,k in axes)
    assert pending_evaluations(tmp_path)==({},[])
    logger.finish()

def test_queue_discovers_only_complete_checkpoint_steps(tmp_path):
    source=Path(__file__).resolve().parents[1]/"examples/grpo_trainer/t069_eval_queue.py"
    spec=importlib.util.spec_from_file_location("t069_queue",source)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    data=tmp_path/"data";data.mkdir()
    for name in ["dev1000.parquet","test6125.parquet"]:(data/name).touch()
    run=tmp_path/"tree";(run/"checkpoints").mkdir(parents=True)
    (run/"evaluation-results").mkdir()
    (run/"checkpoints/latest_checkpointed_iteration.txt").write_text("40")
    for step in [10,20,30,40]:
        p=run/f"checkpoints/global_step_{step}/actor/lora_adapter";p.mkdir(parents=True)
        (p/"adapter_model.safetensors").touch()
    jobs=module.discover(run)
    assert {(j["label"],j["checkpoint_step"]) for j in jobs}=={
        ("eval_dev1000",0),("eval_t6125",0),*[("eval_dev1000",s) for s in [10,20,30,40]]}
