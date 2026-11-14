"""Legacy external eval queue compatibility; T70 now uses in-process 4-GPU evaluation."""
from pathlib import Path
from examples.grpo_trainer.t069_eval_queue import discover

def test_t70_discovery_uses_explicit_external_data_dir(tmp_path):
    data=tmp_path/"T069"/"data"
    data.mkdir(parents=True)
    for filename in ["dev1000.parquet","test6125.parquet"]:
        (data/filename).touch()
    run=tmp_path/"T070"/"conditional-natural4-sft"/"v1"
    checkpoint=run/"checkpoints"
    checkpoint.mkdir(parents=True)
    (checkpoint/"latest_checkpointed_iteration.txt").write_text("50")
    for step in [10,20,30,40,50]:
        path=checkpoint/f"global_step_{step}"/"actor"/"lora_adapter"
        path.mkdir(parents=True)
        (path/"adapter_model.safetensors").touch()
    jobs=discover(run, data_dir=data)
    assert len(jobs)==8
    assert {(x["label"],x["checkpoint_step"]) for x in jobs} == {
        *{("eval_dev1000",s) for s in [0,10,20,30,40,50]},
        ("eval_t6125",0),("eval_t6125",50),
    }
    assert all(Path(x["dataset"]).parent == data for x in jobs)
    assert all(Path(x["output"]).is_relative_to(run) for x in jobs)
    assert any(x["checkpoint_step"]==0 and x["adapter"] is None for x in jobs)

def test_prior_t069_evaluation_default_data_layout_still_works(tmp_path):
    run=tmp_path/"T069"/"tv2"
    run.mkdir(parents=True)
    data=tmp_path/"T069"/"data"
    data.mkdir()
    (data/"dev1000.parquet").touch()
    (data/"test6125.parquet").touch()
    jobs=discover(run)
    assert len(jobs)==2
    assert all(Path(x["dataset"]).parent==data for x in jobs)
