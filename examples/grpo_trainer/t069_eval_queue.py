"""Separate checkpoint-evaluation queue for one T069 training run."""
import argparse,json,os,subprocess,time,fcntl
from pathlib import Path
PYTHON="/home/zehao/.conda/envs/tree-edge/bin/python"

def discover(root, data_dir=None):
    root=Path(root)
    data_dir=Path(data_dir) if data_dir is not None else root.parent/"data"
    latest=root/"checkpoints/latest_checkpointed_iteration.txt"
    maximum=int(latest.read_text().strip()) if latest.exists() else 0
    jobs=[]
    for step in range(0,min(maximum,200)+1,10):
        adapter=root/f"checkpoints/global_step_{step}/actor/lora_adapter" if step else None
        if adapter and not (adapter/"adapter_model.safetensors").exists():continue
        for label,filename in [("eval_dev1000","dev1000.parquet"),("eval_t6125","test6125.parquet")]:
            if label=="eval_t6125" and step%50:continue
            output=root/f"evaluation-results/{label}-step{step}.json"
            if output.exists():continue
            dataset=data_dir/filename
            if not dataset.exists():raise FileNotFoundError(dataset)
            jobs.append(dict(label=label,checkpoint_step=step,dataset=str(dataset),
                adapter=str(adapter) if adapter else None,output=str(output)))
    return jobs

def claim_gpu():
    result=subprocess.run(["nvidia-smi","--query-gpu=index,memory.used","--format=csv,noheader,nounits"],
        capture_output=True,text=True,check=True)
    memory={int(line.split(",")[0]):int(line.split(",")[1]) for line in result.stdout.splitlines()}
    for index in [0,6,7]:
        if memory.get(index,99999)>512:continue
        handle=open(f"/tmp/t069-evaluation-gpu-{index}.lock","w")
        try:fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:handle.close();continue
        return index,handle
    return None,None

def flush_finished(root,run_id,project):
    if not (root/"training-finished.json").exists():return
    from verl.utils.eval_logging import pending_evaluations,acknowledge_evaluations,define_evaluation_axes
    metrics,names=pending_evaluations(root)
    if not names:return
    import wandb
    run=wandb.init(project=project,id=run_id,resume="must")
    define_evaluation_axes(wandb)
    while names:
        wandb.log(metrics,commit=True)
        acknowledge_evaluations(root,names)
        metrics,names=pending_evaluations(root)
    run.finish()

def run_queue(args):
    root=Path(args.run_dir);root.mkdir(parents=True,exist_ok=True)
    (root/"evaluation-results").mkdir(exist_ok=True)
    lease=open(root/"evaluation-queue.lock","w")
    fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
    while True:
        jobs=discover(root, args.data_dir)
        print("QUEUE",json.dumps(dict(pending=len(jobs),jobs=[{"step":j["checkpoint_step"],"label":j["label"]} for j in jobs])),flush=True)
        if jobs:
            gpu,handle=claim_gpu()
            if gpu is not None:
                job=jobs[0];path=root/"evaluation-job.json";path.write_text(json.dumps(job))
                env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS="2",MKL_NUM_THREADS="2",
                    PYTHONPATH=str(Path(__file__).resolve().parents[2]),HF_HUB_OFFLINE="1",
                    TRANSFORMERS_OFFLINE="1",VLLM_USE_V1="1",
                    CC="/home/zehao/.conda/envs/tree-edge/bin/x86_64-conda-linux-gnu-cc")
                with open(root/"evaluation-worker.log","a") as log:
                    code=subprocess.run([PYTHON,str(Path(__file__).with_name("t069_evaluate.py")),"--job",str(path)],
                        env=env,stdout=log,stderr=subprocess.STDOUT).returncode
                handle.close()
                if code:raise RuntimeError(f"Evaluation failed for {job['label']} step {job['checkpoint_step']}, code {code}")
                continue
        flush_finished(root,args.run_id,args.project)
        if args.once:return
        if (root/"training-finished.json").exists() and not jobs:return
        time.sleep(30)

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--run-dir",required=True);parser.add_argument("--run-id",required=True)
    parser.add_argument("--project",default="tree-edge-search");parser.add_argument("--once",action="store_true")
    parser.add_argument("--data-dir",default=None,help="Dataset root; default is run_dir.parent/data")
    run_queue(parser.parse_args())
