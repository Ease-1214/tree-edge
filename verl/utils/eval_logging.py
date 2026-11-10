"""Single-writer ingestion of asynchronous checkpoint evaluation results."""
import json
from pathlib import Path

LABELS=("eval_dev1000","eval_t6125")
def pending_evaluations(root):
    root=Path(root)
    cursor=root/"evaluation-logged.json"
    logged=set(json.loads(cursor.read_text())) if cursor.exists() else set()
    metrics={};selected=[]
    for label in LABELS:
        candidates=[]
        for path in (root/"evaluation-results").glob(label+"-step*.json"):
            if path.name in logged:continue
            result=json.loads(path.read_text())
            if result.get("complete"):
                candidates.append((int(result["checkpoint_step"]),path,result))
        if candidates:
            step,path,result=min(candidates,key=lambda x:x[0])
            metrics[label+"/checkpoint_step"]=step
            metrics.update({label+"/"+key:value for key,value in result["metrics"].items()})
            selected.append(path.name)
    return metrics,selected

def acknowledge_evaluations(root,names):
    root=Path(root);path=root/"evaluation-logged.json"
    logged=set(json.loads(path.read_text())) if path.exists() else set()
    logged.update(names)
    temporary=path.with_suffix(".tmp")
    temporary.write_text(json.dumps(sorted(logged)))
    temporary.replace(path)

def define_evaluation_axes(wandb):
    for label in LABELS:
        wandb.define_metric(label+"/checkpoint_step")
        wandb.define_metric(label+"/*",step_metric=label+"/checkpoint_step",step_sync=False)
