"""Greedy full-episode evaluation using the formal search prompts and scorer."""
import os
os.environ.setdefault("TOKENIZERS_PARALLELISM","false")
import argparse,json,time,sys
from pathlib import Path
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
import pandas as pd
from omegaconf import OmegaConf
from transformers import AutoTokenizer
from vllm import LLM,SamplingParams
from vllm.lora.request import LoRARequest
from agent_system.memory import SearchMemory
from agent_system.environments.prompts.search import SEARCH_TEMPLATE,SEARCH_TEMPLATE_NO_HIS
from agent_system.environments.env_package.search.projection import search_projection
from agent_system.environments.env_package.search.envs import SearchMultiProcessEnv
from agent_system.environments.env_package.search.third_party.skyrl_gym.envs.search.env import SearchEnv

def evaluate(job):
    frame=pd.read_parquet(job["dataset"])
    model="/home/zehao/Tree-GRPO/model/Qwen2.5-3B-Instruct"
    tokenizer=AutoTokenizer.from_pretrained(model,local_files_only=True)
    llm=LLM(model=model,dtype="bfloat16",gpu_memory_utilization=.78,max_model_len=8192,
        max_num_seqs=64,max_num_batched_tokens=16384,enable_chunked_prefill=True,
        enable_prefix_caching=True,enable_lora=True,max_lora_rank=16,seed=0,disable_log_stats=True)
    adapter=LoRARequest("checkpoint",1,job["adapter"]) if job.get("adapter") else None
    config=OmegaConf.create(dict(search_url="http://127.0.0.1:29661/retrieve",
        api_format="search_r1_batch",topk=3,timeout=60,log_requests=False))
    rows=[];started=time.time()
    raw_path=Path(job["output"]).with_suffix(".private.jsonl")
    with raw_path.open("w",buffering=1) as handle:
        for offset in range(0,len(frame),32):
            active=[]
            for i in range(offset,min(offset+32,len(frame))):
                record=frame.iloc[i];values=record["env_kwargs"]
                env=SearchEnv(config);env.reset(dict(ground_truth=values["ground_truth"],
                    data_source=values["data_source"],max_turns=4))
                active.append(dict(index=i,question=values["question"],source=values["data_source"],
                    env=env,history=[],turns=0,trace=[]))
            while active:
                prompts=[]
                for item in active:
                    values=dict(task_description=item["question"])
                    if item["turns"]:
                        memory=SearchMemory();memory.reset(1)
                        for action,observation in item["history"]:
                            memory.store({"search":[action],"information":[observation]})
                        values.update(memory_context=memory.fetch(4,"information","search")[0][0],
                            step_count=item["turns"])
                        text=SEARCH_TEMPLATE.format(**values)
                    else:text=SEARCH_TEMPLATE_NO_HIS.format(**values)
                    rendered=tokenizer.apply_chat_template([{"role":"user","content":text}],
                        tokenize=False,add_generation_prompt=True)
                    prompts.append({"prompt_token_ids":tokenizer.encode(rendered,add_special_tokens=False)[-5000:]})
                output=llm.generate(prompts,SamplingParams(temperature=0,max_tokens=700),
                    lora_request=adapter,use_tqdm=False)
                actions,valids=search_projection([o.outputs[0].text for o in output])
                proxy=SimpleNamespace(envs=[x["env"] for x in active],retrieval_batch_size=128,is_train=False)
                SearchMultiProcessEnv._prefetch_searches(proxy,actions)
                with ThreadPoolExecutor(max_workers=len(active)) as executor:
                    results=list(executor.map(lambda pair:SearchMultiProcessEnv._sync_step(None,*pair),
                        zip(proxy.envs,actions)))
                following=[]
                for item,action,valid,(observation,reward,done,info) in zip(active,actions,valids,results):
                    item["history"].append((action,observation));item["turns"]+=1
                    item["trace"].append(dict(action=action,observation=observation,valid=bool(valid)))
                    if done:
                        row=dict(index=item["index"],source=item["source"],em=float(reward),
                            turns=item["turns"],trace=item["trace"])
                        rows.append(row);handle.write(json.dumps(row)+"\n");item["env"].close()
                    else:following.append(item)
                active=following
            print("EVAL_PROGRESS",len(rows),len(frame),flush=True)
    assert len(rows)==len(frame)
    by_source=defaultdict(list)
    for row in rows:by_source[row["source"]].append(row["em"])
    metrics={"EM":sum(r["em"] for r in rows)/len(rows),"questions":len(rows)}
    for source,scores in by_source.items():metrics["EM_"+source]=sum(scores)/len(scores)
    result=dict(complete=True,checkpoint_step=job["checkpoint_step"],label=job["label"],
        metrics=metrics,seconds=time.time()-started,dataset=job["dataset"],adapter=job.get("adapter"))
    path=Path(job["output"]);temporary=path.with_suffix(".tmp")
    temporary.write_text(json.dumps(result,indent=2));temporary.replace(path)
    print("EVAL_COMPLETE",json.dumps(result),flush=True)

if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--job",required=True)
    evaluate(json.loads(Path(parser.parse_args().job).read_text()))
