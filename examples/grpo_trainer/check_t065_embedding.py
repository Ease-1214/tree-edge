import os, time, json, base64
from pathlib import Path
import numpy as np
import pandas as pd
import requests
from agent_system.memory.experience_memory import ExperienceMemory
from examples.grpo_trainer.embedding_server import _Encoder

root = Path("/home/zehao/echo-collab/T065")
df = pd.read_parquet("/home/zehao/echo-collab/T061/data/train.parquet")
texts = [str(row["prompt"][0]["content"]) for _, row in df.head(64).iterrows()]
(root / "embedding-queries-private.json").write_text(json.dumps(texts))
bank = ExperienceMemory(
    "memory_data/search/state_aware_experiences_v2.json",
    retrieval_mode="embedding",
    embedding_model_path="/home/zehao/echo-collab/T061/assets/Qwen3-Embedding-0.6B",
    device="cuda:0",
)
items = bank.experiences["task_experiences"] + bank.experiences["step_experiences"]
banktexts = [bank._experience_text_for_embedding(x) for x in items]
encoder = _Encoder("/home/zehao/echo-collab/T061/assets/Qwen3-Embedding-0.6B", "cuda:0")
t0 = time.monotonic()
local = encoder._encode_dynamic(encoder.model, texts, True)
lb = encoder._encode_dynamic(encoder.model, banktexts, True)
local_seconds = time.monotonic() - t0
session = requests.Session()
session.trust_env = False
url = "http://127.0.0.1:29700"


def remote(texts):
    t0 = time.monotonic()
    r = session.post(
        url + "/embed",
        json={"texts": texts, "normalize": True, "client": "consistency"},
        timeout=600,
    )
    r.raise_for_status()
    d = r.json()
    return (
        np.frombuffer(base64.b64decode(d["data"]), dtype=np.float32).reshape(
            d["shape"]
        ),
        time.monotonic() - t0,
    )


remoteq, seconds = remote(texts)
rb, bank_seconds = remote(banktexts)
lt = np.argsort(-(local @ lb.T), axis=1)[:, :6]
rt = np.argsort(-(remoteq @ rb.T), axis=1)[:, :6]
ordered = np.mean(np.all(lt == rt, axis=1))
sets = np.mean([set(a) == set(b) for a, b in zip(lt, rt)])
cos = (local * remoteq).sum(1) / (
    np.linalg.norm(local, axis=1) * np.linalg.norm(remoteq, axis=1)
)
result = dict(
    queries=64,
    bank_entries=len(items),
    max_cosine_distance=float((1 - cos).max()),
    mean_cosine_distance=float((1 - cos).mean()),
    top6_order_agreement=float(ordered),
    top6_set_agreement=float(sets),
    local_seconds=local_seconds,
    remote_seconds=seconds + bank_seconds,
    differing_queries=int(np.sum(np.any(lt != rt, axis=1))),
)
(root / "embedding-consistency.json").write_text(json.dumps(result, indent=2))
print(json.dumps(result), flush=True)
np.savez(
    root / "embedding-consistency-private.npz",
    local=local,
    remote=remoteq,
    local_bank=lb,
    remote_bank=rb,
    local_top6=lt,
    remote_top6=rt,
)
