#!/usr/bin/env python3
"""Stateless shared embedding service (T065).

One process serves any number of training runs: it only turns texts into vectors.
Each run keeps its own experience bank and computes similarities locally, so runs
never share state.  Encoding reuses ExperienceMemory's dynamic batching (length-sorted
token budget, duplicate removal, OOM halve-and-retry, bf16, optional memory cap), so
vectors match local encoding up to bf16 batch-composition noise.

    CUDA_VISIBLE_DEVICES=0 EDGE_EMBED_MEM_GB=4 python embedding_server.py \
        --model /path/Qwen3-Embedding-0.6B --port 29700

Client side: export EDGE_EMBED_URL=http://<host>:29700 (ExperienceMemory then never
loads a local model).
"""

import argparse
import os
import base64
import sys
import threading
import time
from pathlib import Path
from typing import List

import numpy as np
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

_repo_root = Path(__file__).resolve().parents[2]
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from agent_system.memory.experience_memory import ExperienceMemory  # noqa: E402


class _Encoder:
    # Borrow the exact encoding code used by training-side ExperienceMemory.
    _maybe_cast_embedding_model = ExperienceMemory._maybe_cast_embedding_model
    _embed_mem_cap = ExperienceMemory._embed_mem_cap
    _encode_dynamic = ExperienceMemory._encode_dynamic

    def __init__(self, model_path: str, device: str):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_path, device=device)
        self._maybe_cast_embedding_model(self.model)
        self.lock = threading.Lock()
        self.calls = 0
        self.max_peak_bytes = 0
        self.total_seconds = 0.0
        self.oom_splits = 0


app = FastAPI()
_enc: _Encoder = None


class EmbedRequest(BaseModel):
    texts: List[str]
    normalize: bool = True
    client: str = ""


@app.post("/embed")
def embed(req: EmbedRequest):
    import torch

    with _enc.lock:  # one GPU stream; concurrent runs queue here
        cuda = torch.cuda.is_available()
        if cuda:
            torch.cuda.reset_peak_memory_stats()
        t0 = time.monotonic()
        vecs = _enc._encode_dynamic(_enc.model, req.texts, req.normalize)
        seconds = time.monotonic() - t0
        peak = torch.cuda.max_memory_allocated() if cuda else 0
        _enc.calls += 1
        _enc.total_seconds += seconds
        _enc.max_peak_bytes = max(_enc.max_peak_bytes, peak)
        batch_stats = dict(getattr(_enc, "_embed_last_stats", {}))
        _enc.oom_splits += int(batch_stats.get("oom_splits", 0))
    vecs = np.ascontiguousarray(np.asarray(vecs, dtype=np.float32))
    print(
        f"T065_EMBED_SERVE client={req.client} rows={len(req.texts)} uniq={len(set(req.texts))} "
        f"seconds={seconds:.3f} peak_GiB={peak / 2**30:.3f} budget={getattr(_enc, '_embed_budget', None)}",
        flush=True,
    )
    return {
        "shape": list(vecs.shape),
        "data": base64.b64encode(vecs.tobytes()).decode("ascii"),
        "seconds": seconds,
        "peak_bytes": peak,
        "batch_stats": batch_stats,
    }


@app.get("/health")
def health():
    return {
        "ok": True,
        "calls": _enc.calls,
        "total_seconds": _enc.total_seconds,
        "max_peak_GiB": _enc.max_peak_bytes / 2**30,
        "budget": getattr(_enc, "_embed_budget", None),
        "dtype": str(next(_enc.model.parameters()).dtype),
        "oom_splits": _enc.oom_splits,
        "configured_budget": int(os.environ.get("EDGE_EMBED_TOKEN_BUDGET", "32768")),
        "max_rows": int(os.environ.get("EDGE_EMBED_MAX_ROWS", "256")),
        "autogrow": os.environ.get("EDGE_EMBED_AUTOGROW", "1"),
    }


def main():
    global _enc
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=29700)
    parser.add_argument("--token-budget", type=int, default=8192)
    parser.add_argument("--max-batch-rows", type=int, default=64)
    args = parser.parse_args()
    if args.token_budget < 1 or args.max_batch_rows < 1:
        parser.error("Batch limits must be positive")
    os.environ["EDGE_EMBED_TOKEN_BUDGET"] = str(args.token_budget)
    os.environ["EDGE_EMBED_MAX_ROWS"] = str(args.max_batch_rows)
    os.environ["EDGE_EMBED_AUTOGROW"] = "0"
    _enc = _Encoder(args.model, args.device)
    print(f"T065_EMBED_SERVER_READY port={args.port}", flush=True)
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
