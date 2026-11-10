"""CPU regression tests for the production encoder batching function."""
import ast
import os
from pathlib import Path
from types import SimpleNamespace, MethodType
from typing import Any, Dict, List, Optional
from unittest.mock import patch
import numpy as np

SOURCE = Path(__file__).resolve().parents[1] / "agent_system/memory/experience_memory.py"

def encoder():
    tree = ast.parse(SOURCE.read_text())
    cls = next(item for item in tree.body if isinstance(item, ast.ClassDef) and item.name == "ExperienceMemory")
    method = next(item for item in cls.body if isinstance(item, ast.FunctionDef) and item.name == "_encode_dynamic")
    namespace = dict(os=os, List=List, Dict=Dict, Any=Any, Optional=Optional)
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(SOURCE), "exec"), namespace)
    instance = SimpleNamespace(_embed_mem_cap=lambda model: None)
    instance.encode = MethodType(namespace["_encode_dynamic"], instance)
    return instance

class FakeModel:
    max_seq_length = 8192

    def __init__(self, fail_above=None):
        self.fail_above = fail_above
        self.calls = []

    def tokenizer(self, texts, **kwargs):
        return {"input_ids": [list(range(int(text))) for text in texts]}

    def encode(self, texts, **kwargs):
        padded_tokens = len(texts) * max(map(int, texts))
        self.calls.append((len(texts), padded_tokens))
        if self.fail_above and padded_tokens > self.fail_above and len(texts) > 1:
            raise RuntimeError("CUDA out of memory")
        return np.asarray([[float(text)] for text in texts], dtype=np.float32)

def test_limits_and_duplicate_order():
    with patch.dict(os.environ, EDGE_EMBED_TOKEN_BUDGET="128", EDGE_EMBED_MAX_ROWS="4", EDGE_EMBED_AUTOGROW="0"):
        instance = encoder()
        model = FakeModel()
        texts = [str(i) for i in range(1, 71)] + ["2", "3"]
        result = instance.encode(model, texts)
        assert result[:, 0].tolist() == list(map(float, texts))
        assert all(rows <= 4 and tokens <= 128 for rows, tokens in model.calls)
        assert instance._embed_last_stats["oom_splits"] == 0

def test_shrink_rechecks_pending_batches_and_stays_shrunk():
    with patch.dict(os.environ, EDGE_EMBED_TOKEN_BUDGET="1000", EDGE_EMBED_MAX_ROWS="64", EDGE_EMBED_AUTOGROW="0"):
        instance = encoder()
        model = FakeModel(fail_above=300)
        texts = [str(i) for i in range(20, 101)]
        fake_torch = SimpleNamespace(cuda=SimpleNamespace(empty_cache=lambda: None))
        with patch.dict("sys.modules", torch=fake_torch):
            result = instance.encode(model, texts)
        assert result[:, 0].tolist() == list(map(float, texts))
        assert instance._embed_last_stats["oom_splits"] > 0
        assert instance._embed_last_stats["preventive_splits"] > 0
        reduced = instance._embed_budget
        for _ in range(40):
            instance.encode(FakeModel(), ["20"])
        assert instance._embed_budget == reduced
