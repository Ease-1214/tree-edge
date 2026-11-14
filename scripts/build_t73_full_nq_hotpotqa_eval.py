"""Build full 3610 NQ + 7405 HotpotQA T73 evaluation in T069 search schema.

Only transforms the full held-out Search-R1 test split. No sampling and no
training examples. Source file is independently audited in T069 manifest.
"""

from pathlib import Path

import pandas as pd

SOURCE = Path("/home/zehao/Search-R1-verl-fuxian/data/nq_hotpotqa_train_30720/test.parquet")
DEST = Path("/home/zehao/echo-collab/T073/data/nq_hotpotqa_full11015.parquet")
EXPECTED = {"nq": 3610, "hotpotqa": 7405}
SYSTEM = "You are a helpful and harmless assistant."


def main():
    if DEST.exists():
        raise FileExistsError(f"Refusing to overwrite evaluation data: {DEST}")
    test = pd.read_parquet(SOURCE)
    assert len(test) == 51713, f"Unexpected complete source size: {len(test)}"
    chosen = test.loc[test["data_source"].isin(EXPECTED)].copy()
    counts = chosen["data_source"].value_counts().to_dict()
    assert counts == EXPECTED, counts
    assert chosen["question"].notna().all()
    rows = []
    seen_ids = set()
    for local_id, (source_index, row) in enumerate(chosen.iterrows()):
        ds = str(row["data_source"])
        question = str(row["question"])
        answers = [str(x) for x in row["golden_answers"]]
        assert question.strip() and answers and all(a.strip() for a in answers)
        qid = f"{ds}:test:{source_index}"
        assert qid not in seen_ids
        seen_ids.add(qid)
        target = {"target": answers}
        env = {"ground_truth": target, "question": question, "data_source": ds}
        rows.append({
            "data_source": ds,
            "prompt": [
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": question},
            ],
            "ability": "fact-retrieval",
            "reward_model": {"style": "rule", "ground_truth": target},
            "extra_info": {
                "index": local_id,
                "need_tools_kwargs": True,
                "question": question,
                "split": "test",
                "tools_kwargs": {"search": {"create_kwargs": env}},
            },
            "metadata": {"qid": qid, "parquet_index": int(source_index)},
            "env_kwargs": env,
        })
    output = pd.DataFrame(rows)
    assert len(output) == 11015
    assert list(output.columns) == [
        "data_source", "prompt", "ability", "reward_model",
        "extra_info", "metadata", "env_kwargs",
    ]
    DEST.parent.mkdir(parents=True, exist_ok=True)
    output.to_parquet(DEST, index=False)
    check = pd.read_parquet(DEST)
    assert len(check) == 11015
    assert check["data_source"].value_counts().to_dict() == EXPECTED
    assert len({x["qid"] for x in check["metadata"]}) == 11015
    print("T73_FULL_EVAL_READY", DEST, check["data_source"].value_counts().to_dict())


if __name__ == "__main__":
    main()
