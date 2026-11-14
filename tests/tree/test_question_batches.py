import math
from agent_system.tree.question_batches import question_batches,collapse_rows


def test_question_bins_and_collapse():
    rows=[]
    for q in range(256):
        for i in range(12+(q%17)):
            rows.append(dict(uid=str(q),node_id=str(q)+":"+str(i%5),advantage=1. if i%2 else -.5,kd_weight=.5 if i==0 else 0.))
    bins=question_batches(rows)
    assert len(bins)==math.ceil(len(rows)/256)
    average=len(rows)/len(bins)
    assert max(abs(len(x)-average)/average for x in bins)<=.2
    seen={}
    for batch,indices in enumerate(bins):
        for i in indices:
            assert rows[i]["uid"] not in seen or seen[rows[i]["uid"]]==batch
            seen[rows[i]["uid"]]=batch
        packed=collapse_rows(rows,indices)
        assert sum(x["w_tok"] for x in packed)==len(indices)
        assert sum(x["kd_weight"] for x in packed)==sum(rows[i]["kd_weight"] for i in indices)
        assert all(x["A_pos"]>=0 and x["A_neg"]<=0 for x in packed)
