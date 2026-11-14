import multiprocessing
import torch.distributed as distributed
from agent_system.tree.metrics import merge_tree_metrics, gather_tree_metrics


def test_disjoint_rank_metrics():
    metrics = merge_tree_metrics(
        [
            {
                "kd/no_T/topk_kl": [0.1, 0.3],
                "perf/edge_teacher_rows": 2,
                "perf/edge_student_rows": 64,
                "perf/max_memory_allocated_gb": 18,
            },
            {
                "kd/with_T/topk_kl": [0.2],
                "perf/edge_teacher_rows": 1,
                "perf/edge_student_rows": 64,
                "perf/max_memory_allocated_gb": 19,
            },
        ]
    )
    assert metrics["kd/no_T/topk_kl"] == [0.1, 0.3]
    assert metrics["kd/with_T/topk_kl"] == [0.2]
    assert metrics["perf/edge_teacher_row_fraction"] == 3 / 128
    assert metrics["perf/max_memory_allocated_gb"] == 19


def collect_worker(rank, path, queue):
    distributed.init_process_group(
        "gloo", init_method="file://" + path, rank=rank, world_size=4
    )
    result = gather_tree_metrics(
        {
            "perf/edge_teacher_rows": rank,
            "perf/edge_student_rows": 64,
            "only_rank_" + str(rank): [rank],
        }
    )
    if rank == 0:
        queue.put(result)
    distributed.destroy_process_group()


def test_four_rank_cpu_collective(tmp_path):
    context = multiprocessing.get_context("spawn")
    queue = context.Queue()
    children = [
        context.Process(
            target=collect_worker, args=(rank, str(tmp_path / "gloo-init"), queue)
        )
        for rank in range(4)
    ]
    for child in children:
        child.start()
    try:
        result = queue.get(timeout=90)
        for child in children:
            child.join(timeout=30)
            assert child.exitcode == 0
        assert result["perf/edge_teacher_row_fraction"] == 6 / 256
        assert all("only_rank_" + str(rank) in result for rank in range(4))
    finally:
        for child in children:
            if child.is_alive():
                child.terminate()
                child.join()


def test_global_actor_counts_are_not_summed_twice():
    local={"actor/statistics_global":1.,"perf/edge_teacher_rows":6.,"perf/edge_student_rows":40.}
    merged=merge_tree_metrics([local]*4)
    assert merged["perf/edge_teacher_rows"]==6
    assert merged["perf/edge_student_rows"]==40
    assert merged["perf/edge_teacher_row_fraction"]==.15
