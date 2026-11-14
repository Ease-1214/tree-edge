#!/usr/bin/env python3
"""Create the curated W&B saved view for the T066 tree runs.

    pip install wandb-workspaces
    WANDB_BASE_URL=https://api.forge.coreweave.com python scripts/wandb_t066_workspace.py \
        --entity wuzehao1214-sun-yat-sen-university --project tree-edge-search

Five sections, 17 panels, read top to bottom:
  1 effect -> 2 experience -> 3 distillation -> 4 health -> 5 cost.
All runs of the project are shown, so A (tier2) and B (tier1) overlay in every panel.
Per-step tree metrics are noisy (64-256 questions), so panels use exponential smoothing.
"""
import argparse

import wandb_workspaces.reports.v2 as wr
import wandb_workspaces.workspaces as ws


def line(title, keys, smooth=0.6):
    return wr.LinePlot(
        title=title, x="Step", y=list(keys),
        smoothing_factor=smooth, smoothing_type="exponential",
    )


SECTIONS = [
    ("1 效果：无经验叶子 = 部署条件", [
        line("无经验叶子成功率 vs 全部叶子", ["1_main/pure_noexp_success", "1_main/leaf_success"]),
        line("无经验叶子成功率（按数据集）", ["1_main/pure_noexp_success_nq", "1_main/pure_noexp_success_hotpotqa"]),
    ]),
    ("2 经验有没有用、经验库在不在演化", [
        line("经验 A/B：右支−左支成功率（任务 T / 步骤 S）", ["2_exp/delta_task_T", "2_exp/delta_step_S"]),
        line("有差别的分叉里：经验帮了 / 害了", ["2_exp/helped_rate", "2_exp/harmed_rate"]),
        line("经验库大小", ["2_exp/bank_size"], smooth=0),
        line("每次更新新增（任务 / 步骤）", ["2_exp/added_task", "2_exp/added_step"], smooth=0),
        line("效用分数的离散度（是否拉开差距）", ["2_exp/utility_std"], smooth=0),
        line("检索集中度：前 5 条经验占比", ["2_exp/retrieval_top5_share"]),
    ]),
    ("3 蒸馏：量够不够、带 T 稳不稳", [
        line("每步蒸馏节点数", ["3_kd/canonical_rows"]),
        line("KD / PG 损失比", ["3_kd/kd_to_pg_ratio"]),
        line("student 在 teacher top-k 上的概率质量（无 T / 带 T）", ["3_kd/student_mass_noT", "3_kd/student_mass_withT"]),
        line("top-k KL（无 T / 带 T）", ["3_kd/kl_noT", "3_kd/kl_withT"]),
    ]),
    ("4 训练健康", [
        line("grad_norm", ["4_health/grad_norm"]),
        line("熵", ["4_health/entropy"]),
        line("回答长度 / 每题搜索次数", ["4_health/response_length", "4_health/search_calls_per_q"]),
    ]),
    ("5 成本", [
        line("每步耗时（秒）", ["5_cost/step_seconds"], smooth=0),
        line("每题生成次数", ["5_cost/generations_per_q"], smooth=0),
    ]),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--entity", required=True)
    ap.add_argument("--project", required=True)
    ap.add_argument("--name", default="t066-overview")
    args = ap.parse_args()
    workspace = ws.Workspace(
        entity=args.entity, project=args.project, name=args.name,
        sections=[ws.Section(name=name, panels=panels, is_open=True) for name, panels in SECTIONS],
    )
    workspace.save()
    print("saved view:", workspace.url)


if __name__ == "__main__":
    main()
