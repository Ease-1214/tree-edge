# T75 方法说明与项目框架（Tree-EDGE）

> 本文档面向后续接手本项目的开发者或 AI，完整记录 **Tree-EDGE** 在分支 `t75` 上的核心设计理由、模块划分与训练/更新闭环，避免重复对齐背景。

---

## 一、核心设计理由（Why）

1. **受控树分叉（Natural 主干 + 条件 Fork），而非单线注入经验**
   * **原因**：若在主干 rollout 中直接注入经验，无法区分"题目本身简单"还是"经验真正起效"，且坏经验会污染整条轨迹的梯度。
   * **做法**：先跑无经验的 Natural 主干；在关键决策点构造受控分叉——**Left 分支不注入新经验，Right 分支注入检索到的 Top-1 经验**。通过同一状态下的对比，直接测量经验的真实因果增益 $\Delta = V_{\text{right}} - V_{\text{left}}$。

2. **V3 双池经验库 + 状态摘要检索 + 0.4 相似度门控**
   * **原因**：开局规划（Task 级）与中途纠错（Step 级）所需策略完全不同；早期单池/多条注入易产生经验冲突或将 ALFWorld 等无关任务经验混入 Search。
   * **做法**：
     * 经验库拆分 `task_experiences`（`depth == 1`）与 `step_experiences`（`depth >= 2`），剔除所有非 Search 经验；
     * 检索时先构造状态摘要 Query（Question + 历史搜索/观察摘要），以 `0.4` 余弦相似度阈值初筛，再用 UCB 重排并**仅注入 Top-1**；若候选全部低于 `0.4`，直接跳过本次分叉，宁缺毋滥。

3. **无偏递归子树价值（`evaluate_forks`）与分层分桶抽样**
   * **原因**：如果用扁平叶子平均计算分叉价值，深层反复分叉的分支会因叶子数多而被错误加权。
   * **做法**：在 `TreeRollout.evaluate_forks()` 中按树结构自底向上递归计算节点期望价值，得到无偏 $\Delta$；Reflector 只抽取 `delta < 0`（注入经验反而变差或未起效）的分叉，按 `depth == 1` 与 `depth >= 2` 分桶抽样，且同一分叉内优先挑选后续干扰更少的成功叶子。

4. **GRPO 组内优势与格式惩罚解耦 + SFT 有效动作掩码**
   * **原因**：难题推理过长被截断（`valid = False`）时，若直接把 `reward` 设为负值再做组内标准化，会把同组其他"格式正确但答错（`0` 分）"的轨迹抬成正优势；同理，若不掩码无效动作，`success_sft` 会学到被截断的坏格式。
   * **做法**：先对原始 `0/1` 正确性奖励做 GRPO 组内标准化，再仅对 `valid == False` 的节点扣除固定惩罚（`0.1`）；`success_sft` 权重严格要求 `leaf.reward == 1 and node.valid`。

5. **Reflector 单次 API 提取 + 0.85 向量去重**
   * **原因**：每轮多次 LLM 调用（生成 + 校验）成本高且易超时。
   * **做法**：默认每窗口仅调 **1 次** Reflector API（设置 `ENABLE_EXPERIENCE_TYPE_VALIDATION=1` 可开启 2-call 类型校验）；Prompt 内明确列出"本分叉已注入的经验原文"与 11 个规范 `task_type`，要求反重复；入库时执行 `0.85` 余弦相似度去重。

6. **精简 6 面板 W&B 指标与本地日志**
   * **原因**：原始 180+ 个指标与旧版 `DASHBOARD` 重复严重，难以监控核心训练健康度。
   * **做法**：`build_clean_wandb_metrics()` 统一收敛为 `1_eval/`、`2_tree_exp/`、`3_memory/`、`4_loss/`、`5_behavior/`、`6_perf/` 六个面板，同时控制 W&B 与 `step_metrics.jsonl`（如需原始全量指标可设 `WANDB_LOG_RAW_METRICS=1`）。

---

## 二、项目代码框架（Where）

| 模块 | 核心文件 | 职责 |
| :--- | :--- | :--- |
| **树结构与价值评估** | `agent_system/tree/core.py` | `TreeRollout`、`DecisionNode`、`ForkRecord`、递归 `evaluate_forks()`、GRPO 优势解耦与 `sft_weight` 计算、`trajectory_reflector_examples()` |
| **Rollout 与打包** | `agent_system/tree/runner.py` | Natural 主干与条件分叉执行、状态摘要构建、Top-1 检索门控、`pack_training_rows()` |
| **经验更新编排** | `agent_system/tree/training.py` | `write_new_experiences()`、窗口效用更新与清退 |
| **监控指标** | `agent_system/tree/metrics.py` | `CLEAN_WANDB_PANELS` 与 `build_clean_wandb_metrics()`（6 面板收敛） |
| **经验库与检索** | `agent_system/memory/experience_memory.py` | V3 双池加载、Embedding 检索（`Qwen3-Embedding-0.6B`）、`0.85` 余弦去重 |
| **反思器（LLM）** | `agent_system/memory/experience_updater.py` | 11 个规范 `SEARCH_TASK_TYPES`、Search 反思 Prompt 构建、1-call/2-call 解析与后处理 |
| **RL 训练主循环** | `verl/trainer/ppo/ray_trainer.py` | `RayPPOTrainer`、FSDP/vLLM 调度、6 面板 W&B 与 `step_metrics.jsonl` 写入 |
| **T75 启动入口** | `examples/grpo_trainer/run_t75.sh` | 统一多卡启动脚本（支持 `2xa40` / `4x3090` / `8x3090` / `auto`） |
