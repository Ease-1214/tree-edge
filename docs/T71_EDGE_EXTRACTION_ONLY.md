# T71 经验提取回退至 EDGE 单阶段（2026-11-14）

## 工作边界

- 独立代码分支：`t71-edge-extraction`。
- 独立工作树：`/home/zehao/echo-collab/T071/conditional-natural4-edge-extraction-worktree`。
- 只修改经验提取链路；**训练 GRPO、自然采样/树分叉、OPSD/SFT、检索、Utility、淘汰、入库指纹均不改**。
- 不修改正在运行的 `t71-bankaware-reflector-v2`，不覆盖其运行目录/checkpoint；本分支未启动训练。

## 运行时链路

1. **保持 T71 Fork 资格与抽样**：`trajectory_reflector_examples(..., sampling="random")`，选择最多 10 个不同问题的合格 Fork；同状态、动作差异和左右收益差过滤不改。
2. `edge_fork_adapter.edge_search_examples_from_forks` 仅把已选 Fork 的问题、历史、左右动作/收益和成功/失败完整路径转换为单阶段反思所需的字段；不做第二次挑选和新颖性判断。
3. 直接调用 `ExperienceUpdater.analyze_failures(failed_trajectories, current_experiences)`：**一次模型请求**，输入旧经验库，使用原有单阶段搜索反思的字段与输出解析；上限改回 EDGE 默认 **3 条/次**。不用 `reflection_two_stage.analyze_forks_two_stage`、Stage1/Stage2、`NOVELTY_SECTION`。
4. 提取后仍使用现有 `write_new_experiences -> ExperienceMemory.upsert_experience_groups`，文本内容指纹与合并逻辑不改。

## 相对于 EDGE 官方仓库的必要 Search 适配

官方 `origin/main:agent_system/memory/experience_updater.py` 的基础反思 Prompt 为 **ALFWorld 状态动作**，不能直接逐字用于 Search QA。当前单阶段 Search Prompt 属于项目内已有的场景适配：仍一次反思、参考现有经验、返回结构化经验；对于左右 Fork 正负收益都按实际数据描述，不虚称“有经验的一侧总是失败”。额外增加成功/失败轨迹的精简可见字段，只用于输入数据呈现，没有添加判重阶段或规则。

因此本分支实现的是 **EDGE 提取流程 + Search/Fork 数据格式适配**，不是逐字使用 ALFWorld Prompt。若要求逐字符复刻官方 ALFWorld 提示词，需要另行确认实验合理性。

## 自检与实验隔离

- 新启动脚本仍叫 `examples/grpo_trainer/run_t71.sh`，但只接受分支 `t71-edge-extraction`，默认输出独立目录 `...-edge-extraction-20261114`，W&B 名称独立。
- 离线回归：`tests/test_t71_edge_single_stage_extraction.py` 验证单次调用、已有经验上下文、Fork 证据源映射、最大 3 条；`tests/test_t71_edge_fingerprint_eviction.py` 验证原入库/淘汰口径。
- 原 T71-V2 的两阶段文件保留在仓库中供旧实验复现，但**本分支运行路径不调用**。
