# T71-V2 当前方法、代码交接与参数（2026-11-14）

**分支**：`t71-bankaware-reflector-v2`
**工作区**：`/home/zehao/echo-collab/T071/conditional-natural4-bankaware-v2-worktree`
**独立启动**：`examples/grpo_trainer/run_t71.sh`（未启动正式训练）
**默认实验目录**：`/home/zehao/echo-collab/T071/conditional-natural4-sft/top46-best1-opsd-delta-flatfork-um0p01-bankaware-v2-20261114`
**边界**：不修改 T70/T069 原实验、原始 checkpoint 和输出；共享 `ppo_trainer.yaml` 仍保持 EDGE 默认淘汰条件。

## 1. 训练主路径

- 数据：Search QA 训练，当前 T71 启动脚本每 Step 配置 256 道题，4 张训练 GPU（设备 2/3/4/5）。
- 先为每道题**独立自然采样 4 条完全无经验轨迹**，计其成功数。
- 自然轨迹**成功数 0 或 1**：启动条件经验树；**成功数 2～4**：跳过树（自然轨迹仍参加训练）。
- 树在有可检索经验时做左右 Fork：L 不注入本节点新经验；R 注入检索到的经验。根部先按 Embedding 相似度召回 **Task Top-4**（`task_top_k=4`），后续回合先按问题+最新观察召回 **Step Top-6**（`step_top_k=6`）。**只在这 4/6 个语义候选内，按现有相似度+Utility+UCB 探索分进行重排序，最终每个 Fork 最多新注入 1 条。**检索服务/本地检索均使用 `semantic_candidate_k=4/6`，防止大候选池提前参与重排；底层 `task_specific_top_k=4` 与树设置一致。之前注入的 Task 经验按原设计保留，因此深层 R Prompt 可能含 1 条历史 Task + 1 条当前 Step；这里限制的是**每次 Fork 的新增数量**，并非总 Prompt 历史经验数。
- T71 经验选择评分不另造公式，复用 `ExperienceMemory` 原有的 `score = 0.5*(similarity+1)/2 + 0.5*(Utility + 0.5*sqrt(log(1+N)/(1+n)))`（`n` 为当前经验累计检索次数，`N` 为对应池累计检索次数）。**第一阶段纯语义筛选 4/6 条，第二阶段依据该综合分仅选 1 条**。本地检索和远端检索都覆盖候选上限，且以同一评分选择。
- 搜索完成后按叶子成功率生成 GRPO 训练数据。共享前缀节点走 `node_mixed` 去重打包，避免重复教师计算。GRPO 主目标不变。
- 辅助优化：成功轨迹 SFT 系数 **0.05**，OPSD 辅助蒸馏系数 **0.05**，OPSD 使用 `active_kd_tokens` 独立归一化，`path_tier=2`。
- OPSD 门控：实际分叉动作不同、R 子树平均奖励严格大于 L、R 注入了新经验，才启用蒸馏；权重直接取正收益差 `weight = delta`，取消 `n_right/(n_right+1)` 收缩。该门控独立于 Utility 统计。

## 2. 经验生成（每 5 个训练 Step 一次）

1. **不修改已确定的 Fork 过滤资格**：同状态、有效动作差异、对照轨迹可配对、左/右成功率不同；筛选实现见 `reflector_selection_v3.py`。
2. 从过滤后的候选中按 Step 固定种子随机选择**最多 10 个不同问题的 Fork**。不执行 LLM Fork 排名筛选，也不先截断至某个 Top-40。
3. 第一阶段反思：每组读取 2 个 Fork 的完整成功/失败轨迹，能输出 `valid` 或 `insufficient`，不强迫每个 Fork 都产生经验。
4. **V2 第二阶段（唯一方法变动）**：在全部 `valid` 反思之外，传入当前完整经验库的精简动作目录（ID、适用条件、动作、避免项、检索描述），按可观察 WHEN / 可执行 DO / 具体 AVOID 比较真实决策增量。已有经验覆盖的行为不得换实体、日期或说法重复生成；真正新增的检查条件、查询变换、证据判别或停止动作可以保留。最终生成 **0～5 条** Task/Step 经验，不强迫凑足 5 条。第一阶段输入及 Fork 选择保持不变。
5. 反思不等于因果证明：左右路径可能受后续工具结果和其他经验注入影响。
6. **入库严格沿用 EDGE 原版规范化内容指纹**（`ExperienceMemory.upsert_experience_groups`）。规范化并比较 `retrieval_obs`、`title`、`principle`、`when_to_apply`、`task_type`；指纹相同则复用旧经验 ID、合并来源证据且保留 Utility/曝光等生命周期；不同则作为全新经验入库。
7. **没有 Embedding 语义去重阈值**，没有全库聚类，也没有 LLM 去重、LLM 合并。Embedding 仅正常用于**经验检索**。

第二阶段此前独立调用真实模型时，两个 10-Fork 重复、3 条有效、全无效的输入分别返回 5、5、3、0 条；此前也完成过 2-Fork 真实 API 链路联调。

## 3. Utility：窗口内每个 Fork 等权（本次调整）

一个管理窗口是**5 个训练 Step**。每个独立动作的有效 Fork 有左右两棵子树的叶子奖励，定义：

`delta_fork = mean(R_leaf_rewards) - mean(L_leaf_rewards)`

对某条经验，在一个窗口内累积的**每个 Fork**都贡献一次 `delta_fork`；包括 `delta=0`，但排除左右相同动作。**不再将同一道题的多个 Fork 先压成一个平均，也不再对不同题目做宏平均。**

`delta_window = mean(all_fork_deltas_for_this_experience_in_last_5_steps)`

`Utility_new = 0.9 * Utility_old + 0.1 * delta_window`

这一层最终平均仍保留，维持收益的可比较尺度（而不是把 Fork 数量直接累加成效用）。新经验 Utility 为 `0`，没有新观测的窗口 Utility 不动。现在每个 Fork 只允许新注入最多一条经验，因此其观测 Δ 对应当前新注入的那一条；仍不能据此严格证明因果关系（先前 Task 经验与后续状态可能影响奖励）。

`QuestionEvidence` 仍按问题 ID 存储完整历史记录以便诊断；其 `tree_question_statistics` 可以继续按问题汇总，但**不参与 Utility 更新和淘汰**。`merge_window` 已对 pending 中所有 Fork 值直接做平坦平均。窗口输出另含 `fork_count`、`group_count`（独立问题数），方便核查。

**真实树验证**：T70 Step 42 的快照产生 3,456 条“经验 × Fork”收益记录，按同题汇总会变成 1,574 个“经验 × 问题”组；65 条有观测的经验中有 7 条在新旧统计方式下的窗口 Δ 符号不同。这表明统计口径变化会实质影响后续 Utility，属于明确的方法修改。

## 4. 负效用和容量淘汰（本次调整）

经验分为两个独立池：**Task 最多 200 条，Step 最多 200 条**。不再采用满额删除固定 20 条或集中去重。

每 5 Step 管理时，满足以下条件才进行硬性淘汰：

| 条件 | T71 最新值 |
|---|---|
| Utility | **≤ -0.01**（负阈值） |
| 累计右分叉注入次数 `retrieval_count` | **≥ 20** |
| 连续负窗口 `consecutive_negative_windows` | **≥ 2** |
| 创建后保护期 | **0 Step（完全取消）** |

连续负窗口指两个相邻管理窗口**均有有效观测且 `delta_window < 0`**。窗口 Δ 为正、为 0 或没有观测会打断连续计数。

如果某个池**超过 200 条**，继续沿用 EDGE 的 `Utility + UCB` 保留分数裁剪超额数量（不是一次固定删 20 条）。`-0.01` 是硬删除阈值，不是 UCB 分数阈值。旧经验 ID 一经删除不复用。

**风险与回放**：`-0.01` 要求 Utility 已明显为负，避免了误设 `+0.01` 时对正 Utility 经验的硬删除；但 `retrieval_count` 在真实树中可能累计很快，20 次不等价于 20 个独立问题。使用 T70 Step 100 的 125 条经验静态回放，**9 条符合全部硬删除条件（Task 6、Step 3）**，其中没有正 Utility 经验。这不是新训练结果的预测，且 T71 的 Fork 等权 Utility 分布可能与 T70 不同，应观察删除率。

这些是**T71 专用启动脚本参数**；共享 YAML 的 EDGE 默认值保持不变，避免影响 T70/T069。

## 5. 可观测性、代码布局、验收

- `agent_system/tree/core.py`：`TreeRollout.fork_credits` 逐 Fork 产生信用记录（旧接口 `question_credits` 保留兼容）；`QuestionEvidence.record` 兼容历史两元键和新的三元键；`merge_window` 用 Fork 等权窗口平均 + EMA，维护连续负窗口，并输出 `fork_count`。
- `agent_system/tree/training.py`：每步统计右节点注入次数，5 Step 管理一次；记录 `bank/utility_fork_observations`、`bank/utility_question_groups`、`bank/utility_updates` 等；调用两阶段反思、EDGE 内容指纹入库、EDGE 淘汰函数，然后保存快照并同步检索服务。
- `agent_system/tree/reflector_selection_v3.py`：保留原资格过滤，随机采样模式。
- `agent_system/tree/reflection_two_stage.py`：两阶段 Prompt、结构检查和入库转换；Reflector API 失败则跳过该批经验，不中断主训练。
- `agent_system/memory/experience_memory.py`：原有检索、规范化指纹入库、生命周期、硬性及 UCB 容量淘汰，不引入新增合并器。
- `examples/grpo_trainer/run_t71.sh`：T71 独立配置、独立输出路径、W&B 设置；确认分支和输出目录为空才可启动。
- `tests/test_t71_edge_fingerprint_eviction.py`：Fork 扁平平均、阈值 -0.01、最低曝光 20、无保护期、启动参数，以及 EDGE 原版指纹入库。
- 其他历史树和体验反思测试作为回归测试一并执行。

**T71-V2 隔离说明**：从原始 35 条经验库开始新实验；旧 T71 训练输出与 checkpoint 原样保留。启动状态以实际日志为准，不允许把新提示词热注入旧 Ray 进程。
