# 下一次实验交接：Reflector 树分叉生成（EDGE 简化版）

> 代码在 249 的隔离 worktree：`/home/zehao/echo-collab/T069/reflector-v3-worktree`；分支 `reflector/v3-generation-candidate`。当前正式训练未修改。本文件优先于旧的 `REFLECTOR_V3_GENERATION.md` 历史试验记录。

## 为什么简化

对照 EDGE 官方初始提交 `origin/main`：
- 原始在线 `ExperienceUpdater._build_analysis_prompt` 从失败/对照轨迹提炼**有触发条件、有下一步行动、受可见证据支持**的状态感知经验，不让 LLM 先优先找 task 经验。
- EDGE 原始的 `experience_generation/search.py` 有 **general / query-type-specific** 离线经验生成，但**不是**我们的 T/S 在线分类。
- EDGE 的 `ExperienceMemory` 把旧 `experiences` 转成 `step_experiences`，并在**旧银行迁移时**按任务类型合成 `task_experiences`。这不是每次在线 Reflector 生成 T 的要求。
- 本项目自己保留 T/S 两个检索池，所以当 Reflector 输出时需要确定**投放在哪个池**，但它不代表哪个类别更好或更难。

## 已确定的生成流程（下次实验保持）

1. 只从一个训练 Step 的树中选最多 40 个题目不同、同状态分叉对照；不混五步。
2. API ① 从 40 个预览中选最多 10 个有希望提炼经验的案例；失败时用确定性回退。
3. API ② 看 10 个**完整成功/失败轨迹**、当时注入的经验和旧经验库，找具体、可复用、可执行的策略。输出 **0–5 条**；不强制 T/S 的数量或比例。
4. **内容先于分类**。只有找到真正的经验后，才按使用时机选池：
   - **T**：在第一个回合就根据原始问题可检索并用于整道题；可以是具体任务的搜索规划或规则，不必一定是多跳。
   - **S**：当前搜索结果与原始问题共同决定是否适用；指导当前回合如何行动。
   - T 可以写未来可能遇到的情况的处理办法；不能依赖已经看过一个不存在于题目中的搜索结果才能触发。
   - 不是“先尽量生成 T，再用 S 兜底”；不是“只要有两步就是 T”。不按根/内部分叉深度分类。
5. **取消额外语义判重 API**，只保留经验库现有的文本指纹去重。不因为它与通用经验内容类似就禁止具体经验。
6. 新旧经验一起加入现有检索库参与竞争，沿用既有训练机制；Utility 与删除的最终规则见文末；新经验初始 Utility=0，真实身份与来源证据按入库数据记录。
7. **不再使用硬编码的英文触发词正则**给 T/S 自动判型或删掉候选：它只识别文字模式，不是可靠的语义判定。
8. 生成数量、T/S 比例不能证明好坏。真正的效果在新经验进入训练后看**新经验 cohort** 的平均 Utility、暴露次数、不同题目支持证据、删除情况，与同期旧版生成经验对照。

## 改动范围

- `agent_system/tree/reflector_selection_v3.py` — 单 Step 40 候选，未改。
- `agent_system/tree/reflector_case_selector_v3.py` — API ① 选 10，未改。
- `agent_system/memory/reflector_prompt_v3.py` — API ② 改回接近 EDGE 的短指令：从证据中找经验，然后按检索时机分 T/S；不再要求两阶段强制规划 T。
- `agent_system/memory/experience_updater.py` — 不调用额外去重 API；仍校验 JSON、类型、来源以及答案泄漏；不硬判 T/S。
- `agent_system/tree/training.py` — 管理步与 Utility 更新照旧；增加/保持必要计数。
- `tests/test_reflector_v3_type_policy.py` — 只测简洁分类说明、保持输出类型、无多余第三次 API。
- `tools/reflector_v3_real_test.py` — 仅做离线真实 API 验证，不接触 live 银行。

## 验证与下次上线

```bash
cd /home/zehao/echo-collab/T069/reflector-v3-worktree
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 /home/zehao/.conda/envs/tree-edge/bin/python -m pytest \
  tests/test_reflector_v3_type_policy.py tests/test_reflector_v3_integration.py tests/tree \
  tests/test_reflector_failure_metrics.py -q
```

旧 v3.4 在 Step 105 生成 4T+1S，是**已撤销的偏向任务级 Prompt** 的历史输出。
**本简化版真实 API 验证**：同一 Step 105 树，40 个候选→API 选 10→Reflector 生成 5 条（本次全部 S），请求和 JSON 解析成功；只读日志 \`reflector-v3-results/final-edge-simple-step105.log\`，结果 \`result-step105-edge-simple.json\`。此次全 S 也不证明应优先 S：采样有波动，**不要再按比例调 Prompt**。下次实验须在训练启动/安全恢复时从本隔离分支加载；旧的 249 正式进程不会因文件覆盖而热替换。

## 2026-11-12 决议：Utility 更新调整（已在本隔离分支实现）

- TreeRollout.question_credits：仍排除左右动作相同的分叉，但不再排除所有结果相同的有效分叉；零增益写入 Utility 和累计题目统计。每经验、每独立题目仍只平均一次，不按叶子条数重复计分。
- KD 蒸馏门控、Reflector 的 40→10 选样及成功/失败轨迹对照均不修改。
- examples/grpo_trainer/run_t069.sh：utility_ema_beta 改为 0.1（旧值 0.3）；更新间隔保持 5 Step。
- 原树方法曾基于全历史累计题目的 upper_2se<0 删除；现已按用户决定改为 EDGE 原版的四条件删除，详见下一节。
- ID 复用问题已修复：经验库 metadata 持久化 task/step/exp 三类 ID 的递增高水位；删除不回收编号；旧 checkpoint 恢复时还从历史 observations/pending 预留旧 ID。旧的 task_000015 混合记录无法反向拆分，修复在新实验生效。
- 历史回溯限制：旧 checkpoint 的 QuestionEvidence 只保留了之前筛选后的收益证据；新代码不会自动补回历史零收益。应在新的干净实验中使用统一统计口径。
- 部署限制：当前修改只在隔离 worktree，run_t069.sh 已改为从脚本所在仓库加载代码。禁止影响 249 正在运行的正式训练；下一次启动须使用新的独立 run 目录和实验 ID。

## 2026-11-12 最终决议：恢复 EDGE 动态删除 + 全生命周期 ID 唯一（已实现）

删除从树方法的历史统计检验恢复为原版 EDGE 动态管理的**联合条件**，但因 beta=0.1 与零收益纳入，树实验将 Utility 阈值调整到 -0.1；必须同时满足：
1. 当前 Utility <= -0.1（树方法阈值；EDGE 基线默认值仍为 -0.25）；
2. 当前经验累计注入次数 >= 5；
3. 连续 >= 2 个管理窗口的当窗平均收益为负；中间未观察到/零收益/正收益都中断连续性；
4. 创建后已满 20 Step 的保护期。

容量清理（T/S 各 200）继续使用原来的 Utility + UCB 保留分数，与硬负向删除分开记录。beta=0.1、5 Step 窗口保持。全部历史问答统计仍可用于诊断，但**不再作为删除标准**。树分叉按独立题目聚合，其对照设计与原版 EDGE 的采样不同，不能据此宣称两者收益估计相同。

代码：agent_system/tree/core.py 的 QuestionEvidence.merge_window 只负责窗口 Utility 和负窗口计数；agent_system/tree/training.py 调用现有 ExperienceMemory.evict_excess_experiences，并将硬删除、容量删除分别计数。agent_system/memory/experience_memory.py 对 task/step/exp 维护银行 metadata 中的永久 ID 高水位，删除后仍不复用，序列经保存/加载/恢复持续；agent_system/memory/experience_updater.py 的 legacy exp 编号也遵循该高水位。

验证覆盖了删掉当前最大 ID、文件保存加载、旧检查点历史 ID 预留、四项联合删除条件、缺失窗口与零收益打断连续负收益。**不自动迁移/清洗 T069 旧运行中已混合的历史观测，不触碰正在运行的实验**。

## 2026-11-12 最后参数更新：删除阈值 -0.25 → -0.1（已实现）

用户确认树方法采用 **Utility <= -0.1**，其他四条件删除逻辑保持 EDGE 原版：累计注入 >=5 次、连续 2 个 5-Step 管理窗口平均收益为负、创建已满 20 Step；T/S 各 200 容量清理仍独立。

- 修改位置仅为候选分支启动脚本：examples/grpo_trainer/run_t069.sh 的 **tree 模式** OPTIONS 额外传入 env.experience_memory.management.eviction_utility_threshold=-0.1。代码从 management 配置读取该参数，无需改删除函数。
- **刻意不改** verl/trainer/config/ppo_trainer.yaml 内 EDGE 基线管理阈值 -0.25，以免污染 baseline。
- beta=0.1、5 Step 更新、零收益纳入 Utility、ID 高水位防复用、Reflector 和 KD 均保持不变。
- 旧 Step 5–155 日志按 beta=0.1 模拟：阈值 -0.25 满足全部删除门槛的经验生命周期数为 0；阈值 -0.1 为 5/90。**这些是旧收益窗口上的近似回放，不是新训练实测**；旧日志缺少零收益分叉，历史里还有 ID 复用，因此实际删除数量可能不同。
- 验收：用 bash -n 检查脚本，单元测试验证 -0.1 会触发四条件删除，而 -0.25 不会；确认启动脚本只有 tree 分支传入 -0.1，edge 模式维持默认 -0.25；不得启动/停止/重启原正式训练。
