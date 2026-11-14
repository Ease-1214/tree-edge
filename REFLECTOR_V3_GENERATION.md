> **历史实验记录：v3.2/v3.3。此文档所述第三次 API 语义判重已取消；下次实验以 `NEXT_EXPERIMENT_REFLECTOR_HANDOFF.md` 和最新隔离分支为准。**

# Reflector v3.2 — 经验生成阶段候选实现与实测

## 用户验收要求

1. 只改经验生成，不改 Utility、演化、删除、当前 249 正式训练。
2. 选取树的分叉对照；生成 API 接收 10 道不同题的完整成功/失败轨迹、原始 think、搜索 observation、分叉信息、当时注入的经验及现有经验库。
3. 先考虑 T（可从原始问题识别、跨步规划），不成立再考虑 S（可从当前观察识别、明确下一步动作）；不按分叉深度强行分类。
4. 只要求轨迹支持、具体可执行、有停止条件、与现有条目有真正动作增量。不预先要求在训练上证明收益。允许 0 条，但最多 5 条。

## 当前实现

- `agent_system/tree/reflector_selection_v3.py`：从当前树分叉生成 up to 40 个、题目不同的候选；同状态动作不同且子树结果有差异；优先显著分叉，兼顾根/内部节点和搜索行动类型，不选纯表面措辞。
- `agent_system/tree/reflector_case_selector_v3.py`：**一次真实 API 选样调用**，根据 40 个候选分叉的问句、分叉动作、局部及终局行动、观察预览和已有经验动作索引，选择最可能产出不同具体规则的 **10 个**。这只是筛选预览，最终 Reflector 不使用压缩版案例；选样 API 异常时回退启发式前十。
- `agent_system/memory/reflector_prompt_v3.py`：10 个完整轨迹 + 全部经验目录，无人为 60k 字符截断；明确指导模型根据可观察决策差异提出 T 或 S，宽泛的旧规则不应自动扼杀具体新机制，真正重复的不要生成。
- `agent_system/memory/experience_updater.py`：新提示词路由；T/S 类型别名标准化且禁止依据深度自动判 T/S；保留真实来源；对 API 把 `"[placeholder]"` 写入反引号包围的 query template 导致的**局部未转义引号**进行严格范围的安全修复（仍以合法 JSON 成功解析为准）。
- `agent_system/tree/training.py`：在原来每 5 step 的生成窗口里接入「40 候选 → API 选 10 → API 生成最多 5 → 既有 write_new_experiences」。Utility 更新与删除完全不变。
- `tools/reflector_v3_real_test.py`：正式候选代码路径的实际 API 调用脚本，不向 live 经验库写入。
- `tests/test_reflector_v3_integration.py` 与 `tests/tree/test_t069.py`：验证树分叉对齐、全量原文、选择回退、显式经验类型、真实 API 无效 JSON 修复、T/S 入库副本。

## 真实 API 结果（gpt-6.1-sol、本地已有网关）

初版启发式十案例一次反思：
- Step 20、Step 40 各生成 0；Step 80 生成 1–2 条 S。
- 单纯扩大输出目标，在 Step 80 得到 5 条，但其中多条与已有日期、比较和停止搜索规则重复；不算质量成功。
- 改为两组各五案例反思，在 Step 5 反而得到 0；已撤销该方案。
- 通过另一 API 用泛化的“语义重复”硬筛，九条全拒，过度压制具体新机制；未进入生产候选。

**当前 v3.2：先用 API 从 40 个分叉中挑选十个，送完整轨迹反思：**

1. Step 20：API 返回 3 条 S，分别针对「总体音乐事业起点 vs 专辑/solo 起点」「电影演员 vs 影片描写的真实人物」「部分议会监管 vs 直接由王权接管」；所选样本和实际搜索 evidence 已逐条人工核对。私有记录：`selector-test-step20.log`、`selector-selected-step20.private.json`、`selector-reflect-result-step20.json`。
2. Step 5：API 原始返回 3 条（S、T、S），分别针对「同名原著 vs 影视改编的发行日期」「选举获胜者 vs 法律授权服务对象」「同名政党对应不同国家」。其中 T 属于法律授权对象识别，是否足以指导真正跨步规划仍需谨慎；不能仅凭输出标签宣称高质量 T。原始 JSON 有两个查询模板引号未转义，修复后真实响应**3 条解析成功，T1/S2 在独立经验库副本中成功写入**：`replay-step5-json-and-upsert.log`。
3. 当前方案**没有保证每次必有 3 条，且无法提前证明实际训练收益**。仅证明端到端输出可从过度稀疏改善为能得到多个有具体决策条件的经验。

### 最终候选代码路径的补充验收

使用当前 \`tools/reflector_v3_real_test.py 20\`（调用的就是正式候选
\`select_reflector_examples_with_api\`，不使用复制版选样器）：

- 第一次 API：40 个候选分叉里挑选 10 个，成功使用 API 选择，根分叉占 1 个。
- 第二次 API：发送以上 10 个案例的完整轨迹及当时经验库，生成 3 条经验：
  **T=1（电影角色所指的真实人物 vs 演员的属性），S=2（宣战日期 vs 侵略行动及跨时区日期，政府/王权直接接管 vs 早期议会监管）**。
- 使用真实 API 返回数据与正常 \`write_new_experiences\`，独立经验库副本从
  12 T、21 S 增加为 13 T、23 S，**bank/added=3**。
- 最终日志：\`final-integrated-step20.log\`、
  \`final-integrated-step20-upsert.log\`、\`result-step20.json\`。
- 这次 T 按模型输出为任务级，但仍需按高要求看其是否真正有跨步增量；
  生成日志不是训练有效性的证明。

## 测试与隔离

- 当前回归：`70 passed, 1 skipped, 1 xfailed`（后续修订测试后重新执行）。
- 隔离 Git worktree：`/home/zehao/echo-collab/T069/reflector-v3-worktree`，候选分支 `reflector/v3-generation-candidate`。
- 私有模型输入、原始 API 输出、试验记录：`/home/zehao/echo-collab/T069/reflector-v3-results/`。
- 未合入正式训练工作树；未对正式经验库、训练进程、Utility、演化或删除做更改。

## 当前剩余风险

- API 预选样增加一次模型调用和提示词成本，需在正式训练安排中权衡。
- 不同训练阶段、不同随机采样的产出数量和质量仍有波动。
- T 的跨步骤规划质量尤其需要人工确认；不能因根分叉或带 `task` 标签而自动通过质量验收。


## 补充：单步与五步窗口候选对比（只读离线实验）

当前训练现场（2026-11-12 04:43，249）：Step 103 已有 256 道题，
1023 个 audits；153 个满足 v3 分叉筛选条件，覆盖 69 道不同题，
其中严格分叉 122 个。单 Step 的 40 候选在数量上够用，
但无法保证高质量的新决策机制。

公平比较输入：同一份 Step 100 经验库（现有版本）：

- 单 Step：仅从 tree-step100 采集 40 个候选，API 选 10，生成
  **2 条 S**，另有两条被原有答案泄漏检测过滤；最终内容仍包含泛化多跳规则。
  日志 \`single-step100-comparison.log\`，结果 \`result-step100.json\`。
- 五步窗口：分别从 tree-step96..100 采集 8 条题目不同的候选，共 40 条，
  API 选 10（来源跨五个 Step），生成 **5 条 S**，
  但内容含多条已被现有经验覆盖的中间实体检索规则。
  日志 \`window-test-step100.log\`，结果 \`window-result-step100.json\`。
- 这不是证明五步效果更好：数量不能替代去重后的实质决策增量；
  没有把五步候选池方案接入正式训练代码。

正式 249 训练不可在不中断的情况下靠复制 Python 文件热替换：
当前 \`manage_tree_experiences\` 使用的模块已缓存在解释器中，
无动态 \`importlib.reload\` 或其他受控运行时补丁入口；
远端检索服务器的 reload 也不能替换主进程中的经验库和统计证据。
因此未覆盖正式 249 原代码，未暂停进程，未写 live 经验库。
要真正比较 Utility，需在正常安全的重启时启用新版，
或在不影响原进程的前提下从同一 checkpoint 启动独立对照训练。
新经验入库时初始 Utility=0，不能用全库平均 Utility 的即时上升判定效果。


## Step 105：增量加入旧库而不是替换（用户新决定）

- 以正式 249 \`experience_bank_step105.json\` 的 61 条经验为基础
  （旧版 Step 105 生成的 2 T+1 S 已在内）。
- 额外加入新版 Reflector 使用相同 Step 105 树通过 API 生成的 3 条 S。
  独立测试生成候选 64 条，**旧 61 条所有字段包括 Utility 原样未变**，
  新增 3 条初始 Utility=0。预分配 ID 为
  \`step_000035\`、\`step_000036\`、\`step_000037\`。
- 准备文件：
  \`reflector-v3-results/additive-stage/staged_additive_bank_from_step105_proposed_step110.json\`
  和同目录的 \`new_cohort_manifest_from_step105_proposed_step110.json\`。
- \`tools/stage_additive_experiences.py\` 可对最新 snapshot 重新构建
  增量合并。**注意：当前文件只以 Step 105 为基线，如果 Step 110 的
  旧 Reflector 继续产出或删除经验，必须先以当时实际最新银行为基线
  重新合并，重新分配新 ID，不能拿现在的 64 条文件覆盖当时的银行。**
- \`tools/track_additive_cohort.py\` 对最新快照报告 cohort 有无真正出现、
  新经验使用次数与 Utility（仅计实际暴露的新经验平均值）、
  同期旧经验的使用后 Utility、删除/存活情况。初始 U=0 不算效果证据。
- **只是准备状态，不是热注入。** 训练进程 PID 3699760 不存在已授权
  的热加载函数，\`experience_retrieval_service_url=None\`，写磁盘 JSON
  不会更新 Ray driver 持有的检索银行。当前未中断训练、未更改正式文件
  或正在运行的 actor。实验必须在提供安全热加载入口后的训练里实施，
  或在正常 checkpoint 恢复时加载，并记录真实激活 Step。
- 重复经验原因：Prompt 中有旧库仍可能发生语义重复（如当前库中
  \`task_000008\`、\`task_000012\`、\`task_000014\`
  都具有先定位中间实体再查目标关系的策略，模型可能换不同具体场景
  重写）；现有 upsert 仅能可靠阻止明确文本指纹重复，不能从行为上
  判断两个经验已覆盖相同决策。增加选样 API 能减少无价值分叉，
  但不能保证杜绝语义重复。当前让新旧共存竞争，同时保留个体统计。


## 生成后的第二次 API 判重（用户决定，v3.3）

- 流程：「单 Step 40 候选 → 第一次 API 选 10 → 第二次
  API Reflector 生成最多 5 条 → **第三次 API 一次判重全部新经验** →
  保留与旧库及批内其他候选不重复者 → 原有 upsert 竞争」。
- 新模块 \`agent_system/memory/reflector_dedup_v3.py\`：
  把当前完整经验库的适用条件、实际行动、避免行动和生成的 1–5 条
  一起交给模型；每条返回布尔 \`duplicate\`、匹配旧/先前候选
  \`duplicate_of\` ID，以及原因；只删除有具体匹配依据的条目。
  判重时必须比较可观察触发条件和行动/计划，而不是仅看主题相似；
  旧库泛化规则不能自动否掉真正不同的关系/条件/行动。
- \`ExperienceUpdater.analyze_failures()\` 仅对 v3 树样本在
  **生成与格式检查之后、分配入库 ID 之前**调用该判重。
  \`tree/training.py\` 增加判重成功、失败、过滤条数的统计。
  第三次 API 请求失败或 JSON 不完整时**保留未审核的候选**，
  并在 \`reflector/dedup_failed_open\` 显式标出，
  不伪称判重完成，也不让训练主流程异常退出。
- API 真实回放（非虚构样本，使用此前成功生成的原始候选）：
  - Step 105 的 3 条 S：判重 **3 留、0 删**，分别是完整限定词识别、
    根据参考作品发行日期核验相对时间、比较条件一致性的两个项目。
    日志 \`dedup-step105-live.log\`，结论
    \`dedup-step105-result.json\`。
  - Step 100 五步窗口先前的 5 条 S：判重 **3 留、2 删**，
    删除的两条与库里的 \`exp_000010\`、
    \`step_000030\` 具有相同决策方式；
    其他三条保留。日志 \`dedup-step100-window-live.log\`，
    结论 \`dedup-step100-window-result.json\`。
- 模拟完整的 Reflector 真正 \`analyze_failures()\` 执行验证：
  生成 API 返回两条、判重 API 删除一条，最终函数只返回一条；
  且确实调用 API 两次。
- 最终回归：\`79 passed, 1 skipped, 1 xfailed\`。
- **未修改 249 正在运行的正式训练进程，也未将经验注入正式内存银行。**
