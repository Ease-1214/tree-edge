# T75 方法说明与项目框架（Tree-EDGE 完整版）

> 本文档面向后续接手本项目的开发者或 AI，完整记录 **Tree-EDGE** 在分支 `t75` 上的方法原理、数学公式、超参设计理由、外部接口规范与代码框架索引。

---

## 一、核心方法与设计理由（Why & How）

### 1. 受控树分叉（Natural 主干 + 条件 Fork）与无偏递归子树价值
* **设计痛点**：在单条轨迹中直接塞入检索到的经验（如原版 EDGE / SkillRL），无法区分“模型本来就能答对”与“经验真正起效”；一旦经验有误或冲突，还会直接污染主干梯度。
* **Tree-EDGE 做法**：
  1. 每题先采样 $N=4$ 条无经验的 **Natural 主干轨迹**（对应部署时不带经验的真实泛化条件，指标 `1_eval/train_natural_pass_rate`）。
  2. 当 Natural 主干至多成功 1 条（`pass_counts <= 1`）时，在关键决策点构造受控分叉（Fork）：
     * **Left 分支**：不注入当前轮次的新经验；
     * **Right 分支**：注入当前轮次检索出的 **唯一最优（Top-1）经验**；
     * `fork_depth == 1`（开局第一轮搜索前）仅检索并注入 **Task 级经验**；`fork_depth >= 2`（拿到检索片段后的中途步）仅检索并注入 **Step 级经验**。
  3. **无偏递归子树价值（`TreeRollout.evaluate_forks`）**：
     不采用扁平叶子均值（否则深层多次分叉的子树会因叶子数量多而产生权重偏差），而是自底向上递归计算每个决策节点的期望回报：
     $$V(u) = \frac{1}{|\text{children}(u)|} \sum_{v \in \text{children}(u)} V(v)$$
     分叉点的因果经验增益定义为：
     $$\Delta(f) = V(u_{\text{right}}) - V(u_{\text{left}}) \in [-1, 1]$$
     * $\Delta(f) > 0$：经验起正作用 $\rightarrow$ 给该经验正效用反馈，并触发 **OPSD 自蒸馏**；
     * $\Delta(f) < 0$：经验起负作用或误导 $\rightarrow$ 给该经验负效用反馈，并将该分叉送入 **Reflector 反思器** 提取新经验。

### 2. V3 双池经验库、状态摘要检索与分池 `0.3` 相似度门控
* **双池结构（`state_aware_experiences_v2.json`）**：
  * 彻底清理了早期混入的 ALFWorld 经验；
  * 拆分为独立的 `task_experiences`（初始 10 条，已移除纯格式冗余 `task_000001`）与 `step_experiences`（初始 13 条）；
  * 统一规范为 11 类 Search `task_type`：`general`、`direct_retrieval`、`entity_attribute_lookup`、`multi_hop_reasoning`、`comparison`、`temporal`、`numeric`、`list_intersection`、`source_identification`、`location_chain`、`answer_normalization`。
* **初筛 + 温和 UCB 重排 + Top-1 注入**：
  * **Query 构造**：Task 级仅使用初始问题；Step 级使用 `build_step_retrieval_query()` 构造纯本地确定性状态摘要（可观测状态诊断 + 标题 + `Doc 1..3` 各取前 100 字多文档均衡摘要，去除稀释语义的原题与步数模板）。
  * **分池 `0.3` 相似度门槛**：使用 `Qwen3-Embedding-0.6B`（默认通过 `http://127.0.0.1:29700` 调用 248 共享服务，亦可本地加载）计算余弦相似度，分别使用 `similarity_threshold=0.3`（Task）和 `step_similarity_threshold=0.3`（Step）过滤低相关候选（若全部低于门槛，直接跳过本次分叉，绝不强行注入无关经验）。
  * **UCB 重排选 Top-1**：在通过门槛的候选中（候选池上限 12），以 `retrieval_alpha=0.5, retrieval_ucb_c=0.2` 平衡相似度、历史效用与温和探索，选出 **Top-1** 注入，避免多条经验相互冲突。

### 3. GRPO 组内优势与格式惩罚解耦
* **设计痛点**：遇到难题时模型易在 `<think>` 中长篇推理直至超出 500 tokens 被截断（`valid = False`）。若先对 `valid = False` 扣分再做 GRPO 组内标准化，会导致同组“格式合法但答错（reward=0）”的叶子获得**正优势**，变相鼓励模型乱猜短答案。
* **T75 解耦做法（`TreeRollout.expanded_rows` & `apply_invalid_action_penalty`）**：
  1. 先仅用原始正确性奖励 $r \in \{0, 1\}$ 计算组内标准化优势（且按节点被多少叶子共享做同步权重稀释，使提前答对与提前答错权重对称）；
  2. 标准化完成后，再对 `valid == False` 的节点单独扣除固定格式惩罚（`invalid_action_penalty_coef = 0.5`）。

### 4. 三项损失协同：GRPO PG + OPSD（21 维 `[Top-20, Rest]` Reverse KL）+ 非满分题 Success SFT
总优化目标为：
$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{PG}} + \lambda_{\text{OPSD}} \mathcal{L}_{\text{OPSD}} + \lambda_{\text{SFT}} \mathcal{L}_{\text{SFT}}, \quad \lambda_{\text{OPSD}} = 0.05,\ \lambda_{\text{SFT}} = 0.05$$

* **OPSD（On-Policy Self-Distillation）设计原理**：
  * **触发门控（Gate）**：仅当分叉满足 $\Delta(f) > 0$（带经验的 Right 分支严格优于无经验的 Left 分支）且动作有效时激活；
  * **Teacher 与 Student**：Teacher 为注入了经验 Prompt 的当前策略 $\pi_{\theta}(\cdot \mid s, e)$（`stop_grad`），Student 为同一状态下**不带经验**的策略 $\pi_{\theta}(\cdot \mid s)$，目标是把“有经验才做出的好动作分布”内化到无经验的模型参数中；
  * **权重**：直接乘以分叉优势 $\Delta(f)$（不乘右子树叶子数折扣，避免惩罚一锤定音快速成功的右分支）；
  * **为什么用 Top-20 Reverse KL（模式寻优 Mode-Seeking）**：
    Forward KL $\text{KL}(\pi_T \parallel \pi_S)$ 会强迫无经验的 Student 去覆盖 Teacher 在长尾词表上的所有概率质量（mean-seeking，导致熵增和胡言乱语）；而 **21 维分区精确 Reverse KL $\text{KL}_{21}(\pi_S \parallel \pi_T)$（在 `[Top-20, Rest]` 完备粗化划分上计算，利用寄存器内全词表 `logsumexp` 零显存构造第 21 维尾桶）是 mode-seeking**，既能将落在 Top-20 外的错误主峰拉回 Top-20 内，又能对齐 Top-20 内部相对分布，且零额外显存开销；
  * **为什么系数取 `0.05`（梯度占比 5%~30%）**：主干信号必须是环境真实奖励驱动的 $\mathcal{L}_{\text{PG}}$；OPSD 和 SFT 系数设为 `0.05`（按 active tokens 归一化）使其梯度范数稳定在 PG 梯度的 `5%~30%` 之间，既能辅助内化成功分叉与经验策略，又不会喧宾夺主掩盖 RL 探索。
* **Success SFT（跳过全对零方差毕业题 + 无效动作掩码）**：
  * `sft_weight = float(leaf.reward == 1 and node.valid and uid not in all_correct_uids)`：对齐 GRPO 零方差毕业条件，跳过整道题所有采样全部答对（`all_correct`）的已掌握简单题；保留 `0/4`（树救援）及 `1/4, 2/4, 3/4` 中所有自然采样与树采样的成功合法轮次（任何 `valid == False` 的截断轮次均不参与 SFT）。

### 5. Reflector 反思器闭环、效用更新与淘汰机制
* **素材筛选（`trajectory_reflector_examples` & `select_window_reflector_contrasts`）**：
  * 仅选 $\Delta(f) < 0$ 且满足“Left 至少 1 条成功、Right 子树全失败（`max(left)==1, max(right)==0`）”的高置信分叉；
  * **完整 5 步窗口累积**：窗口内每步收集并持久化候选对比对，在每 5 步触发反思时对整个 5 步累积池按问题去重后分桶抽样；
  * 按 `depth == 1`、`depth == 2`、`depth >= 3` 分 3 桶分层抽样（每桶最多 4 个，缺桶顺延补齐，总计最多 **12 个**案例）；
  * 同一分叉内优先挑选后续经验干扰最少的成功叶子。
* **单次 LLM 提取 + 反重复 + `0.85` 向量去重**：
  * 默认每窗口仅调 **1 次** `gpt-6.1-sol`（`ENABLE_EXPERIENCE_TYPE_VALIDATION=0`，每窗口最多新增 5 条，宁缺毋滥）；
  * 保持单条经验完整对比上下文不做丢例裁剪（仅保留 `180000` 字符防异常文本保护）；
  * Prompt 中显式提供分叉点已注入的旧经验原文，执行 `STRICT ANTI-DUPLICATION`（禁止同义复述，但允许提炼具体可操作的中途 `step` 细则，可返回 `[]`）；
  * 入库前与同池现有经验计算余弦相似度，`>= 0.85` 直接拦截去重。
* **生命周期管理**：
  * 每个窗口对本窗口实际曝光的经验计算平均 $\Delta$，用 EMA 更新效用值（`utility_ema_beta = 0.2`）；
  * **洗白与淘汰**：连续负向窗口但突然打出正向窗口时重置连续负计数；满足最小曝光量（`>= 20` 次）、连续 `3` 个负向窗口且效用 `<= -0.005` 自动淘汰，超出池容量时按最低效用清退（保护最近新增经验）。

---

## 二、外部服务与检索端调用规范（API & Retrieval）

### 1. Reflector API（经验提取与反思）
* **调用地址**：`http://127.0.0.1:29663/v1`（31 本机回环端口，由 251 服务器 `reflector_gateway.py` 通过 SSH `-R` 映射，上游对接 `https://sub.ddpapi.top/v1` 的 3 把 Key 自动故障切换）
* **模型名称**：`gpt-6.1-sol`
* **环境变量**：从 `/home/zehao/.edge_reflector.env` 加载 `EXPERIENCE_UPDATE_API_KEY`、`EXPERIENCE_UPDATE_BASE_URL`、`EXPERIENCE_UPDATE_MODEL`
* **超时配置**：客户端默认 `timeout=330.0s`、总截止时间 `340.0s`（覆盖 251 网关 `3 × 95s = 285s` 轮询上限）
* **健康检查**：`curl -sS --max-time 5 http://127.0.0.1:29663/healthz`（返回 `{"status":"ok"}`）

### 2. Search API（训练与评测环境网页/文档检索）
* **代码位置**：`agent_system/environments/env_package/search/third_party/skyrl_gym/tools/search.py`
* **调用地址**：`POST http://127.0.0.1:29661/retrieve`（由 248 服务器 `172.18.167.248:18125` 经 SSH `-R` 映射至 31/249，由 `watchdog_wandb_proxy.py` 自动守护）
* **请求与返回格式**（`api_format="search_r1_batch"`，`topk=3`，`request_batch_size=128`，`timeout=60s`）：
  ```json
  // Request
  {"queries": ["Who discovered penicillin?"], "topk": 3, "return_scores": false}
  // Response: 必须包含 "result" 字段，且长度与 queries 一致
  {"result": [[{"contents": "..."}]]}
  ```

### 3. Embedding 经验向量检索（默认 29700 远程零显存服务 + 本地回退）
* **代码位置**：`agent_system/memory/experience_memory.py` (`_encode_remote` / `_encode_dynamic`)
* **默认方式（零本地 GPU 显存开销）**：`run_t75.sh` 默认设置 `EDGE_EMBED_URL=http://127.0.0.1:29700`（由 248 服务器 `embedding_server.py` 托管 `Qwen3-Embedding-0.6B` 并经 SSH `-R` 映射至 31/249，`EDGE_CONTROLLER_GPUS=0`）。
* **本地回退方式**：若清空 `EDGE_EMBED_URL=""`，则进程内通过 `SentenceTransformer("/home/zehao/echo-collab/T061/assets/Qwen3-Embedding-0.6B", device="cuda:0")` 本地加载。

### 4. W&B 高可用代理隧道
* **集群常驻通道（首选）**：`http://127.0.0.1:29662`（由 248 映射至 31/249，并由 248 上的 `/home/zehao/echo-collab/T062/wandb-import/watchdog_wandb_proxy.py` 每 60s 主动探测 `https://api.wandb.ai`，异常时毫秒级自动测速切换健康出口节点并守护 SSH 隧道）。
* **开发机直通备份**：`http://127.0.0.1:29790`（用户 SSH 连接时自动 RemoteForward 映射，避开 `7890` 多用户端口冲突）。

---

## 三、项目核心代码与文档索引（Where）

| 模块 | 路径 | 说明 |
| :--- | :--- | :--- |
| **树核心逻辑** | `agent_system/tree/core.py` | `TreeRollout`、递归 `evaluate_forks()`、GRPO 解耦优势、`QuestionEvidence` 效用窗口合并 |
| **树执行与打包** | `agent_system/tree/runner.py` | Natural 主干与条件分叉 rollout、确定性状态摘要 `build_step_retrieval_query()`、`pack_training_rows()` |
| **共享前缀去重分批** | `agent_system/tree/question_batches.py` | `node_mixed_plan()`、`prepare_dedup_batch()` 唯一节点聚合（`w_tok, A_pos, A_neg, sft_weight`） |
| **树节点策略损失** | `agent_system/tree/dedup.py` | `weighted_node_policy_loss()` 去重节点正负优势分离裁剪 PPO 目标 |
| **树自蒸馏损失** | `agent_system/tree/loss.py` | `tree_topk_loss()`（Top-20 Reverse KL，按有效 KD Token 归一化）与 `tree_kd_metrics()` |
| **反思分桶选择器** | `agent_system/tree/reflector_selection_v3.py` | 5 步窗口候选去重与 3 深度桶分层抽样 `select_window_reflector_contrasts()` |
| **反思格式适配器** | `agent_system/tree/edge_fork_adapter.py` | 将树分叉对比结构映射为 `ExperienceUpdater` 统一输入 Schema |
| **经验生命周期编排** | `agent_system/tree/training.py` | `manage_tree_experiences()`、`write_new_experiences()`、Checkpoint 树状态精确保存/恢复 |
| **W&B 6 面板指标** | `agent_system/tree/metrics.py` | `CLEAN_WANDB_PANELS` 与 `build_clean_wandb_metrics()` |
| **双池经验库** | `agent_system/memory/experience_memory.py` | V3 双池读写、`29700` 向量检索、UCB 重排、`format_for_prompt()`、`0.85` 入库去重与淘汰 |
| **反思器客户端** | `agent_system/memory/experience_updater.py` | 11 类 `SEARCH_TASK_TYPES`、反重复 Prompt、330s 超时、JSON 解析与后处理 |
| **分布式 Actor 更新** | `verl/workers/actor/dp_actor.py` | Response-only rmpad 前向、Active-only Teacher 蒸馏、去经验 SFT 与 `replica_sync` 全局分母同步 |
| **RL 训练主循环** | `verl/trainer/ppo/ray_trainer.py` | `RayPPOTrainer.fit()`、Rollout $
ightarrow$ 经验管理 $
ightarrow$ `old_log_prob` $
ightarrow$ `update_actor` 编排 |
| **统一启动入口** | `examples/grpo_trainer/run_t75.sh` | 支持 `GPU_PROFILE=2xa40 / 4x3090 / 8x3090 / auto` 及 W&B 代理自动探测回退 |
| **架构与接口手册** | `docs/T75_ARCHITECTURE_AND_INTERFACES.md` | 全链路数据流、核心类/Tensor/JSON 接口契约与历史文档归档索引 |
