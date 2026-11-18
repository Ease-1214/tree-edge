# Tree-EDGE (`t75`) 代码架构、数据流与接口契约手册

> 本文档整理 `t75` 分支完整的代码分层、端到端训练数据流、核心类与 Tensor/JSON 接口契约、外部微服务协议及仓库历史文档索引。方法原理与数学推导见 `../T75_METHOD_AND_FRAMEWORK.md`。

---

## 一、端到端单步训练数据流（Step Lifecycle）

```
[Train DataLoader: 256 Questions]
        │
        ▼
1. Rollout 采样与受控树展开 (`agent_system/tree/runner.py :: tree_multi_turn_loop`)
   ├── 1a. Natural 主干 (`run_natural_rollouts`): 每题独立采样 N=4 条纯无经验轨迹 -> 统计 pass_counts[uid] ∈ {0..4}
   ├── 1b. 条件触发过滤: 仅 pass_counts <= 1 的难题进入受控经验树
   ├── 1c. 逐层受控分叉 (depth = 1..4):
   │     • depth == 1: 用原问题检索 `task_experiences` (阈值 0.3 + UCB 重排 -> Top-1)
   │     • depth >= 2: 用 `build_step_retrieval_query(state)` 检索 `step_experiences` (阈值 0.3 + UCB 重排 -> Top-1)
   │     • 若无候选过门槛: 仅走 L 分支 (不建 ForkRecord)
   │     • 若有 Top-1 经验 e: 同步展开 L (无新经验) 与 R (注入单条新经验 e)，共享完全相同的前置历史状态
   └── 1d. 递归价值评估 (`TreeRollout.evaluate_forks`):
         • 自底向上计算 V(u_R), V(u_L) 与因果增益 Δ(f) = V(u_R) - V(u_L)
         • gate = (有新经验) ∧ (L/R 归一化动作不同) ∧ (Δ(f) > 0)
        │
        ▼
2. 唯一节点打包与共享前缀去重 (`runner.py :: pack_training_rows` -> `question_batches.py :: prepare_dedup_batch`)
   ├── 每个唯一 `DecisionNode` 仅保留 1 份物理 Token 张量 (`dedup_batching="node_mixed"`)
   ├── 聚合所有经过该节点的叶子贡献: `w_tok` (叶子重数), `A_pos` (正优势和), `A_neg` (负优势和), `sft_weight`
   └── 为 `kd_canonical=True` 的节点构造 `kd_student_*` (无经验 Prompt + L 响应) 与 `edge_teacher_*` (带经验 Prompt + L 响应)
        │
        ▼
3. 经验生命周期与窗口反思 (`agent_system/tree/training.py :: manage_tree_experiences`)
   ├── 每步累加右分支曝光次数 `retrieval_count` 并向 `QuestionEvidence` 写入非同动作分叉的 `Δ(f)` 及严格负向对比案例
   └── 每 5 步 (`utility_update_interval=5`):
         • 窗口平均 Δ 更新效用 EMA (`beta=0.2`) 并维护 `consecutive_negative_windows`
         • 从 5 步累积池按问题去重 + 3 深度桶 (`d=1, d=2, d>=3`) 分层抽取最多 12 个负向分叉 (`L成R败`)
         • 单次调用 `ExperienceUpdater.analyze_failures` (`gpt-6.1-sol`) -> `0.85` 向量去重入库
         • 执行负效用硬淘汰 (`n>=20, neg_win>=3, u<=-0.005`) 与池容量裁剪 (`max=200`)
         • 原子落盘 `tree_management_step{S}.json` 与 `experience_bank_step{S}.json`
        │
        ▼
4. 旧策略对数似然与多目标策略更新 (`ray_trainer.py` -> `verl/workers/actor/dp_actor.py`)
   ├── `compute_log_prob`: `_forward_response_only_rmpad` 仅对响应预测位 [-R-1:-1] 投影词表计算 `old_log_probs`
   └── `update_policy`:
         • `L_PG`: `weighted_node_policy_loss` (正负优势独立 PPO Clip，全局展开 Token 数作分母)
         • `L_OPSD` (coef=0.05): 仅对 `edge_distill_weight > 0` 的活跃行运行 Teacher no_grad 前向与 (with-T 时) 无经验 Student 前向，计算 Top-20 Reverse KL
         • `L_SFT` (coef=0.05): 对 `sft_weight > 0` 的合法成功响应（擦除 Prompt 中经验后）计算负对数似然
         • `replica_sync`: `fsdp_size=1` 全副本模式下通过 `summed_denominators` 与 `average_replica_gradients` 精确同步全局梯度
```

---

## 二、核心数据结构与模块接口契约（Interfaces & Schemas）

### 1. 树结构核心类（`agent_system/tree/core.py`）
* **`SearchBranchState`**：可深拷贝的环境快照，包含 `uid`, `question`, `ground_truth`, `data_source`, `history: list[(action, obs)]`, `env_history`, `turns`, `task_memory`（携带 `depth=1` 注入的 Task 经验沿子树向下继承）, `path: tuple[str, ...]`。
* **`DecisionNode`**：单步模型生成节点，核心字段：
  * `node_id`（如 `s30:q5:node:RL`）、`parent_id`、`fork_id`、`depth`（`1..4`）、`branch`（`"L"` / `"R"` / `"N"`）
  * `new_experience_ids: tuple[str, ...]`（仅记录**本层新注入**的至多 1 条经验 ID）
  * `path_has_T: bool`（路径根部是否携带 Task 经验）
  * `actual_prompt_text`（实际推理 Prompt）与 `noexp_prompt_text`（擦除全部 Task/Step 经验后的纯环境 Prompt，供 OPSD Student 与 SFT Student 使用）
  * `valid: bool`（动作解析器判定格式合法且未截断）
* **`ForkRecord`**：受控分叉元数据 `(fork_id, uid, depth, left_node_id, right_node_id, new_experience_ids, utility_snapshot)`。

### 2. 训练打包 `DataProto` 张量契约（`runner.py` $ightarrow$ `dp_actor.py`）
| 字段名 | 形状 / 类型 | 语义说明 |
| :--- | :--- | :--- |
| `input_ids`, `attention_mask`, `position_ids` | `[B, L_prompt + R]` int64 | 节点实际采样时的完整输入与响应序列 |
| `responses` | `[B, R]` int64 | 节点生成的响应 Token 序列（`R = max_response_length = 700`） |
| `w_tok` | `[B]` float32 | 共享前缀去重后，该节点承载的下游叶子重数（Padding 行为 `0`） |
| `A_pos`, `A_neg` | `[B]` float32 | 经过该节点的所有正优势叶子之和（`>=0`）、负优势叶子之和（`<=0`） |
| `kd_canonical` | `[B]` bool | 是否为触发 OPSD 门控（`gate=True`）的分叉左节点唯一代表行 |
| `edge_distill_weight` | `[B]` float32 | OPSD 蒸馏权重，等于分叉正增益 `Δ(f) = V(u_R) - V(u_L)`（非活跃行为 `0`） |
| `kd_student_input_ids` | `[B, L + R]` int64 | 无经验 Student Prompt + 左节点响应（供 `path_has_T=True` 时剥离 T 重新前向） |
| `edge_teacher_input_ids` | `[B, L + R]` int64 | 对应右节点带经验 Teacher Prompt + 左节点响应 |
| `sft_weight` | `[B]` float32 | 该节点属于多少条 `reward==1 and node.valid` 的成功叶子（Padding 或无效节点为 `0`） |
| `sft_student_input_ids` | `[B, L + R]` int64 | 无经验 Prompt + 成功节点自身响应（确保内化到无经验部署策略） |
| `is_padding_copy` | `[B]` bool | 多卡等长对齐填充标记（为 `True` 时在 PG / OPSD / SFT 中全部掩码归零） |

### 3. V3 双池经验库 JSON 契约（`memory_data/search/state_aware_experiences_v2.json`）
顶层键：`version` (`"3.0"`), `metadata`（含 `experience_id_high_watermarks` 防已删 ID 复用）, `task_experiences: list`, `step_experiences: list`。单条经验结构：
```json
{
  "experience_id": "task_000015",
  "task_type": "numeric",
  "title": "简明策略标题或触发条件",
  "principle": "核心可执行动作建议与避坑指南",
  "when_to_apply": "适用条件说明",
  "retrieval_obs": "用于 Qwen3-Embedding-0.6B 编码检索的语义文本",
  "trigger": "Reflector 原始触发描述",
  "recommended_action": "Reflector 原始推荐动作",
  "avoid_action": "Reflector 原始禁止动作",
  "expected_effect": "Reflector 原始预期效果",
  "state_cues": ["cue_1", "cue_2"],
  "utility": 0.0170,
  "retrieval_count": 157,
  "last_retrieval_step": 30,
  "created_at_step": 15,
  "evidence": {
    "positive_gain_count": 0,
    "negative_gain_count": 0,
    "source_traj_uids": [],
    "last_updated_step": 30,
    "consecutive_negative_windows": 0,
    "last_negative_management_step": null,
    "tree_question_statistics": {"n": 142, "mean": 0.0343, "sample_std": 0.31, "standard_error": 0.026, "upper_2se": 0.086}
  }
}
```

### 4. 外部服务本地端口契约（31 / 249 统一回环地址）
| 本地端口 | 服务职责 | 上游宿主机与守护进程 | 健康检查命令 |
| :--- | :--- | :--- | :--- |
| `127.0.0.1:29661` | Search-R1 维基检索 (`POST /retrieve`) | `248:18125` (`watchdog_wandb_proxy.py` 守护隧道) | `curl -sS -X POST http://127.0.0.1:29661/retrieve -H 'Content-Type: application/json' -d '{"queries":["test"],"topk":1}'` |
| `127.0.0.1:29662` | W&B / 外网高可用 HTTP 代理 | `248:7890` (`watchdog_wandb_proxy.py` 自动测速切节点 + 守护隧道) | `curl -I -sS -m 5 -x http://127.0.0.1:29662 https://api.wandb.ai` |
| `127.0.0.1:29663` | Reflector LLM 网关 (`/v1/chat/completions`) | `251:29663` (`reflector_gateway.py` 三 Key 自动故障切换) | `curl -sS -m 5 http://127.0.0.1:29663/healthz` |
| `127.0.0.1:29700` | 共享 `Qwen3-Embedding-0.6B` 服务 | `248:29700` (`embedding_server.py` + `watchdog` 守护隧道) | `curl -sS -m 5 http://127.0.0.1:29700/health` |

---

## 三、仓库文档导航与历史归档说明

### 1. 当前现行核心文档（`t75` 唯一权威基准）
* `T75_METHOD_AND_FRAMEWORK.md`：核心方法公式、超参设计依据与服务概览
* `T75_AI_HANDOFF.md`：接手清单、服务器规范、启动命令与线上运行状态
* `USER_COLLABORATION_PREFERENCES.md`：沟通原则、实验信号洁净要求与训练监控干预边界
* `REFLECTOR_API_31_251_AI_GUIDE.md`：`29663` Reflector 网关排障指南
* `docs/T75_ARCHITECTURE_AND_INTERFACES.md`（本文档）：代码架构、数据流与接口契约
* `docs/T75_DEEP_AUDIT_AND_OPTIMIZATION_20261117.md`：方法、工程、LoRA、经验提示词与真实样本全量审计及下一轮优化蓝图

### 2. 历史演进归档文档（仅作背景溯源，不再作为现行参数依据）
* `README.md` 与 `docs/*.rst`、`docs/algo/*`：上游 UCAS `EDGE` / `verl-agent` 原始框架说明
* `docs/t70/*`：早期 T70 阶段条件自然采样与双 `0.05` 梯度探测实验记录（2026-11-12）
* `docs/T71_*`：T71 阶段扁平 Fork 效用统计与早期淘汰阈值探索记录（2026-11-14）
