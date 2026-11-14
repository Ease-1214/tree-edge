# T70 — 条件触发经验树 + 成功轨迹 SFT：实现与训练前检查

日期：2026-11-12
状态：**T70 独立分支已完成；尚未启动正式训练；旧 T069 的数据、历史提交和结果不修改。**

## 1. Git 隔离

- 机器：249
- 新工作树：/home/zehao/echo-collab/T070/conditional-natural4-worktree
- 当前分支：t70（由 tree/t069-conditional-natural4-sft 改名，保留全部历史提交）
- 基点：fbe0ac3cf30fb0a5adcdd5a97cbee42ac392ad0f（249 tree/t069 已提交版本）
- 未移植主树工作区的 experience_updater.py、t069_eval_queue.py、test_t069_eval_gpu_retries.py 未提交修改；原树主工作区不变。
- 后续根据用户最终修订文件，专门将已提交的 Reflector 候选分支 aac5306 合入**本条件树独立分支**。候选分支本身不改变；不会更新旧 249 训练进程。
- 251 SFT 独立工程在实施时离线，无法读取其 0774702 原码。本分支按交接方案**独立实现同等 SFT 语义**，不是对 251 的精确 cherry-pick；未来核对实现差异仍是后续项。
- 未改动任何旧 run、旧 checkpoint、评测文件或 PID；未读取密钥。

## 2. 实际实现

1. **自然四采样**：每个问题 UID 先产生 4 条完整且独立的无 T/S 搜索轨迹，按真实 SearchEnv 执行动作并获得二值终局奖励，每条有独立的轨迹和节点 ID。
2. **条件触发**：自然答对数 0 或 1 则从同题初始状态运行原完整 T/S 多步 L/R 树；答对数 2、3、4 则完全跳过树。不复用自然路径中间状态。
3. **统一 GRPO**：自然叶子和实际产生的树叶子进入同一 TreeRollout，相同题级 UID 下统一进行叶子奖励归一化，并保留原无效动作惩罚。tree_advantage 在节点合并/分区之前确定。现有 node_mixed 去重的 w_tok/A_pos/A_neg 继续保持展开等价。
4. **OPSD**：只由真实树 L/R Fork 产生，本轮新增经验、L/R 当前动作不同且 p_R > p_L 才触发；权重为 (p_R-p_L)*n_R/(n_R+1)，Teacher Top-k 与两档路径处理保持原样。自然轨迹不会产生 Fork 或 KD 监督。无树时 OPSD 权重为零。
5. **成功 SFT**：配置开关独立，系数 0.05。只有终局成功的自然或树叶子参与，监督完整 Think/Search/Answer 响应 Token，不监督 Observation。Student 总是同一真实搜索历史下无经验 Prompt；已经无经验的节点复用 Actor 的 PG logprob，只有含经验 Prompt 的成功节点另做前向。共享前缀按成功叶子出现次数加权。损失在每个优化 mini-batch 的全部成功 Token 上求均值，跨数据并行卡使用共同分母同步；不按叶子、问题或来源拆分。
6. **Reflector / Utility 分离**：只有真实 T/S 树 Fork 的同状态分叉可作为 Reflector 候选，最多 40 个；自然采样只用于 GRPO 和成功 SFT，不进入 Reflector、Utility 或 OPSD。
7. **监控**：natural/pass_0_questions ～ natural/pass_4_questions、natural/pass_rate、natural/unique_first_action_rate、tree/trigger_rate、tree/rescue_rate、mix/nonzero_advantage_token_rate、mix/total_rollout_tokens、mix/natural_expanded_rows、mix/tree_expanded_rows、reflector/natural_pairs、sft/active_tokens、sft/loss_token_mean 等指标。
8. **兼容**：两个新功能默认关闭，旧版树路线保持原调用；新 launcher 在独立目录显式开启。没有引入动态门槛、无经验二叉树、来源平衡、额外 GRPO 损失、熵奖励或学习率改动。

## 3. 独立启动入口（预留，未执行）

examples/grpo_trainer/run_t70.sh

该脚本：
- 验证所在分支；拒绝把新 Run 写入 tv2、ev2、ev3 等旧目录。
- 默认新实验目录：/home/zehao/echo-collab/T070/conditional-natural4-sft/v1。
- 默认复用 T069 的 Qwen2.5-3B 和只读训练/评估数据、GPU 2–5、LoRA rank16、alpha32、fp32 trainable / bf16 base、256 题、200 步及 node_mixed 去重。**T70 独立 LR=2e-5，显式 warmup_steps=5，warmup_ratio=0，warmup_style=constant。**
- 开启 conditional_natural.enable=True、natural_rollouts=4 和 success_sft.enable=True、coef=0.05。

**不要在旧训练进程中热替换或重启。** 新训练需要独立确认资源和单卡/多卡真实环境烟测后启动。

## 4. 已覆盖的边界与验证

- 混合题组中的 0/4、1/4、2/4、3/4、4/4；仅低成功数触发树。
- 自然 4/4：无树、无 PG 优势，但成功 SFT 仍能产生梯度。
- 自然 0/4 且树全错：全组 GRPO 优势为零，不制造成功标签。
- 自然 1/4 + 树：自然与树共用一个题级优势组。
- UID、经验门控、无 T/S Prompt、真实节点回溯、树触发。
- node_mixed 合并后成功共享前缀 multiplicity 与完整展开的 SFT Token mean 等价。
- Actor 层无经验复用与有经验时另做学生前向；PG 零优势仍可 SFT 更新。
- Reflector v3 仅收集最多 40 个真实树 Fork；不足 40 个不补足。API 从中选最多 10 个并产生 0–5 条经验，完全不混入自然轨迹。
- Reflector 生成素材的来源隔离和 Utility 不受自然对照污染。
- 四进程 Gloo CPU 分布式 SFT 共同分母与梯度等价测试。
- 旧版树相关单元与集成回归。

**尚未验证**：GPU 真正多卡 3B 模型更新、vLLM/工具环境全链路和新方案训练收益；不能把 CPU 测试称为正式训练成功。

## 5. 2026-11-12 最终 Reflector / Utility 修订（合入本分支）

来源：独立候选 reflector/v3-generation-candidate@aac5306；与本条件树实现合并，**不是**直接修改 249 原 tree/t069。

- Utility 题级证据：排除同动作的真实 T/S Fork，**包括零收益分叉**；同一经验在同一题中的多个 Fork 按均值聚合。自然采样不会构造真实经验 Fork，不能评估已有经验 Utility。
- β=0.1，每 5 Step 汇总一次；连续负窗口计数在无观察、零收益或正收益时清零；旧全历史 mean+2SE<0 不再直接删除。
- 删除四条件：Utility ≤ -0.1、累计注入 ≥5、连续 2 个负收益管理窗口、创建后 ≥20 Step。T/S 各 200 条容量限制独立执行，UCB 原参数不变。
- ExperienceMemory metadata 持久化 task/step/exp 经验 ID 的递增高水位；加载/恢复时预留历史 evidence 里的已用 ID。旧训练中已混用的经验 ID 数据无法靠新代码追溯拆分。
- 旧版 EDGE baseline 的 YAML 默认经验删除阈值仍为 -0.25，未改；本条件树启动脚本专门覆盖为 -0.1。
- Reflector v3 原有 40→10→0–5、短 Prompt、T/S 使用时机分类及文本指纹去重均保留。T70 删除了自然对照候选适配器，仅依据真实树 Fork 生成经验。
- 用户明确要求后，旧 249 tree run t069-tree-formal-20261111-v2 于 2026-11-12 15:21:48 CST 在最后完整 Step 158 后被 SIGSTOP 挂起，PGID 3699749；即时检查 Ray/Actor 为 T 状态。**重要事后异常：15:29:50 再查时该进程组已完全消失，不再可用 SIGCONT 恢复；没有发 SIGTERM/SIGKILL、没有自行重启，原因未查明。** 最后完整日志 Step 158，最后留存 checkpoint Step 150；恢复需要新的明确安排，历史文件保留。

## 6. 验收状态

- 在新条件分支执行 Reflexor v3 + 全部树相关 CPU 测试：**94 passed，1 skipped，1 xfailed**（另有依赖/Checkpoint 警告）。
- 已对新条件树启动脚本执行 shell 语法检查，确认显式 beta 0.1 和阈值 -0.1，通用 YAML 仍 -0.25。
- 真正 3B 多 GPU + Ray/vLLM 流程烟测及新正式训练**尚未运行**；不把 CPU 测试当成训练收益验证。

## 7. T70 分支与学习率修订（2026-11-12）

- 通过 Git worktree move 从 T069/conditional-natural4-worktree 搬迁至 T070/conditional-natural4-worktree，并将**分支精确重命名为 t70**，而不是创建一个没有历史的新分支。
- 新的脚本名：examples/grpo_trainer/run_t70.sh；新目录：/home/zehao/echo-collab/T070/conditional-natural4-sft/v1；W&B run ID/group/tags 均使用 t70。
- 显式 Hydra 启动覆盖项：actor_rollout_ref.actor.optim.lr=2e-5、actor_rollout_ref.actor.optim.lr_warmup_steps=5、actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.0。
- 依据 verl/workers/fsdp_workers.py，Actor scheduler 在每个**外层训练 Step 完成后**更新一次，不跟随每个 Optimizer Mini-batch 更新。当前 constant warmup 由 0 线性到全 LR，Step 1 的首次权重更新按现有调度器实现使用 0 LR，Step 6 开始使用完整 2e-5；是否希望第一步非零可另行讨论。
- T069 训练数据仍在原路径，只读复用；T70 的结果和 checkpoints 使用独立新目录。旧的 Reflector-v3 工作树和 tree/t069 分支未改动。

## 8. 尚需讨论（不擅自改算法）

1. **评测安排（已复用；需独立运行队列）**：T70 保持训练内部验证关闭，复用原 t069_eval_queue.py 与 t069_evaluate.py，每 10 Step Dev1000、每 50 Step 6125。修复 T70 数据路径兼容，增加 --data-dir 和独立启动入口 run_t70_eval_queue.sh。不会重复启用训练内部验证。
2. **真实 GPU/工具链烟测（高优先级）**：CPU 回归不是 3B + Ray/vLLM/检索/Actor-SFT 的端到端验证。启动正式 200 Step 前应在不占用别人 GPU 任务的情况下检验 1 个极小训练更新及配置实际生效。
3. **Reflector 候选来源（已确认）**：只用真实树分叉，不再用自然成功/失败轨迹补足 Reflector 候选。自然轨迹继续参与 GRPO、SFT。
4. **学习率及训练稳定性（高优先级）**：2e-5 是原 5e-6 的 4 倍；联合 GRPO+SFT(0.05)+KD 的影响尚未实测。重点观察 entropy、grad_norm、PPO KL/clipfrac、无经验准确率、SFT/KD 活跃 Token 和 Loss，不能因为训练 loss 短期下降就直接下结论。
5. **触发阈值合理性**：自然 2/4 跳过树，与 0/4、1/4 触发树是方案决定；建议统计 0–4 的分布、触发率、树救援率和全组非零 Advantage Token。后期 4/4/0/4 可能导致 GRPO 信号不足，已有成功 SFT 可缓解但尚未验证收益。
6. **恢复与持久化（工程风险）**：脚本默认阻止在有内容的旧 Run 目录上再次启动，能防覆盖，却也阻止崩溃后的直接自动恢复。若需要恢复必须先设计有明确授权、与正确 checkpoint 对应的恢复入口；不能用旧 T069 已混淆 ID 的 checkpoint 作为 T70 干净初始化。

## 9. 本次审查结论与已落实修改

- OPSD 全局损失系数 1.0，T70 正式门控要求本轮新增经验、当前动作不同和 p_R>p_L；分叉权重 w=(p_R-p_L)*n_R/(n_R+1)。KL 按 GRPO Token 分母归一化。成功轨迹 SFT 系数单独为 0.05。
- 现有评估并非缺失：run_t062_perf.sh 的 test_freq=-1 只关闭内置评估，原有独立队列继续保留。为 T70 增加外部 T069/data 数据路径支持以及 run_t70_eval_queue.sh；单独运行队列，不会自动启动训练。
- 旧审查关于分支尚有未提交改动的判断已过时：配置已在 704b585 提交，本次变更还会产生新的独立提交。
- KD 门控历史记录：本段原先检查到的「左全失败、右有成功」旧实现已于 T70 最新修订中被替换为 p_R>p_L，且排除同动作。根据此前用户已明确确认的方案，不加入历史 Utility≥0 的硬门控，不对祖先 T 路径额外乘 0.5。
- 按用户决定不改难题附加树叶子导致的训练权重、PG/KD/SFT 相互作用、自然触发阈值、2e-5 学习率和 warmup。
- 工程约束：run_t70.sh 拒绝非空输出目录，恢复需专用入口；GPU/Ray/vLLM 完整 3B 烟测仍未执行。

## 10. T70 训练效率改进（2026-11-12，训练未启动）

### 审查实测证据与结论

- T069 Step 139–158 的 20 Step 均值：总耗时 695.0771s，其中 Actor 更新 385.4089s（55.4%）、Rollout 193.4349s（27.8%）、Old Logprob 109.1769s（15.7%）。这是旧 T069 的实测，**不是 T70 运行速度**。
- T069 Step 150 的真实树快照只读加载：**3282 个展开行，2120 个唯一节点**，重复行占 35.4%。
- CPU 准备基准：使用上述真实树的已保存 Tensor，无经验 SFT Prompt 使用相同长度的测试替代 Tokenizer（并非真实 GPU 训练）。旧版 pack=2.243s、dedup=2.952s、合计=5.195s；先唯一节点预打包 pack=1.143s、dedup=2.653s、合计=3.795s，减少 1.400s（准备流程相对缩短 26.9%）。新旧最终 Batch 核心 Tensor 与 Metadata SHA-256 完全一致。**不能据此声称整体训练加速 26.9%**；相对旧训练约 695s/Step，仅约 0.2%。
- 历史完整树 158 份，累计 78,146,566,242 Byte（约 72.8 GiB）。不删除历史文件，当前保留逐步完整快照以供 replay、审计和经验暴露重算；新增保存耗时指标。

### 已实现的保守优化

1. 只对 T70 使用的 node_mixed 去重模式，先按 node_id 提取**唯一代表行**（若该节点包含 KD canonical 行，则优先选取），仅为唯一行构造 Prompt、Response、Teacher 和 SFT Student Tensor。
2. 原始展开的每叶子 advantage/reward/KD/SFT 信息仍参与 w_tok、A_pos、A_neg、sft_weight 汇总；共享节点的无经验 SFT Prompt 依据**全部**成功叶子决定。prepare_dedup_batch 新增可选 expanded_rows，无参数时兼容旧全展开路径。
3. 不修改 Rank/Mini-batch 分配；新增 dedup/rank_unique_forward_tokens_* 与 dedup/rank_estimated_forward_tokens_*，后者考虑 KD Teacher、with-T 无经验 Student 及成功 SFT 额外前向。它们是 Token 数近似，不是实测 FLOPs。
4. 新增 natural/exact_duplicate_episode_rate 和 natural/exact_duplicate_node_rate，以 Prompt+Response+Attention Mask 的精确 Token 字节计算；只监控，不合并自然轨迹、不丢掉四条独立奖励。
5. 新增 perf/pack_training_rows_seconds、perf/dedup_batch_seconds 和 perf/tree_snapshot_seconds；同时监控展开/预打包/真实/Padding 行数与 Rank 计算负载。
6. 原 question_grouped 路径及回放调用保持兼容；原 T069、251 基线、原日志与 Checkpoint 未改动。T70 的 LR=2e-5、warmup=5、GRPO/OPSD/SFT Loss 和条件门控均未更改。

### 有意暂缓的优化

- **不跨自然轨迹复用 Actor 前向**：T70 尚无真实重复率；相同 Action 不意味着完全相同 Token 序列或奖励。
- **不修改 GPU 分组**：历史 Padding 比例约 4.3%，暂无严重负载失衡证据；先看新 Rank 负载指标。
- **暂不改变完整树逐步保存**：现有 replay、审计与 scripts/t066_recount_retrieval.py 依赖逐步树快照，磁盘还剩约 40 TiB；先测树保存时间。
- **不合并 OPSD/SFT 前向、不改 SearchEnv 线程池**：没有足够 T70 Profile 支持重写核心 GPU 计算。

### 验证状态

- T70 0/4～4/4 与 SFT 开关的 10 组完整 Tensor/Metadata 等价对照，加上共享节点负/正 Advantage 与 KD canonical 选择。
- 基于 T069 Step 150 真实树的 CPU Benchmark，新旧最终 Batch SHA-256 完全一致。
- 新增自然重复率与 Rank 额外 Token 负载指标测试；尚未进行 3B 多 GPU 端到端训练或正式 T70 实验。

## 11. OPSD 条件和权重纠正（用户已明确确认，2026-11-12）

曾冻结的旧版「L 全失败、R 至少有一条成功；w=k_R/(n_R+1)」已被用户更晚确认的新方案取代。之前审查遗漏了此项决定，误把旧门控当成 T70 最终要求。

**T70 最终实现**（agent_system/tree/core.py，TreeRollout.evaluate_forks）：

- gate = bool(new_experience_ids) and not same_action and (p_R > p_L)；
- w = (p_R-p_L) * n_R/(n_R+1)，其中 p_L/p_R 是左右真实终局叶子的平均成功率，n_R 是右子树叶子数；
- 不把 utility_mean >= 0 设成硬门控，不加入含祖先 T 路径的额外 0.5 系数；
- 仍只向本轮真实经验 Fork 生成 OPSD 监督；自然采样不构成 KD Fork；
- Utility 计算逻辑、Reflector、GRPO、SFT、OPSD 全局系数（1.0）、Top-k=20 及 Actor 的 KL 数值归一化均保持不变。

回归：120 passed、1 skipped、1 xfailed。测试包括正增益但左侧已部分成功、零增益、同动作、负历史 Utility、不含/含祖先 T 和右子树小样本收缩。

只读历史数据对照：T069 Step 150 的旧门控触发 37 个 Fork，新规则在同一棵历史树上会触发 47 个 Fork（旧独有 3 个同动作、新增 13 个正增益分叉）；新门控平均分叉权重约 0.44494。该统计不是 T70 正式训练数据，也不能直接推断 T70 的最终收益。

## 13. T70 249 四卡训练内评估（最终启动规则）

用户明确要求在 249 上使用训练相同的四张 GPU 2–5 做评估，不再使用任何独立 GPU 评估队列。

- 训练器的 val_dataloader 仍是 T069/data/dev1000.parquet，trainer.test_freq=10，trainer.val_before_train=True，每次 data.val_batch_size=128。验证执行 TrajectoryCollector.multi_turn_loop(is_train=False)，贪心且不注入经验。
- 额外 t70_test6125_dataloader 使用 T069/data/test6125.parquet，t70_test6125_freq=50，每 50 Step（以及 Step 0 和最后一步）同步交换验证数据加载器并调用同一个 _validate()，结束后恢复 Dev1000 数据加载器。
- 训练进程在验证期间停止参数更新，复用同一 Ray Actor 四张卡，而不是另开 vLLM 单卡评估。
- 记录 Dev1000 val/strict_em 与 val/macro_em，Test6125 test6125/strict_em、test6125/macro_em 和每数据集 *_success_rate；Micro EM 按所有题目的独立轨迹等权，Macro EM 按七个数据集均值。
- 移除 T70 专用 run_t70_eval_queue.sh，不再设置 T069_RUN_DIR 触发旧队列补录，不执行评估 sidecar。旧通用评估模块仅保留以兼容 T069，不影响 T70。
- OPSD 全局系数 0.1，SFT 0.05；保持已有 p_R>p_L 门控、LR=2e-5 和 5 Step warmup 不变。

## 14. 启动前 Hydra 配置校验记录

首次 249 T70 启动在 Hydra 配置校验阶段立即退出（未启动 GPU 训练），提示旧的 min_delete_questions 字段不再属于 YAML 结构化配置；现移除该废弃启动覆盖项。启动失败的原始 train.log 保存在 T70 专有归档目录，重新启动前先执行不加载模型的 --cfg job 配置解析，避免重复浪费 GPU。

## 15. Ray Unix Socket 路径修正

在 249 上再次尝试启动时 Hydra 配置已通过，但 Ray 初始化 AF_UNIX socket path 超过 107 bytes，进程立即退出、未进入 GPU。原因是旧脚本把 T062_ROOT 指向深层 RUN_DIR，导致 RAY_TMPDIR=RUN_DIR/r；已改为 T062_ROOT=/home/zehao/echo-collab/T070，让 Ray 临时目录为更短的 T070/r，同时 PERF_RUN_DIR 仍使用隔离的 T70/conditional-natural4-sft/v1。失败日志在 T70 内归档。


## 15. 初始 6125 评估调整和 Ray 路径修正

用户最终要求 Step 0 只评估 Dev1000，Test6125 从 Step 50 开始，每 50 Step 一次；最后一步按计划评估。已删除 fit() 前置 Test6125 调用，并在 t70_test6125_due 中明确 global_steps>0。

第二次 249 启动在 Ray 初始化阶段因 AF_UNIX socket 路径超过 Linux 107 字节限制退出，尚未进入 GPU 训练。T70 launcher 现在使用短 T062_ROOT=/home/zehao/echo-collab/T070 （Ray 使用其 r 子目录），PERF_RUN_DIR 仍独立指向 T70/v1 checkpoint/log；失败的旧日志独立归档，不影响正式运行。
