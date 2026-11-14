> **2026-11-12 19:57现场验收**：V2 Step1真实四卡第一次参数SHA一致（c581cc4d...），Step1梯度探测 KD/PG=3.99%、SFT/PG=18.19%，Step2仍待二次SHA。详见 docs/t70/T70_V2_训练交接文档_20261112.md。

> **V2更新提示**：2026-11-12 用户已确认新一轮运行使用OPSD有效KD Token独立均值、OPSD=0.05、成功SFT=0.05；V1=0.1与PG分母仅作历史。详见 docs/t70/T70_V2_OPSD_ACTIVE_AND_SFT_GRAD_20261112.md。

# T70 项目索引与最终方法口径（2026-11-12）

> 本文件只整理项目，不修改运行中的训练。核心训练代码基线：fb91896。后续文档提交不应被理解成训练代码版本变更。

## 一、最终方法（避免按旧审查回滚）

**T70 = 每题4条无经验自然采样 → 仅自然成功数≤1启动完整T/S经验树 → 所有同题自然/树叶子统一GRPO → 正收益真实分叉的弱OPSD（有效KD Token独立均值，系数0.05）+成功SFT（0.05）→ 真实树Fork驱动经验生命周期。**

- OPSD门控：R本次真正注入新经验、左右当前规范化动作不同、右子树成功率严格大于左子树。权重是 Δ×n_R/(n_R+1)，其中Δ=p_R−p_L。不是旧版“L全部失败，R至少一个成功”。
- 历史Utility为负**不是**OPSD硬门控；存在祖先T也**不**额外乘0.5。
- 自然轨迹仅参与GRPO/成功SFT，不参与Reflector候选、Utility或KD分叉。
- Reflector：最多40个真实树同状态Fork → API最多选择10个 → 输出0–5条经验；不足40个不拿自然轨迹补足。
- 同一题的4条自然轨迹独立执行，即使Token完全相同也不会提前合并奖励；共享树Node训练前向按精确保留正负Advantage权重去重。
- 经验硬删除为四条件AND：Utility≤−0.1、真实注入暴露≥5、连续负收益管理窗口≥2、创建后保护≥20 Step；容量清理另计。
- 新经验ID持久化递增，不复用已删除ID。不要将历史审查稿中的旧设定当成最新方案。

## 二、训练运行与评估

- 工作树：/home/zehao/echo-collab/T070/conditional-natural4-worktree
- 分支：t70
- 启动脚本：examples/grpo_trainer/run_t70.sh
- 当前运行目录：/home/zehao/echo-collab/T070/conditional-natural4-sft/v2-opsd-active-005
- 设备：249服务器，GPU 2–5，4张RTX3090，训练和验证共用。
- 模型：Qwen2.5-3B-Instruct；LoRA rank16/alpha32；LR 2e−5；前5个外层Step Warmup；200 Step、每Step 256题；最多4个Search Turn。
- 自然探索：每题4条完整无经验轨迹；成功0或1才触发树。
- 目标：GRPO为主，OPSD全局系数0.05，成功SFT系数0.05。
- 评估：Dev1000从Step0开始、每10 Step；Test6125 **Step0不测**、从Step50开始每50 Step。使用同四张GPU进行无经验贪心多轮Search，不启动T70独立评估队列。
- W&B：项目tree-edge-search；Run t70-conditional-natural4-opsdactive005-v2-20261112。
- 初始经验库：本次启动日志为11条Task+24条Step种子经验，并非旧T069最后一次训练的经验库。
- 恢复限制：全新启动器拒绝复用非空run目录，不等于可直接运行原脚本进行断点恢复。

## 三、代码模块导航

| 模块 | 文件 |
|---|---|
| Tree/Node/Fork/Leaf、OPSD门控和权重、组优势 | agent_system/tree/core.py |
| 自然4采样、条件树Rollout、Token打包、树快照 | agent_system/tree/runner.py |
| 同题共享节点去重、精确保留A_pos/A_neg/w_tok | agent_system/tree/question_batches.py |
| 精确GRPO Node目标 | agent_system/tree/dedup.py |
| OPSD Teacher Top-k Reverse KL | agent_system/tree/loss.py |
| Actor中的PG、OPSD、SFT梯度聚合 | verl/workers/actor/dp_actor.py |
| 真实树Fork候选生成和轨迹选择 | agent_system/tree/reflector_selection_v3.py |
| API从40候选最多选10个 | agent_system/tree/reflector_case_selector_v3.py |
| 五步Utility EMA、Reflector调度、删除、树状态恢复 | agent_system/tree/training.py |
| 检索/UCB/Task+Step经验池/ID和删除 | agent_system/memory/experience_memory.py |
| 0–5条经验生成和校验 | agent_system/memory/experience_updater.py |
| 训练和验证调度 | verl/trainer/ppo/ray_trainer.py |
| 同四卡Test6125评估调度 | agent_system/tree/t70_inprocess_eval.py |
| 启动参数 | examples/grpo_trainer/run_t70.sh |
| 单元和等价性测试 | tests/tree/ |

## 四、论文和实验说明

本方法拟回答：**困难问题上条件触发的经验树探索，能否以GRPO为主、弱蒸馏为辅，提升学生在不调用外部经验时的自主多轮Search QA能力？**

训练数据：NQ 78,667 + HotpotQA 89,947，共168,614题。Dev1000：NQ500 + Hotpot500。Test6125：2Wiki、Hotpot、MuSiQue、NQ、PopQA、TriviaQA各1000，Bamboogle125。Micro EM按题数，Macro EM按7个数据集等权。

结果必须基于T70实际评估，不得由T069旧日志直接推断T70性能。值得做的最小消融是Natural-only（自然4+GRPO+SFT，不运行树/OPSD），防止学习率和采样差异掩盖树本身贡献。子树奖励差是随机执行下的关联证据，**不是严格识别的因果效果**。

## 五、文档优先级

1. 本项目索引：最终决策和源代码入口。
2. 本次《T70条件经验树方法说明书》（Word和Markdown）：完整研究方法、公式、实验配置、论文Method草稿。
3. T70_IMPLEMENTATION_AND_REVIEW.md：保留历次修改的审计记录；其早期门控描述可能已过期，不能覆盖上述最终结论。

整理项目时不移动训练代码、不清理已有树快照、不触碰T069历史或正在运行的249训练。
