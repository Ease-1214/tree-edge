# T70 V2 — OPSD独立Token均值 + 双0.05 + PG/KD/SFT梯度诊断

## 用户2026-11-12批准的最新决策

在T70 v1保持其它训练、环境、经验检索、L/R分叉、经验删除、Reflector及无经验评估协议不变的前提下：

1. OPSD使用**有效KD Token独立均值**，不再用全部GRPO Token：L_KD=sum(mask_KD*w_fork*KL)/sum(mask_KD)，其中 w_fork=1[新经验且当前动作不同且pR>pL]*(pR-pL)*nR/(nR+1)，分母**不含权重**。
2. 全局OPSD系数从0.1改0.05，SFT仍为0.05、仍用成功Token独立均值；GRPO主目标不变。
3. 四GPU训练通过所有Replica共享的每Minibatch有效KD Token总量归一化：Micro-batch分别计算KD活跃均值，再按活跃Token数/全局活跃数加权，避免零活跃GPU稀释有效监督。
4. Step1/50/100分别测PG/KD和PG/SFT：每对选同一份局部Micro-batch数据，KD优先、SFT优先允许两对不一样；四Replica梯度全局平均后比较范数/余弦。探测不调用Optimizer.step，恢复随机态。比例是局部探测值，不代表全Step平均。
5. W&B新增：grad/pg_norm、grad/kd_norm、grad/kd_to_pg、grad/pg_kd_cosine、grad/sft_probe_pg_norm、grad/sft_norm、grad/sft_to_pg、grad/pg_sft_cosine、grad/kd_probe_active_tokens、grad/sft_probe_active_tokens。
6. 更新已有W&B Loss指标：edge/distill_loss和kd/weighted_loss_active_tokens为KD有效Token均值；kd/weighted_loss_all_tokens保持旧GRPO分母供对照；kd/to_pg_loss_abs_ratio是0.05*独立KD损失相对PG损失的数值比例，不是梯度比例。
7. 旧实验保留在conditional-natural4-sft/v1。V2独立运行目录conditional-natural4-sft/v2-opsd-active-005，独立W&B Run ID t70-conditional-natural4-opsdactive005-v2-20261112，从Step0重新训练，Step0仅Dev1000，首次Test6125在Step50。

完整CPU测试、四卡Gloo归一化测试、Hydra解析均已验证。新的W&B上传及GPU训练状态需要以V2实际运行日志为准。

[executed on device: ollab-server-167-249 (a969b34c-d210-47f7-b6b7-00f31b07d34f)]
