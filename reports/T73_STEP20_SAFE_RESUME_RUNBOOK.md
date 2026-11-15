# T73 Step20 安全切换方案（当前尚未切换）

## 环境及风险
- 原训练：/home/zehao/echo-collab/T073/t71replica-r64-eightgpu-eval50-source
- 原 user systemd 服务：t73-a40x2-3b-libcuda-20261115.service，使用 GPU 2 和 3。
- Step10 检查点含 LoRA 模型、两 rank 优化器+调度器+RNG 的 training_state_rank_[01].pt、DataLoader data.pt、tree_state.json。
- 原码的 checkpoint 加载器确实恢复两 rank 优化器+调度器+RNG，正式启动仍必须见到两条 optimizer_restored=1。
- 每10步保存；只有 latest_checkpointed_iteration.txt 显示20且全部文件完整才能准备。
- W&B 远程存在原 Run，ID t73-qwen25-3b-r64a128-a40x2-full11015-20261115；项目 wuzehao1214-sun-yat-sen-university/tree-edge-search，续训 WANDB_RESUME=must。
- 31没有找到 Slurm/PBS/LSF/Condor 队列指令或已启用的调度服务；这不排除人工或平台外部排队。无法证明不存在候补者或能保留这两卡。GPU不安全时绝不停止旧训练。

## 独立修复
- 分支 fix/t73-reflector-type-validation；工作树 /home/zehao/echo-collab/T073/t73-reflector-type-validation-offline
- 新版本强化 Task/Step 的 question-only 触发标准，生成后单次批量语义校验。错误/不明确的不入库，遇接口错误 fail closed。
- 修订 task_000012、task_000016：仅重写 trigger、state_cues、title、when_to_apply、retrieval_obs。完整保留 ID、来源证据、Utility、曝露数、创建 Step、对应行动。
- 新版启动脚本仅允许独立恢复运行目录，要求 T73_SAFE_RESUME=1、明确 T73_RUN_DIR、Step20 两 rank 检查点；强制指定 resume_from_path=global_step_20 并设置 WANDB_RESUME=must。
- 新版不会触碰原始运行目录。目标目录 /home/zehao/echo-collab/T073/qwen25-3b-r64a128-a40x2-classification-resume20-20261115

## Step20 完成后复制，绝不停止原训练
进入独立工作树，执行：
PYTHONPATH="$PWD/tools:$PWD" /home/zehao/.conda/envs/tree-edge/bin/python tools/t73_prepare_step20_resume_copy.py

此脚本检查 latest tracker、actor 文件、两个优化器、data.pt、tree_state.json 后，完全复制 Step20 检查点，在新的复制版本里修订经验库并写 resume_preparation_manifest.json。无 GPU 计算，也不修改原检查点。

## 真正切换的硬条件
1. 先明确 GPU 2、3 没有候补及有保卡或连续使用许可。若不能确认，保持旧服务运行。
2. 只从已完成且没有产生新的已完成训练进度的 Step20 切换；若旧服务已继续 Step21+，不可随意回滚，需考虑下一个完整检查点。
3. 对两个 rank 校验 optimizer_restored=1、LoRA 配置、DataLoader、树状态、经验库、Embedding 同步和 Global Step，任何失败立即禁止继续。
4. 同一 W&B Run ID 必须存在，恢复用 WANDB_RESUME=must，检查 Step21 之后续写到原 Run。新经验指标从续训开始出现。
5. 不删除原始日志或检查点，保留回滚通路。

## 已完成离线测试
- 真实 Step15 历史经验库修订演练通过，经验数量与所有受保护统计字段保持不变。
- 两条修订经验使用实际 Reflector API 分类验证均为 task。
- Step20 尚未保存时，准备脚本正确拒绝执行。
- 原 W&B Run 通过只读 API 确认存在、仍为 running。
- 尚未进行实际 GPU 重启/恢复测试，不能声称无缝切换已经完成。
