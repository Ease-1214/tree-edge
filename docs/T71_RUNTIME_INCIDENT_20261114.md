# T71 首次运行异常排查与代理修复（2026-11-14）

## 故障

- 初次实验目录：`/home/zehao/echo-collab/T071/conditional-natural4-sft/top46-best1-opsd-delta-flatfork-um0p01`。
- 2026-11-14 07:10 CST 启动，Step 1～4 完成；Step 5 的树已采样并写出 `tree-private/tree-step5.pt`，但第 5 次训练未提交，也未执行经验新增和经验库更新。
- 2026-11-14 08:38 CST 主进程退出，`T71_EXIT_CODE=1`。原实验和 W&B 数据均保留。
- 根因：`agent_system/tree/training.py::manage_tree_experiences` 每 5 Step 第一次构造 `ExperienceUpdater`；`OpenAI` 客户端初始化经过 `httpx2`，环境继承了 `ALL_PROXY=socks5h://...`、`all_proxy=socks5h://...`，环境又缺少 `socksio`。因此异常：`ImportError: Using SOCKS proxy, but the 'socksio' package is not installed`。
- 此时的训练、梯度、经验选择、OPSD、CUDA、NCCL 均未因其本身的错误报错。四卡 Worker 随父进程退出。原设置 `save_freq=10`，**没有**可恢复 Step 4 权重的 checkpoint。

## 修复（只针对 T71）

- 在专用 `examples/grpo_trainer/run_t71.sh` 中，读取已有反思服务凭据后，`unset ALL_PROXY all_proxy`。保留 HTTP/HTTPS 代理及本地 NO_PROXY，避免全局修改和额外安装依赖。
- 加入 **GPU 分配前**的 `ExperienceUpdater` 客户端构造 preflight。下次如果环境依赖仍缺失，会在加载模型前失败，而不是白跑 4 Step。
- 按相同代理设置，以真实 `gpt-6.1-sol` 接口执行了极小 completion 测试，成功返回 `READY`。未更改经验生成、Utility、OPSD、检索或优化器逻辑。
- 增加 `tests/test_t71_proxy_preflight.py`，覆盖启动脚本顺序及 HTTP 代理存在、SOCKS 环境被清理时的 OpenAI 客户端初始化。
- 初始经验库沿用原始 `memory_data/search/state_aware_experiences_v2.json`（11 Task + 24 Step），不续接 T70 动态经验。

## 重启原则

保留第一次失败运行的全部日志、结构快照、W&B 历史。新目录和新 W&B Run ID 中**从 Step 0 重新开始**，因为第一次失败运行没有保存模型 checkpoint。禁止假称从 Step 4 恢复。通过此次修复仅能证明 SOCKS 初始化错误消失；正式第 5 步反思/经验写入仍需观察实际运行。
