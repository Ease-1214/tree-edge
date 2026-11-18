# T75 AI 接手说明（Handoff & Operational Guide）

> 接手前请依次阅读：`USER_COLLABORATION_PREFERENCES.md` $\rightarrow$ `T75_METHOD_AND_FRAMEWORK.md` $\rightarrow$ `docs/T75_ARCHITECTURE_AND_INTERFACES.md` $\rightarrow$ 本文件。

---

## 一、仓库与线上训练当前状态（2026-11-17）

1. **代码分支与多端同步状态**：
   * 仓库：`git@github.com:Ease-1214/tree-edge.git`，活跃分支：`t75`
   * 部署路径：
     * `31`（`172.18.166.31`，4×A40）：`/home/zehao/echo-collab/T075/tree-edge`
     * `249`（`172.18.167.249`，8×3090）：`/home/zehao/echo-collab/T075/tree-edge`
   * 已合并落盘但**尚未重启生效**的最新改进（commit `ae74d82` 及之后文档提交）：
     * `runner.py`：纯本地确定性状态诊断摘要 `build_step_retrieval_query()`（修复嵌套括号正则，去除原题稀释）；
     * `experience_memory.py`：分池独立相似度门槛 `similarity_threshold=0.3` / `step_similarity_threshold=0.3`；
     * `experience_updater.py`：Prompt 上限放宽至 `180,000` 字符（不做丢例裁剪），优化 Rule 5 反重复细则以释放具体战术型 `step` 经验提取；
     * `training.py` & `reflector_selection_v3.py`：完整 5 步窗口候选累积 + 3 深度桶（`1 / 2 / >=3`）分层抽样最多 12 例；
     * `run_t75.sh`：`utility_ema_beta=0.2`，`eviction_utility_threshold=-0.005`，`eviction_negative_windows=3`，W&B 代理自动探测优先级 `29662 -> 29790 -> 7890 -> offline`。
2. **线上正在运行的训练任务**：
   * 目录：`31:/home/zehao/echo-collab/T075/t75-2xa40-r64-20261117-085804`（运行在 `CUDA_VISIBLE_DEVICES=2,3`，启动时代码快照为 `2b5339d`）
   * W&B Cloud Run：`wuzehao1214-sun-yat-sen-university/tree-edge-search/runs/rvjkjv5k`
   * 进展（截至 Step 30 / 200）：`train_natural_pass_rate` 从 `0.294` 稳步升至 `0.378`（Step 29 峰值 `0.4180`），`valid_action_ratio = 0.9995`，`bank_size = 36`，`utility_positive_rate = 61.1%`。预计 Step 50 触发首次全量验证集评测。

---

## 二、环境依赖与四大服务端口规范

> **安全红线**：
> 1. **31 服务器的 GPU `0,1` 属于其他用户，严禁触碰！** 任何命令必须限定 `CUDA_VISIBLE_DEVICES=2,3`（或纯 CPU 执行）。
> 2. **严禁打印或提交任何 API Key**（`/home/zehao/.edge_reflector.env`、`/home/zehao/.netrc` 等）。

| 服务 / 依赖 | 31 / 249 地址或路径 | 宿主机与高可用守护机制 |
| :--- | :--- | :--- |
| **Python 环境** | `/home/zehao/.conda/envs/tree-edge/bin/python` | Python 3.11 + PyTorch 2.8 + vLLM 0.11 + flash-attn |
| **Base 模型** | `/home/zehao/Tree-GRPO/model/Qwen2.5-3B-Instruct` | 本地权重，`bf16` 冻结底座 + `fp32` LoRA (`r=64, alpha=128`) |
| **Search API** | `http://127.0.0.1:29661/retrieve` | `248:18125` 经 SSH `-R` 转发，`248` 上 `watchdog_wandb_proxy.py` 自动重连 |
| **W&B HTTP 代理** | `http://127.0.0.1:29662`（首选）/ `29790`（本机备份） | `248:7890` 经 `watchdog_wandb_proxy.py` 每 60s 探测 `api.wandb.ai` 并自动测速切换健康节点 |
| **Reflector API** | `http://127.0.0.1:29663/v1`（`gpt-6.1-sol`） | `251:29663` `reflector_gateway.py` 三 Key 故障切换并经 SSH `-R` 转发 |
| **Embedding API** | `http://127.0.0.1:29700`（`Qwen3-Embedding-0.6B`） | `248:29700` `embedding_server.py` 共享向量编码服务，训练进程零额外显存占用 |

---

## 三、标准健康检查与正式启动命令

### 1. 启动前四端口一键预检
```bash
# Search (29661) / W&B Proxy (29662) / Reflector (29663) / Embedding (29700)
curl -sS -m 5 http://127.0.0.1:29663/healthz && echo " [29663 OK]"
curl -I -sS -m 5 -x http://127.0.0.1:29662 https://api.wandb.ai | head -n 1
curl -sS -m 5 -X POST http://127.0.0.1:29661/retrieve -H "Content-Type: application/json" -d '{"queries":["Who discovered penicillin?"],"topk":1,"return_scores":false}' | head -c 120 && echo " [29661 OK]"
ss -tlpn | grep -E '29661|29662|29663|29700'
```

### 2. 标准启动 / 断点续跑命令
```bash
cd /home/zehao/echo-collab/T075/tree-edge
source ~/.edge_reflector.env

# 31 服务器 (仅用 GPU 2,3):
GPU_PROFILE=2xa40 CUDA_VISIBLE_DEVICES=2,3 bash examples/grpo_trainer/run_t75.sh

# 249 服务器 (4x3090 或 8x3090):
GPU_PROFILE=4x3090 CUDA_VISIBLE_DEVICES=0,1,2,3 bash examples/grpo_trainer/run_t75.sh

# 从指定 Checkpoint 恢复:
GPU_PROFILE=2xa40 CUDA_VISIBLE_DEVICES=2,3 RESUME_PATH=/path/to/checkpoints/global_step_50 bash examples/grpo_trainer/run_t75.sh
```
