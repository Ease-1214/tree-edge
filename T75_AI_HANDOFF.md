# T75 AI 接手说明（Handoff & Pre-Launch Checklist）

> 接手 AI 请先阅读本文件与 `T75_METHOD_AND_FRAMEWORK.md`、`REFLECTOR_API_31_251_AI_GUIDE.md`。所有核心改动与测试均已在分支 `t75` 完成。

---

## 一、当前完成度与验证结果

1. **已完成并提交至 `t75` 分支的全部改动**：
   * **模块 1（V3 双池经验库）**：剔除 ALFWorld 污染，拆分 `task_experiences`（11 条）与 `step_experiences`（13 条），对齐 11 个 Search `task_type`。
   * **模块 2（检索状态摘要与门控）**：`build_step_retrieval_query()` 构造状态摘要，`0.4` 余弦相似度硬门槛初筛 + UCB 重排选 Top-1，全低于 `0.4` 则跳过分叉。
   * **模块 3（无偏递归子树价值）**：`TreeRollout.evaluate_forks()` 自底向上递归计算子树期望，消除深层多叶分支权重偏差。
   * **模块 4（Reflector 分层抽样与单次调用）**：仅选 `delta < 0` 分叉，按 `depth=1` / `depth>=2` 分桶抽样；默认 1-call 提取（`ENABLE_EXPERIENCE_TYPE_VALIDATION=0`），Prompt 注入已有经验与 `STRICT ANTI-DUPLICATION` 铁律，入库 `0.85` 向量去重；客户端默认超时延至 `330s`（deadline `340s`）。
   * **模块 5（GRPO 格式惩罚解耦与 SFT 掩码）**：先对 `0/1` 原始奖励做组内标准化，再对 `valid=False` 扣 `0.1`；`sft_weight = float(leaf.reward == 1 and node.valid)`。
   * **模块 6（训练性能优化、多卡启动与 W&B 重写）**：
     * 向量化 `EpisodeRewardManager` 与 `apply_invalid_action_penalty`，默认关闭每步同步写盘 `tree-step*.pt`（需开启设 `SAVE_TREE_SNAPSHOT=1`），精简 `compute_log_prob` 传输；
     * 新增统一启动脚本 `examples/grpo_trainer/run_t75.sh`（支持 `2xa40`、`4x3090`、`8x3090`、`auto`）；
     * 移除旧版 `DASHBOARD`/`dashboard_metrics`，W&B 与 `step_metrics.jsonl` 统一使用 `build_clean_wandb_metrics()` 收敛为 6 个面板（`1_eval/`、`2_tree_exp/`、`3_memory/`、`4_loss/`、`5_behavior/`、`6_perf/`）。
2. **实测通过项**：
   * 31 服务器 Conda 环境（`/home/zehao/.conda/envs/tree-edge/bin/python`）下 4 个核心测试套件共 **22 项单元测试全部通过**，`RayPPOTrainer` 正常导入；
   * 31 服务器上运行 `python tools/test_t75_reflector_closed_loop.py --live` 真实调用 `gpt-6.1-sol` 通过（`CLOSED_LOOP_OFFLINE_VERIFICATION_OK` + `CLOSED_LOOP_LIVE_API_OK`）。

---

## 二、环境依赖与三大服务调用方式

| 依赖项 | 位置 / 地址 | 调用与检查方式 |
| :--- | :--- | :--- |
| **Python 运行环境** | 31：`/home/zehao/.conda/envs/tree-edge/bin/python` | Python 3.11 + PyTorch 2.8 + vLLM 0.11 + sentence-transformers + openai |
| **Base 模型** | 31：`/home/zehao/Tree-GRPO/model/Qwen2.5-3B-Instruct` | `run_t75.sh` 默认通过 `MODEL_PATH` 加载 |
| **本地 Embedding** | 31：`/home/zehao/echo-collab/T061/assets/Qwen3-Embedding-0.6B` | 进程内 `SentenceTransformer(..., device="cuda:0")` 加载，不走 HTTP |
| **Reflector API** | 31：`http://127.0.0.1:29663/v1`（模型 `gpt-6.1-sol`） | `source ~/.edge_reflector.env`；健康检查：`curl -sS http://127.0.0.1:29663/healthz`（由 251 `reflector_gateway.py` 三 Key 自动切换并经 SSH `-R` 转发） |
| **Search 检索 API** | 31：`http://127.0.0.1:29661/retrieve` | `POST {"queries": ["..."], "topk": 3, "return_scores": false}`（由 251 经 SSH `-R` 转发） |

---

## 三、正式启动训练前的必查清单（Pre-Launch Checklist）

> **重要警告（接手 AI 必看）**：
> 1. **31 服务器的 GPU `0,1` 属于其他用户（正在跑任务），严禁触碰！** 在 31 上启动训练时，务必指定 `CUDA_VISIBLE_DEVICES=2,3`（先用 `nvidia-smi` 确认 `2,3` 空闲）。
> 2. **切勿打印或提交任何 API Key**（`/home/zehao/.edge_reflector.env` 及 251 上的 `*.json` 密钥文件）。

### Step 1：直接在 31 或 249 的 T075 仓库目录启动（已同步最新 `t75` 分支）
两台训练机均已统一部署仓库至 `/home/zehao/echo-collab/T075/tree-edge`（并已配置好 `248 -> 31/249` 的 `29661`/`29700`/`29662` 隧道与 `251 -> 31` 的 `29663` 隧道）：
* **31 服务器（4×A40）**：`cd /home/zehao/echo-collab/T075/tree-edge`，使用 `GPU_PROFILE=2xa40 CUDA_VISIBLE_DEVICES=2,3 bash examples/grpo_trainer/run_t75.sh`（自动挂载 `TRITON_LIBCUDA_PATH` 并对齐 `api.wandb.ai`）。
* **249 服务器（8×3090）**：`cd /home/zehao/echo-collab/T075/tree-edge`，使用 `GPU_PROFILE=8x3090` 或 `4x3090` 启动（已补齐 `nq_hotpotqa_full11015.parquet`）。

### Step 2：在 31 上检查两个 SSH 反向隧道端口（`29663` 与 `29661`）
在 31 上执行：
```bash
# 1. 检查 Reflector 网关 (29663)
curl -sS --max-time 5 http://127.0.0.1:29663/healthz

# 2. 检查 Search 检索服务 (29661) —— 训练启动前必须确认此端口已转发！
curl -sS --max-time 10 -X POST http://127.0.0.1:29661/retrieve \
  -H "Content-Type: application/json" \
  -d '{"queries":["Who discovered penicillin?"],"topk":3,"return_scores":false}' | head -c 200
```
* 若 `29663` 未通：按 `REFLECTOR_API_31_251_AI_GUIDE.md` 第 6 节在 251 检查 `reflector_gateway.py` 并重建 `-R 127.0.0.1:29663:127.0.0.1:29663`。
* 若 `29661` 未通：在 251 上确认 Search-R1 检索服务已启动，并建立 `-R 127.0.0.1:29661:127.0.0.1:29661`（以及按需转发 `29662` HTTP 代理以供 W&B Online 同步，或设 `WANDB_MODE=offline`）。

### Step 3：启动训练并观察前 5 步验收指标
```bash
source ~/.edge_reflector.env
GPU_PROFILE=2xa40 CUDA_VISIBLE_DEVICES=2,3 bash examples/grpo_trainer/run_t75.sh
```
* **Step 1 重点观察**：显存峰值（`6_perf/max_gpu_mem_gb`）、`4_loss/pg_loss`、`4_loss/opsd_loss`、`4_loss/sft_loss` 是否正常非 NaN；
* **Step 5（首个经验反思窗口）重点观察**：`3_memory/reflector_submitted_forks`、`3_memory/added_task`、`3_memory/added_step`、`6_perf/reflector_time_s` 是否正常完成更新。
