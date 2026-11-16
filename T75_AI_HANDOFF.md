# T75 AI 接手说明（Handoff）

* **当前分支**：`t75`
* **基础模型**：`Qwen2.5-3B-Instruct`（BF16 base + LoRA r=64 / alpha=128）
* **Embedding 模型**：`Qwen3-Embedding-0.6B`（本地加载，不走远程 HTTP API）
* **Reflector API**：`http://127.0.0.1:29663/v1`（模型 `gpt-6.1-sol`，由 251 服务器 `reflector_gateway.py` 提供三 Key 自动故障切换并通过 SSH `-R` 映射至 31；详见 `REFLECTOR_API_31_251_AI_GUIDE.md`）
* **Search 规范 `task_type`（11 类）**：`general`、`direct_retrieval`、`entity_attribute_lookup`、`multi_hop_reasoning`、`comparison`、`temporal`、`numeric`、`list_intersection`、`source_identification`、`location_chain`、`answer_normalization`

---

## 一、已验证状态

1. **单元测试（已在 31 服务器 `/home/zehao/.conda/envs/tree-edge/bin/python` 验证）**：
   * `tests/test_t75_core_and_reflector.py`
   * `tests/test_t71_edge_fingerprint_eviction.py`
   * `tests/test_t71_edge_single_stage_extraction.py`
   * `tests/test_t71_proxy_preflight.py`
   以上 4 个文件 **22 项测试全部通过**；`verl.trainer.ppo.ray_trainer.RayPPOTrainer` 在服务器 Conda 环境可正常导入。
   *注*：`tests/test_actor_metric_totals.py::test_inactive_microbatches_stay_in_coverage_denominator` 为既有历史失败（`0.6 != 0.24`），与 T75 无关。
2. **离线与真实在线 Reflector API（`gpt-6.1-sol`）闭环验证通过**：
   * `python tools/test_t75_reflector_closed_loop.py --live` 在 31 服务器实测通过（`CLOSED_LOOP_OFFLINE_VERIFICATION_OK` + `CLOSED_LOOP_LIVE_API_OK`）：
     * **反重复验证（完整 24 条经验库）**：当测试案例策略已被现有经验库覆盖时，`gpt-6.1-sol` 严格遵守 `STRICT ANTI-DUPLICATION` 规则返回 `[]`（耗时约 `4.8s`），不凑数生成重复规则；
     * **新增提取验证（最小 2 条种子库）**：对 `depth=1` 分叉准确提取出 `experience_type="task"`（`location_chain`），对 `depth=2` 分叉准确提取出 `experience_type="step"`（`entity_attribute_lookup`），无实体泄露，耗时约 `24.0s`。

---

## 二、31 与 251 服务器关键环境

* **31 服务器（`zehao@172.18.166.31`）**：
  * Conda Python：`/home/zehao/.conda/envs/tree-edge/bin/python`（Python 3.11 + torch 2.8 + vLLM 0.11）
  * GPU：4x NVIDIA A40（48GB）
  * Embedding 路径：`/home/zehao/echo-collab/T061/assets/Qwen3-Embedding-0.6B`
  * Reflector 凭据文件：`/home/zehao/.edge_reflector.env`（客户端鉴权 Key，指向 `http://127.0.0.1:29663/v1`）
  * Search API：`http://127.0.0.1:29661/retrieve`（Search-R1 批量接口）
* **251 服务器（`zehao@172.18.167.251`）**：
  * Reflector 三 Key 自动切换网关：`/home/zehao/echo-collab/T073/reflector_gateway.py`（每 Key 超时 `95s`，三 Key 最坏 `285s`；客户端默认超时已设为 `330s` / deadline `340s`）
  * 详细运维与故障排查见 `REFLECTOR_API_31_251_AI_GUIDE.md`

---

## 三、接手后常用命令

1. **检查 Reflector 网关健康状态并跑在线闭环验收（在 31 上）**：
   ```bash
   curl -sS --max-time 5 http://127.0.0.1:29663/healthz
   source ~/.edge_reflector.env
   /home/zehao/.conda/envs/tree-edge/bin/python tools/test_t75_reflector_closed_loop.py --live
   ```
2. **启动多卡训练**（按实际机器选择 `GPU_PROFILE`）：
   ```bash
   GPU_PROFILE=2xa40 CUDA_VISIBLE_DEVICES=2,3 bash examples/grpo_trainer/run_t75.sh
   # 或 4x3090 / 8x3090 / auto
   ```
