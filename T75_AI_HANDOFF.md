# T75 AI 接手说明（Handoff）

* **当前分支**：`t75`
* **基础模型**：`Qwen2.5-3B-Instruct`（BF16 base + LoRA r=64 / alpha=128）
* **Embedding 模型**：`Qwen3-Embedding-0.6B`（本地加载，不走远程 HTTP API）
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
2. **离线闭环测试**：
   `python tools/test_t75_reflector_closed_loop.py` 已通过（`CLOSED_LOOP_OFFLINE_VERIFICATION_OK`）。

---

## 二、31 服务器（`zehao@172.18.166.31`）关键环境

* **Conda Python**：`/home/zehao/.conda/envs/tree-edge/bin/python`（Python 3.11 + torch 2.8 + vLLM 0.11）
* **GPU**：4x NVIDIA A40（48GB）
* **Embedding 路径**：`/home/zehao/echo-collab/T061/assets/Qwen3-Embedding-0.6B`
* **Reflector 凭据文件**：`/home/zehao/.edge_reflector.env`
  * 默认指向 `EXPERIENCE_UPDATE_BASE_URL=http://127.0.0.1:29663/v1`、`EXPERIENCE_UPDATE_MODEL=gpt-6.1-sol`
* **Search API**：`http://127.0.0.1:29661/retrieve`（Search-R1 批量接口）

---

## 三、当前未打通项（接手必看）

1. **31 服务器无直接外网，依赖 `server251` 反向隧道**：
   31 本机的 `29661`（Search）与 `29663`（Reflector）端口是通过 `server251` 的 SSH 反向隧道映射过来的。**隧道未开启时，训练与 `--live` 测试均会报 `APIConnectionError`**。
2. **中转站 `https://sub.ddpapi.top/v1`**：
   可从有外网的机器直接访问，但需要用户提供有效的 `sub.ddpapi.top` API Key（31 服务器 `~/.edge_reflector.env` 中保存的 Key 是 `server251` 网关 Token，直接请求 `sub.ddpapi.top` 会返回 `401 INVALID_API_KEY`）。

---

## 四、接手后推荐执行顺序

1. **确认隧道或中转 Key 就绪后，先跑在线闭环验收**：
   ```bash
   source ~/.edge_reflector.env
   python tools/test_t75_reflector_closed_loop.py --live
   ```
   重点抽检提取出的 `task` 级（`depth=1`）与 `step` 级（`depth>=2`）经验质量；如 `task`/`step` 分类不准，可临时开启 `ENABLE_EXPERIENCE_TYPE_VALIDATION=1` 对比。
2. **启动多卡训练**（按实际机器选择 `GPU_PROFILE`）：
   ```bash
   GPU_PROFILE=2xa40 CUDA_VISIBLE_DEVICES=2,3 bash examples/grpo_trainer/run_t75.sh
   # 或 4x3090 / 8x3090 / auto
   ```
