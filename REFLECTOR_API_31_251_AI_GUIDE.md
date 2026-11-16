# Reflector API 中转（31 ↔ 251）：给接手 AI 的使用说明

更新：2026-11-16（北京时间）。**这是实际部署说明，不包含任何密钥明文。**

## 1. 当前状态（已实际验证）

- 从服务器 **31** 通过本机 `127.0.0.1:29663` 调用 Reflector：`GET /healthz` 返回 `{"status":"ok"}`，`GET /v1/models` 返回 HTTP 200，`POST /v1/chat/completions` 使用 `gpt-6.1-sol` 返回 HTTP 200。
- 251 的上游 Key **共有 3 把**，网关按顺序自动故障切换（不能把三把上游 Key 当作三个“模型”）。
- 客户端 `ExperienceUpdater` 默认超时已设为 `330s`（总截止时间 `340s`，可通过 `EXPERIENCE_UPDATE_TIMEOUT_SECONDS` / `EXPERIENCE_UPDATE_DEADLINE_SECONDS` 调整），充分覆盖 251 网关最坏 `3 × 95s = 285s` 的三 Key 轮询窗口。

## 2. 网络结构与地址

- 训练/客户端：**31 服务器** `172.18.166.31`。
- API 中转网关：**251 服务器** `172.18.167.251`，服务只绑定其 `127.0.0.1:29663`。
- 251 通过 SSH **反向端口转发** `-R 127.0.0.1:29663:127.0.0.1:29663`，把服务提供到 31 的回环端口。
- 31 实际使用的 OpenAI 兼容 `base_url`：**`http://127.0.0.1:29663/v1`**。
- 目标模型：**`gpt-6.1-sol`**；接口：`/v1/models` 和 `/v1/chat/completions`。**未提供 `/v1/responses` 接口。**
- 上游地址配置在 251 的 `/home/zehao/.edge_api_gateway.json`，当前为 `https://sub.ddpapi.top/v1`。客户端不要直接访问 251 的内网 IP/端口。

## 3. 密钥如何区分（最容易搞错）

| 用途 | 保存位置 | 数量 |
| --- | --- | --- |
| 251→上游服务商的真实 API Keys（故障切换池） | 251：`/home/zehao/.edge_reflector_api_keys.json` | **3** |
| 31→251 中转网关的客户端鉴权 Key | 31：`/home/zehao/.edge_reflector.env` 中的 `EXPERIENCE_UPDATE_API_KEY` | **1** |
| 251 网关接受的客户端鉴权白名单 | 251：`/home/zehao/.edge_reflector_client_keys.json` | **1** |

**绝对不要**把客户端 Key 加进“三个上游 Key”池；不要把三个上游 Key 用作网关客户端密钥。以上密钥文件和日志都不要提交 GitHub，也不要在聊天、日志、命令输出中打印 Key。

## 4. 在 31 上如何直接调用与闭环验收

31 已配置（路径：`/home/zehao/.edge_reflector.env`）：

```bash
export EXPERIENCE_UPDATE_BASE_URL=http://127.0.0.1:29663/v1
export EXPERIENCE_UPDATE_MODEL=gpt-6.1-sol
# EXPERIENCE_UPDATE_API_KEY 已保存在上述文件，不要在文档中粘贴
```

在 **31** 上运行完整 T75 Reflector 离线 + 在线闭环测试：

```bash
source /home/zehao/.edge_reflector.env
/home/zehao/.conda/envs/tree-edge/bin/python tools/test_t75_reflector_closed_loop.py --live
```

## 5. 三 Key 切换规则

执行代码位于 251：`/home/zehao/echo-collab/T073/reflector_gateway.py`

- 按 **Key 1→Key 2→Key 3** 尝试。请求成功立即返回。
- 上游出现 `401/403/404/408/409/425/429` 或 `5xx`，或网络失败，会尝试下一把 Key。一般请求错误（如参数无效 `400`）直接返回，不盲目换 Key。
- 每个 Key 的超时为 **95 秒**；三个 Key 最坏可能累计约 **285 秒**，客户端超时设为 `330s`（deadline `340s`）。

## 6. 快速健康检查与故障定位

在 **31** 上（无需打印密钥）：

```bash
curl -sS --max-time 5 http://127.0.0.1:29663/healthz
```

正常应返回 `{"status":"ok"}`。

在 **251** 上检查网关进程/监听/日志：

```bash
pgrep -af '^python3 -u /home/zehao/echo-collab/T073/reflector_gateway.py$'
ss -lnt | grep ':29663'
tail -n 40 /home/zehao/echo-collab/T073/reflector-gateway.log
```

251→31 的反向隧道重建命令（**仅在确认旧隧道已断后使用**）：

```bash
ssh -fNT -o BatchMode=yes -o ExitOnForwardFailure=yes \
  -o ServerAliveInterval=30 -o ServerAliveCountMax=3 \
  -R 127.0.0.1:29663:127.0.0.1:29663 \
  zehao@172.18.166.31
```
