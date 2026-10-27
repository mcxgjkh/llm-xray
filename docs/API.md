# API 参考

本页描述 v1.1.0 的本地 HTTP / WebSocket 接口。服务只监听 `127.0.0.1:8765`，供同一台机器上的
桌面外壳或开发服务器调用，不面向公网。

契约来源：`docs/SPEC.md`（v1.0.0 契约）、`apps/desktop/src/types.ts`、`backend/xray/app.py` 与
`backend/xray/native_bridge.py`。若本页与运行中的服务不一致，以服务端实际返回为准，并在发布前
同步本页。v1.1.0 新增：`mode=native`、原生精确通道接口与状态、`done.capabilities/exact`、
`status(state=no_logprobs)`。

约定：

- 所有请求与响应均为 UTF-8 JSON；文档原文接口返回 `text/markdown`；导出接口返回
  `application/x-ndjson`。
- 时间戳：`started_at`、`created_at` 使用 Unix 秒（浮点），`at` 使用生成开始后的秒数。
- 概率字段使用自然对数 log-prob；`prob` 为同一 top-k 内重新归一化后的概率。
- 未知或不适用字段返回 `null`，不用 `0` 表示“未知”。
- REST 的错误使用 HTTP 状态码与 `{"detail": "..."}`；生成过程中的错误优先通过 WebSocket
  `error` 事件送达。

## 1. REST

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| GET | `/api/health` | 版本、运行时长、LM Studio 可达性、透镜状态、GPU、指标快照、会话数 |
| GET | `/api/config` | 默认采样、模型、能力、透镜状态 |
| GET | `/api/models?refresh=false` | LM Studio 模型列表（归一化）；`refresh=true` 强制刷新 |
| GET | `/api/sessions?limit=20` | 最近会话摘要列表 |
| GET | `/api/sessions/{id}` | 会话元信息、消息与步骤计数 |
| GET | `/api/sessions/{id}/steps?offset=0&limit=200` | 精简步骤列表 |
| GET | `/api/sessions/{id}/steps/{index}` | 单步完整快照（含 probe） |
| GET | `/api/sessions/{id}/export` | 下载 JSONL，一行一个步骤 |
| POST | `/api/sessions/{id}/stop` | 尽力停止当前生成 |
| POST | `/api/rag/search` | 混合检索，body：`{query, top_k}` |
| GET | `/api/rag/documents` | 记忆库清单与索引统计 |
| POST | `/api/rag/reindex` | 重建检索索引 |
| GET | `/api/lens/status` | 透镜状态、模型、设备、校准结果 |
| POST | `/api/lens/load` | body：`{model?, device?}`，后台加载，不自动下载 |
| POST | `/api/lens/unload` | 释放透镜模型与显存 |
| GET | `/api/native/status` | 原生精确通道状态、模型、设备、精确标记 |
| POST | `/api/native/load` | body：`{model?, device?, load_in_4bit?, device_map?}`，后台加载 |
| POST | `/api/native/unload` | 卸载原生模型并释放内存 |
| GET | `/api/docs` | `docs/*.md` 文件清单 |
| GET | `/api/docs/{name}` | 返回 `docs/*.md` 原文，`name` 必须带 `.md` 且只允许白名单文件名 |

### 1.1 GET /api/health

```json
{
  "status": "ok",
  "version": "1.1.0",
  "uptime_s": 362.4,
  "lmstudio": {
    "reachable": true,
    "error": null,
    "models": [],
    "loaded": ["deepseek-r1-distill-llama-8b"]
  },
  "lens": {
    "status": "unloaded",
    "model": "Qwen/Qwen2.5-0.5B-Instruct",
    "device": null,
    "surrogate": true,
    "same_weights": false,
    "calibrated_layers": 0,
    "message": "透镜模型未加载；可在设置中显式加载",
    "threads": 2
  },
  "native": {
    "available": true,
    "status": "unloaded",
    "model": "Qwen/Qwen2.5-0.5B-Instruct",
    "device": null,
    "surrogate": false,
    "exact": false,
    "same_weights": false,
    "calibrated_layers": 0,
    "generation": false,
    "message": "原生精确通道未加载；需要本地 HF 因果模型目录（不会自动下载）",
    "threads": 2
  },
  "gpu": { "available": true, "util": 72, "vram_used": 4.8, "vram_total": 8.0 },
  "metrics": {
    "cpu": 18.2, "ram": 6.1, "gpu": 72, "vram_used": 4.8, "vram_total": 8.0,
    "tps": 12.4, "ttft_ms": 420, "tokens": 128, "context": 512, "lens_lag_ms": 0
  },
  "sessions": 3,
  "demo_available": true
}
```

`/api/health` 可能触发 LM Studio 模型列表刷新与透镜/原生状态读取，但不会加载任何模型。

### 1.2 GET /api/config

```json
{
  "version": "1.1.0",
  "lmstudio_url": "http://127.0.0.1:1234",
  "default_model": "deepseek-r1-distill-llama-8b",
  "default_mode": "lmstudio",
  "lens": {
    "status": "unloaded",
    "model": "Qwen/Qwen2.5-0.5B-Instruct",
    "device": null,
    "surrogate": true,
    "same_weights": false,
    "calibrated_layers": 0,
    "threads": 2
  },
  "native": {
    "available": true,
    "status": "unloaded",
    "model": "Qwen/Qwen2.5-0.5B-Instruct",
    "device": null,
    "surrogate": false,
    "exact": false,
    "same_weights": false,
    "calibrated_layers": 0,
    "generation": false,
    "threads": 2,
    "message": "原生精确通道未加载"
  },
  "capabilities": { "logprobs": true, "attention": true, "lens": true, "rag": true, "gpu": true },
  "exact_available": true,
  "sampling_defaults": {
    "temperature": 0.7, "top_p": 0.95, "top_k": 40,
    "repetition_penalty": 1.05, "max_tokens": 4096, "seed": null
  },
  "demo_available": true
}
```

`capabilities.attention` 在替代透镜或原生精确通道可用时为 true；`capabilities.lens` 取决于透镜
模块是否可用；`exact_available` 表示原生精确通道的依赖与 bridge 是否可用。它们都不代表模型
已加载。

### 1.3 GET /api/models

返回 LM Studio 模型列表的归一化结果。读取列表**不会**加载模型。

```json
{
  "reachable": true,
  "models": [
    {
      "id": "deepseek-r1-distill-llama-8b",
      "object": "model",
      "type": "llm",
      "publisher": "deepseek",
      "arch": "llama",
      "quantization": "Q4_K_M",
      "state": "loaded",
      "max_context_length": 32768
    }
  ],
  "error": null
}
```

### 1.4 GET /api/sessions

返回会话摘要数组：

```json
[
  {
    "id": "20261021-231500-7f3a",
    "model": "deepseek-r1-distill-llama-8b",
    "mode": "lmstudio",
    "created_at": 1792586100.1,
    "tokens": 128,
    "preview": "请解释一下 Tuned Lens 的校准过程。",
    "has_lens": true,
    "path": "data/sessions/20261021-231500-7f3a.jsonl"
  }
]
```

### 1.5 GET /api/sessions/{id}

```json
{
  "id": "20261021-231500-7f3a",
  "mode": "lmstudio",
  "model": "deepseek-r1-distill-llama-8b",
  "started_at": 1792586100.1,
  "capabilities": { "logprobs": true, "attention": true, "lens": true,
                    "exact": false, "surrogate": true, "runtime": "lmstudio" },
  "sampling": { "temperature": 0.7, "top_p": 0.95, "top_k": 40,
                "repetition_penalty": 1.05, "max_tokens": 4096, "seed": null },
  "rag_enabled": false,
  "lens_enabled": true,
  "messages": [{ "role": "user", "content": "你好" }],
  "step_count": 128,
  "has_probe": true
}
```

会话不存在返回 404。

### 1.6 GET /api/sessions/{id}/steps

`offset` 默认 0，`limit` 默认 200（上限由服务端限制，界面使用 500）。精简步骤不含完整 probe：

```json
{
  "session_id": "20261021-231500-7f3a",
  "offset": 0,
  "limit": 200,
  "total": 128,
  "steps": [
    {
      "index": 0,
      "delta": "你",
      "kind": "answer",
      "chosen_prob": 0.82,
      "entropy": 0.7,
      "at": 0.42,
      "has_probe": true
    }
  ]
}
```

### 1.7 GET /api/sessions/{id}/steps/{index}

直接返回 `ReplayStep`（含 probe 快照）。会话或步骤不存在返回 404：

```json
{
  "index": 12,
  "delta": "紫",
  "kind": "answer",
  "topk": [
    { "token": "紫", "id": 123, "logprob": -0.21, "prob": 0.81, "rank": 1 },
    { "token": "蓝", "id": 456, "logprob": -1.84, "prob": 0.16, "rank": 2 }
  ],
  "chosen_prob": 0.81,
  "entropy": 0.62,
  "at": 3.5,
  "probe": { "step": 12, "model": "Qwen/Qwen2.5-0.5B-Instruct", "surrogate": true }
}
```

`probe.surrogate` 为 `true` 表示替代透镜（LM Studio 模式）；原生精确会话中为 `false`，且
`same_weights` 为 `true`，表示与生成共用同一次前向。

### 1.8 GET /api/sessions/{id}/export

返回 `Content-Type: application/x-ndjson; charset=utf-8` 的文件下载，文件名
`<session-id>.jsonl`，一行一个 `ReplayStep`（含 probe）。会话不存在或没有可导出步骤时返回 404。
示例见第 4 节。

前端导出行为（不属于本接口）：桌面版通过 pywebview 原生保存对话框写文件，浏览器版回退 Blob
下载；失败或取消不会写文件，界面会显示提示。接口本身始终返回完整文件内容。

### 1.9 POST /api/sessions/{id}/stop

请求体为空。返回：

```json
{ "stopped": true, "session_id": "20261021-231500-7f3a" }
```

没有进行中的生成时返回：

```json
{ "stopped": false, "session_id": "20261021-231500-7f3a", "reason": "no_active_generation" }
```

停止是尽力而为：已经由 LM Studio 采样并送达的 Token 无法撤回，会话数据保留。

### 1.10 RAG

`POST /api/rag/search`：

```json
{ "query": "Tuned Lens 的校准", "top_k": 3 }
```

响应：

```json
{
  "query": "Tuned Lens 的校准",
  "backend": "hybrid",
  "items": [
    {
      "id": "memory/interp.md#2",
      "text": "Tuned Lens 为每层训练一个仿射 translator……",
      "source": "memory/interp.md",
      "heading": "校准",
      "cosine": 0.78,
      "bm25": 1.32,
      "hybrid": 0.65,
      "hits": ["Tuned", "校准"]
    }
  ]
}
```

`cosine` 在纯 BM25 模式下为 `null`；`backend` 为 `hybrid` 或 `bm25`。`query` 为空返回 400。

`GET /api/rag/documents`：

```json
{
  "documents": [
    {
      "id": "USAGE",
      "source": "USAGE.md",
      "chunks": 42,
      "chars": 18000,
      "headings": ["使用说明", "快速开始"]
    }
  ],
  "stats": {
    "documents": 1, "chunks": 42, "chars": 18000, "backend": "hybrid",
    "embedding_model": "text-embedding-nomic-embed-text-v1.5",
    "chunk_chars": 320, "overlap": 48, "built_at": 1792586100123.4
  },
  "memory_dir": "D:\\path\\to\\llm-xray\\memory"
}
```

`documents[].source` 是相对于 `memory/` 的文件名。embedding 不可用时 `backend` 为 `bm25`，
`embedding_model` 为 `null`。

`POST /api/rag/reindex` 重建索引并返回统计：

```json
{
  "ok": true,
  "backend": "hybrid",
  "stats": { "documents": 1, "chunks": 42, "chars": 18000, "backend": "hybrid" }
}
```

### 1.11 透镜

`GET /api/lens/status` 返回 `LensStatus`，与 `hello` 事件中的 `lens` 字段相同。

`POST /api/lens/load`：

```json
{ "model": "Qwen/Qwen2.5-0.5B-Instruct", "device": "cpu" }
```

返回当前状态（加载在后台进行，随后通过 `status` 事件或轮询 `GET /api/lens/status` 获取进度）：

```json
{
  "status": "loading",
  "model": "Qwen/Qwen2.5-0.5B-Instruct",
  "device": "cpu",
  "surrogate": true,
  "same_weights": false,
  "calibrated_layers": 0,
  "message": "正在加载透镜模型（低优先级 CPU 线程）",
  "threads": 2
}
```

`POST /api/lens/unload` 返回卸载后的 `LensStatus`（`status` 为 `unloaded`）。

`status` 取值：`unloaded`、`loading`、`ready`、`error`、`disabled`。`disabled` 表示透镜模块
不可用（未安装可选依赖）；此时应用仍可使用 LM Studio 通道与原生精确通道（若可用）。

### 1.12 原生精确通道

`GET /api/native/status` 返回 `NativeStatus`：

```json
{
  "available": true,
  "status": "ready",
  "model": "Qwen/Qwen2.5-0.5B-Instruct",
  "device": "cpu",
  "surrogate": false,
  "exact": true,
  "same_weights": true,
  "calibrated_layers": 0,
  "generation": true,
  "threads": 2,
  "message": "原生精确通道就绪：生成与观测共用同一次前向（surrogate=false）"
}
```

`status` 取值：`unloaded`、`loading`、`ready`、`error`、`disabled`。`exact=true` 需要同时满足
`status=ready` 且 `generation=true`（即桥接支持逐 Token 生成）；`surrogate` 恒为 `false`。
`available=false` 表示缺少 torch/transformers 或 native_bridge。

`POST /api/native/load`：

```json
{ "model": "Qwen/Qwen2.5-0.5B-Instruct", "device": "cpu", "load_in_4bit": false }
```

加载在后台进行，接口返回当时的 `NativeStatus`（通常为 `loading`），随后通过
`status(scope=native)` 事件或轮询写入进度。模型必须已在本地（`models/lens/`、HF 缓存或绝对
路径）；后端以 `local_files_only` 加载，不会下载。当前 `load_in_4bit=true` 会被拒绝（需要
CUDA torch + bitsandbytes，且原生通道固定 CPU float32）。生成进行中调用 load/unload 会被拒绝，
返回 `ok=false` 与说明。

`POST /api/native/unload` 卸载模型并释放内存，返回卸载后的 `NativeStatus`。

### 1.13 文档

`GET /api/docs`：

```json
{
  "documents": [
    { "name": "API.md", "size": 12873 },
    { "name": "USAGE.md", "size": 20114 }
  ]
}
```

`GET /api/docs/{name}` 返回 `text/markdown; charset=utf-8`。`name` 必须以 `.md` 结尾，不允许
路径分隔符或 `..`，只允许 `docs/` 下的文件；否则返回 404。

## 2. WebSocket

连接：`ws://127.0.0.1:8765/ws`（开发时前端通过 Vite 代理连接同一路径）。连接建立后服务端立即
发送 `hello`。

### 2.1 客户端事件

```json
{ "type": "ping" }
```

```json
{
  "type": "start",
  "payload": {
    "messages": [{ "role": "user", "content": "你好" }],
    "model": "deepseek-r1-distill-llama-8b",
    "mode": "lmstudio",
    "sampling": {
      "temperature": 0.7, "top_p": 0.95, "top_k": 40,
      "repetition_penalty": 1.05, "max_tokens": 4096, "seed": null
    },
    "rag": { "enabled": true, "top_k": 3, "inject": false },
    "lens": { "enabled": true, "model": null },
    "system": "你是本地观测助手。"
  }
}
```

原生精确模式（不经过 LM Studio，`lens` 会被忽略；probe 来自同一次前向）：

```json
{
  "type": "start",
  "payload": {
    "messages": [{ "role": "user", "content": "你好" }],
    "mode": "native",
    "model": "Qwen/Qwen2.5-0.5B-Instruct",
    "sampling": { "temperature": 0.7, "top_p": 0.95, "top_k": 40,
                  "repetition_penalty": 1.05, "max_tokens": 4096, "seed": null }
  }
}
```

```json
{ "type": "stop" }
```

`mode` 取值 `lmstudio`（默认）、`native` 或 `demo`。`sampling` 可只传部分字段，其余用默认值
（v1.1.0 默认 `max_tokens=4096`）。`rag.inject` 为 `false` 时只展示召回结果，不写入 system
prompt。`native` 模式同一时间只允许一个生成会话，冲突返回 `BUSY`；原生模型未加载时后端会自动
尝试加载（本地权重），失败返回 `NATIVE_UNAVAILABLE` 或 `NATIVE_RUNTIME_ERROR`。

### 2.2 服务端事件

| type | 频率 | 说明 |
| --- | --- | --- |
| `hello` | 连接时一次 | 版本、配置、LM Studio 可达性、透镜与原生状态、demo 文本 |
| `started` | 每次生成一次 | 会话元信息与能力（含 `exact` / `runtime`） |
| `rag` | 每次生成一次（启用时） | 召回结果与是否注入 |
| `token` | 每个 Token | 文本增量与 top-k 分布 |
| `probe` | 节流后 | 观测快照；`surrogate=true` 为替代透镜，`false` 为原生精确 |
| `projection` | 节流后 | UMAP/PCA 投影点 |
| `metrics` | 约 1.5 秒 | CPU/内存/GPU/VRAM/tok/s |
| `kv` | probe 之后 | KV 缓存估算 |
| `done` | 生成结束 | 用量、结束原因、耗时、能力标记 |
| `status` | 状态变化 | 透镜/原生加载、生成阶段、`no_logprobs` 等 |
| `error` | 出错时 | 错误码与消息 |
| `pong` | 心跳响应 | 返回服务端时间 |

`hello`：

```json
{
  "type": "hello",
  "version": "1.1.0",
  "config": { "version": "1.1.0", "lmstudio_url": "http://127.0.0.1:1234",
              "default_model": "deepseek-r1-distill-llama-8b", "default_mode": "lmstudio",
              "lens": { "status": "unloaded", "model": null, "device": null,
                        "surrogate": true, "same_weights": false, "calibrated_layers": 0 },
              "native": { "available": true, "status": "unloaded", "model": null,
                          "device": null, "surrogate": false, "exact": false,
                          "same_weights": false, "generation": false },
              "capabilities": { "logprobs": true, "attention": true, "lens": true,
                                "rag": true, "gpu": true },
              "exact_available": true,
              "sampling_defaults": { "temperature": 0.7, "top_p": 0.95, "top_k": 40,
                                     "repetition_penalty": 1.05, "max_tokens": 4096, "seed": null },
              "demo_available": true },
  "lmstudio": { "reachable": true, "models": [], "error": null },
  "lens": { "status": "unloaded", "model": null, "device": null, "surrogate": true,
            "same_weights": false, "calibrated_layers": 0 },
  "native": { "available": true, "status": "unloaded", "model": null, "device": null,
              "surrogate": false, "exact": false, "same_weights": false, "generation": false },
  "demo_text": "演示数据说明"
}
```

`started`：

```json
{ "type": "started", "session": { "id": "20261021-231500-7f3a", "mode": "lmstudio",
  "model": "deepseek-r1-distill-llama-8b", "started_at": 1792586100.1,
  "capabilities": { "logprobs": true, "attention": true, "lens": true,
                    "exact": false, "surrogate": true, "runtime": "lmstudio" },
  "sampling": { "temperature": 0.7, "top_p": 0.95, "top_k": 40,
                "repetition_penalty": 1.05, "max_tokens": 4096, "seed": null },
  "rag_enabled": false, "lens_enabled": true } }
```

`mode=native` 时 `capabilities` 为：

```json
{ "logprobs": true, "attention": true, "lens": true,
  "exact": true, "surrogate": false, "runtime": "native" }
```

`rag`：

```json
{ "type": "rag", "items": [], "query": "你好", "injected": false }
```

`token`：

```json
{
  "type": "token",
  "step": {
    "index": 0,
    "delta": "你",
    "kind": "answer",
    "topk": [
      { "token": "你", "id": 123, "logprob": -0.21, "prob": 0.81, "rank": 1 },
      { "token": "您", "id": 456, "logprob": -1.90, "prob": 0.15, "rank": 2 }
    ],
    "chosen_logprob": -0.21,
    "chosen_prob": 0.81,
    "entropy": 0.62,
    "at": 0.42,
    "ttft_ms": 420,
    "tps": 12.4,
    "generated": 1,
    "text_len": 1
  }
}
```

`kind` 为 `answer` 或 `reasoning`；reasoning 是 DeepSeek-R1 的思考内容，单独显示并默认折叠。
原生精确模式目前只产出 `answer`；原生模型的 chat template 若缺失会回退为 `role: content` 文本。

`probe`（LM Studio + 替代透镜；原生精确模式下 `surrogate=false`、`same_weights=true`，
且 probe 与 token 来自同一次前向）：

```json
{
  "type": "probe",
  "step": 12,
  "probe": {
    "step": 12,
    "model": "Qwen/Qwen2.5-0.5B-Instruct",
    "surrogate": true,
    "same_weights": false,
    "latency_ms": 38.2,
    "context": { "tokens": ["<|im_start|>", "user", "你好"], "kinds": ["prompt", "prompt", "prompt"] },
    "attention": {
      "layers": [12, 23],
      "selected": {
        "12": { "layer": 12, "mean": [0.1, 0.2, 0.7],
                "heads": [[0.0, 0.1, 0.9]], "head_types": ["previous-token"] },
        "23": { "layer": 23, "mean": [0.05, 0.15, 0.8],
                "heads": [[0.0, 0.0, 1.0]], "head_types": ["local"] }
      },
      "head_count": 14
    },
    "lens": [
      { "layer": 12, "topk": [{ "token": "你", "id": 123, "logprob": -0.2, "prob": 0.8, "rank": 1 }],
        "tuned": true, "agreement": 0.61 }
    ],
    "hidden": [0.12, -0.03, 0.07],
    "residual": [
      { "layer": 12, "attn_norm": 0.8, "mlp_norm": 0.2, "attn_logit": 0.3, "mlp_logit": -0.1 }
    ],
    "features": [
      { "id": 42, "label": "紫色 / violet", "activation": 0.87, "tokens": ["紫", "violet"] }
    ],
    "kv": { "used": 312, "capacity": 8192, "bytes": 40894464, "bytes_per_token": 131072 },
    "vector": { "x": 0.2, "y": -0.8, "z": 0.4 }
  }
}
```

`projection`：

```json
{
  "type": "projection",
  "projection": {
    "points": [
      { "id": "t12", "token": "紫", "index": 12, "kind": "generated",
        "x": 0.1, "y": -0.3, "z": 0.5, "step": 12 }
    ],
    "reducer": "UMAP",
    "fitted_at": 12
  }
}
```

`metrics` 的 `metrics` 字段与 `/api/health` 的 `metrics` 相同。`kv`：

```json
{ "type": "kv", "kv": { "used": 312, "capacity": 8192, "bytes": 40894464,
  "bytes_per_token": 131072, "layers": 32, "kv_heads": 8, "head_dim": 128,
  "dtype": "float16", "estimated": true } }
```

`done`：

```json
{
  "type": "done",
  "usage": { "prompt_tokens": 100, "completion_tokens": 128, "total_tokens": 228 },
  "finish_reason": "stop",
  "elapsed_s": 11.4,
  "capabilities": { "logprobs": true, "attention": true, "lens": true,
                    "exact": false, "surrogate": true, "runtime": "lmstudio" },
  "exact": false,
  "surrogate": true
}
```

`capabilities`、`exact`、`surrogate` 是 v1.1.0 新增的可选字段。原生会话为 `exact=true`；LM Studio
流未返回 top-k 的会话为 `capabilities.logprobs=false`；demo 会话不保证这些字段存在。

`status`：

```json
{ "type": "status", "scope": "lens", "state": "loading", "message": "正在加载透镜模型" }
```

```json
{ "type": "status", "scope": "native", "state": "prefill",
  "message": "正在编码提示词并建立 KV cache…" }
```

```json
{ "type": "status", "scope": "runtime", "state": "no_logprobs",
  "message": "LM Studio 未返回 top_logprobs；概率云为空，可切换原生精确模式" }
```

原生 `state` 取值：`loading`、`prefill`、`generating`、`ready`、`error`。`no_logprobs` 不是错误，
不会中断连接。

`error`：

```json
{ "type": "error", "code": "LMSTUDIO_UNREACHABLE",
  "message": "无法连接 LM Studio（http://127.0.0.1:1234）。请确认 LM Studio 本地服务已启动。",
  "recoverable": true }
```

## 3. Probe 字段表

`probe` 是观测载荷。LM Studio 模式下它来自替代透镜（`surrogate=true`）；原生精确模式下它来自与
生成同一次前向（`surrogate=false`、`same_weights=true`）。

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `step` | int | 对应的 Token 步号 |
| `model` | string | 观测模型的 HF 名称或本地路径 |
| `surrogate` | bool | true 为替代透镜；false 为原生精确通道（同一模型） |
| `same_weights` | bool | 是否与生成模型同名同权重；原生精确模式为 true |
| `latency_ms` | number | 该步观测前向耗时（原生模式包含在同一次前向中） |
| `context.tokens` | string[] | 观测模型的 tokenizer 切分结果，可能与 LM Studio 不同 |
| `context.kinds` | string[] | `prompt` 或 `generated` |
| `attention.layers` | int[] | 已捕获逐头数据的层号 |
| `attention.selected` | object | 层号到注意力行数据的映射 |
| `attention.selected[layer].mean` | number[] | 该层平均注意力行 |
| `attention.selected[layer].heads` | number[][] | 每个头的注意力行（仅焦点层） |
| `attention.selected[layer].head_types` | string[] | 启发式头分类 |
| `attention.head_count` | int | 头数 |
| `lens[]` | array | 代表层的 Tuned/Logit Lens 的 top-k（均匀分布并包含最后一层） |
| `lens[].layer` | int | 层号 |
| `lens[].topk` | TokenOption[] | 该层解码候选 |
| `lens[].tuned` | bool | 是否使用 Tuned Lens（false 为 Logit Lens） |
| `lens[].agreement` | number \| null | 该层 top-1 与校准目标在最终输出空间 top-1 的一致率 |
| `hidden` | number[] | 最终层隐状态的 16 维降采样，用于向量条带 |
| `residual[]` | array | 每层 attention/MLP 分支的范数与 logit 贡献 |
| `features[]` | array | 稀疏特征探针结果（`probe`，非 SAE） |
| `kv` | object \| null | KV 估算：`used`、`capacity`、`bytes`、`bytes_per_token` |
| `vector` | object \| null | 该 Token 的 3D 投影坐标 |

`TokenOption`：

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `token` | string | 词表 token（可能带空格或控制符） |
| `id` | int \| null | token id |
| `logprob` | number | 原始 log-prob |
| `prob` | number | 重新归一化后的概率 |
| `rank` | int | 从 1 开始的排名 |

`KVInfo`（`kv` 事件与 `probe.kv` 的扩展形态）额外包含 `layers`、`kv_heads`、`head_dim`、
`dtype`、`estimated`。`estimated: true` 表示数值按模型配置推算，不等同于实际显存占用。

## 4. JSONL 导出示例

每行一个 `ReplayStep`：

```jsonl
{"index":0,"delta":"你","kind":"answer","topk":[{"token":"你","id":123,"logprob":-0.21,"prob":0.81,"rank":1}],"chosen_prob":0.81,"entropy":0.62,"at":0.42,"probe":null}
{"index":1,"delta":"好","kind":"answer","topk":[{"token":"好","id":789,"logprob":-0.31,"prob":0.73,"rank":1}],"chosen_prob":0.73,"entropy":0.55,"at":0.51,"probe":{"step":1,"model":"Qwen/Qwen2.5-0.5B-Instruct","surrogate":true}}
```

导出包含 reasoning 步骤（`kind: "reasoning"`），也包含 `probe: null` 的步骤（未采样到 probe 的
Token）。原生精确会话的 probe 为 `surrogate=false`、`same_weights=true`；LM Studio 无 top-k 的
会话中 `topk` 为空数组，会话能力为 `logprobs=false`。分析时请注意区分。

## 5. 错误码

WebSocket `error` 事件的稳定字符串：

| 错误码 | 典型场景 | 处理建议 |
| --- | --- | --- |
| `LMSTUDIO_UNREACHABLE` | 连接 LM Studio 超时/拒绝 | 启动 LM Studio 本地服务器，确认端口 |
| `LMSTUDIO_HTTP_ERROR` | LM Studio 返回非降级类 HTTP 错误 | 查看 LM Studio 日志与模型状态 |
| `LMSTUDIO_STREAM_ERROR` | SSE 流中返回错误或流中断 | 重试；检查模型是否仍在加载 |
| `MODEL_NOT_LOADED` | 请求的模型未加载 | 在 LM Studio 中加载模型后重试 |
| `BUSY` | 当前连接已有生成，或原生精确通道正被其他会话使用 | 先停止或等待结束 |
| `NATIVE_UNAVAILABLE` | 原生通道不可用：缺少 torch/transformers 或 native_bridge | 安装 `backend/requirements-lens.txt`，确认代码完整 |
| `NATIVE_RUNTIME_ERROR` | 原生模型加载、prefill 或逐 Token 生成失败 | 检查本地模型路径/内存；CPU 0.5B 仍需数 GB 内存 |
| `INVALID_PAYLOAD` | `start` 载荷缺失或类型错误 | 按本页字段表修正 |
| `UNKNOWN_MESSAGE` | 未知消息类型 | 检查 `type` 字段 |
| `INTERNAL` | 未预期异常 | 查看后端日志 `data/server.log` |

REST 错误使用 HTTP 状态码与 `detail`：成功 200，参数错误 400，不存在 404，服务端错误 500。
生成过程中的错误优先通过 WebSocket `error` 事件送达，而不是中断连接。
