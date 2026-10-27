# LLM-Xray 内部规格与协作契约

本文档是 v1.0.0 的实现契约。所有模块围绕同一个事件模型协作。

## 1. 现实约束

本机通过 LM Studio 0.4.x 提供本地推理，模型为 `deepseek-r1-distill-llama-8b`。
LM Studio 的 HTTP API 只暴露文本、采样参数、统计信息和可选的 `top_logprobs`，**不暴露隐藏状态、
逐层注意力或 MLP 输出**。因此项目采用双通道架构：

1. **实时通道（真实）**：LM Studio `/v1/responses`（优先）或 OpenAI 兼容接口，采集真实 Token、
   top-logprobs、TTFT、tok/s、结束原因、用量。用于概率云、解码树、采样重算、性能仪表盘、
   对话、时间回放、导出。
2. **透镜通道（替代模型，明确标注）**：一个独立、低优先级的本地小模型（默认
   `Qwen/Qwen2.5-0.5B-Instruct`）对同一段文本做前向观测，提取隐藏状态、注意力、调优透镜、
   残差贡献、稀疏特征、UMAP 投影。通道默认不启动，用户显式开启；即使开启，也不阻塞 LM Studio
   Token 流。

文档、界面徽标和事件字段都必须明确区分 `runtime=lmstudio` 与 `lens=surrogate`。
如果用户把透镜模型指向与 LM Studio 相同的 HF 权重，界面自动标记为 `same-weights`；这是获得精确
内部张量的唯一路径（需要额外显存/内存）。

## 2. 技术栈

- 桌面外壳：pywebview 6 + Edge WebView2。无捆绑 Chromium，窗口常驻内存显著低于 Electron。
- 前端：React 19 + TypeScript + Vite + Three.js。Canvas/SVG 自绘图表，不引入图表库。
- 后端：Python 3.11 + FastAPI + Uvicorn + httpx + NumPy + scikit-learn + UMAP（可选）。
- 透镜：PyTorch + Transformers（可选依赖，启用时才加载）。
- 检索：混合检索（字符 n-gram BM25 + LM Studio embedding 余弦），无外部向量数据库。
- 版本：Git + SemVer + `VERSION` + `CHANGELOG.md` + `docs/VERSIONS.md`。

## 3. 目录

```
backend/xray/            FastAPI、LM Studio 适配、会话、采样、RAG、回放、指标
backend/xray/lens/       替代透镜模型、Tuned Lens、注意力、残差、稀疏特征、UMAP
apps/desktop/            React 界面
desktop/                 pywebview 启动器
docs/                    中文技术文档与静态文档页
scripts/                 开发、构建、发布、模型准备脚本
memory/                  本地记忆库（Markdown）
data/                    运行期缓存（git 忽略）
```

## 4. 事件协议

传输：`WS /ws`，JSON 文本帧。

客户端 -> 服务端：

```jsonc
{"type":"start","payload":{
  "messages":[{"role":"user","content":"..."}],
  "model":"deepseek-r1-distill-llama-8b",
  "mode":"lmstudio",              // lmstudio | demo
  "sampling":{"temperature":0.7,"top_p":0.95,"top_k":40,"repetition_penalty":1.05,"max_tokens":512,"seed":null},
  "rag":{"enabled":true,"top_k":3},
  "lens":{"enabled":true,"model":null},
  "system":"..."
}}
{"type":"stop"}
{"type":"ping"}
```

服务端 -> 客户端：

```jsonc
{"type":"hello","version":"1.0.0","config":{...},"lmstudio":{"reachable":true,"models":[...]},"lens":{"status":"unloaded",...}}
{"type":"started","session":{"id":"...","mode":"lmstudio","model":"...","started_at":0,"capabilities":{"logprobs":true,"attention":true,"lens":true},"sampling":{...}}}
{"type":"rag","items":[{...}]}
{"type":"token","step":{"index":0,"delta":"你","kind":"answer","topk":[...],"chosen_logprob":-0.2,"chosen_prob":0.82,"entropy":0.7,"at":0,"ttft_ms":420,"tps":12.4,"generated":1,"text_len":1}}
{"type":"probe","step":0,"probe":{...}}
{"type":"projection","points":[...],"reducer":"UMAP","fitted_at":12}
{"type":"metrics","metrics":{"cpu":18.2,"ram":6.1,"gpu":72,"vram_used":4.8,"vram_total":6.0,"tps":12.4,"ttft_ms":420,"tokens":128,"context":512}}
{"type":"done","usage":{"prompt_tokens":100,"completion_tokens":128,"total_tokens":228},"finish_reason":"stop","elapsed_s":11.4}
{"type":"status","scope":"lens","state":"loading","message":"正在加载透镜模型"}
{"type":"error","code":"LMSTUDIO_UNREACHABLE","message":"..."}
```

### 4.1 TokenOption

```jsonc
{"token":"你","id":123,"logprob":-0.21,"prob":0.81,"rank":1}
```

`topk` 是模型原始分布（未应用温度）。前端拖动采样滑块时，使用同一组 `topk` 在浏览器重算 Softmax、
Top-P、Top-K 与重复惩罚，不重新请求模型。

### 4.2 Probe

```jsonc
{
  "step": 12,
  "model":"Qwen/Qwen2.5-0.5B-Instruct",
  "surrogate": true,
  "latency_ms": 38.2,
  "context": {"tokens":["<|im_start|>","user",...],"kinds":["prompt","prompt","generated"]},
  "attention": {
    "layers":[0,6,12,18,23],
    "selected":{"0":{"mean":[...],"heads":[[...]],"head_types":["local",...]}},
    "head_count":14
  },
  "lens": [{"layer":0,"topk":[...],"tuned":true,"agreement":0.61}],
  "hidden": [0.12,-0.03,...],           // 16 维降采样，用于向量条带
  "residual": [{"layer":0,"attn_norm":0.8,"mlp_norm":0.2,"attn_logit":0.3,"mlp_logit":-0.1}],
  "features": [{"id":42,"label":"紫色 / violet","activation":0.87,"tokens":["紫","violet"]}],
  "kv": {"used":312,"capacity":8192,"bytes":40894464,"bits_per_token":131072},
  "vector": {"x":0.2,"y":-0.8,"z":0.4}
}
```

为控制带宽，实时 `probe` 只带 `attention.selected` 的 2 个焦点层（默认第 12 层与最后一层）的
per-head 行；其余层只带 mean 行。完整多层数据存服务端，回放时通过 REST 获取。

### 4.3 投影点

```jsonc
{"id":"t12","token":"紫","index":12,"kind":"generated","x":0.1,"y":-0.3,"z":0.5,"step":12}
```
`reducer` 为 `UMAP`（已安装且点数 >= 8）或 `PCA`。投影后台计算，不阻塞生成。

## 5. REST

- `GET /api/health` — 版本、运行时长、LM Studio、透镜、GPU 能力。
- `GET /api/config` — 默认采样、路径、能力、模型建议。
- `GET /api/models` — 归一化 LM Studio 模型列表。
- `GET /api/sessions` — 最近会话摘要。
- `GET /api/sessions/{id}` — 会话元信息。
- `GET /api/sessions/{id}/steps?offset=0&limit=200` — 精简步骤。
- `GET /api/sessions/{id}/steps/{index}` — 单步完整快照。
- `GET /api/sessions/{id}/export` — `.jsonl` 下载，一行一个步骤（含 probe）。
- `POST /api/sessions/{id}/stop` — 尽力停止当前生成。
- `POST /api/rag/search` — `{query, top_k}`，混合检索。
- `GET /api/rag/documents` — 记忆库清单与统计。
- `POST /api/rag/reindex` — 重建索引。
- `GET /api/lens/status` — 透镜状态、模型、设备、校准结果。
- `POST /api/lens/load` — `{model?, device?}`，后台加载；不会自动下载。
- `POST /api/lens/unload` — 释放透镜模型与显存。
- `GET /api/docs/{name}` — 返回 `docs/*.md` 原文（只读白名单）。

## 6. LM Studio 适配

优先级：

1. `POST /v1/responses`，SSE，`stream=true`，`include=["message.output_text.logprobs"]`，
   `top_logprobs=12`，`store=false`。解析 `response.output_text.delta`、reasoning 事件、
   `response.completed`。这是唯一能同时拿到文本与 `top_logprobs` 的官方路径。
2. `POST /v1/chat/completions`，`stream=true`，`logprobs=true`，`top_logprobs=12`，
   额外传 `top_k`、`repeat_penalty`（LM Studio 支持）。解析 `choices[].delta`。
3. `POST /api/v1/chat`（原生 SSE），无 logprobs，只用于文本/推理流与统计。

自动降级仅在 HTTP 400/404/405 或明确的 "unsupported parameter" 时发生；连接超时不降级，
而是报 `LMSTUDIO_UNREACHABLE`，避免在用户模型加载期间反复请求。**启动应用、读取模型列表绝不
加载模型**；只有用户点击发送才会发起生成。

思考内容（reasoning）单独作为 `kind="reasoning"` 事件，对话窗默认折叠。

## 7. 采样重算

后端发送原始 top-k log-probs。前端：

1. 对重复惩罚：按最近 64 个 Token 的出现次数，对 logit 做 `>0 ? /p : *p`。
2. `logit / temperature`，Top-K 截断，Top-P 截断，Softmax。
3. 显示重新归一化后的概率云、熵、候选树。当前已生成 Token 的概率若不在 top-k 中显示 `—`。
4. 修改滑块只影响显示与**下一次请求**；当前生成中的 Token 已经由 LM Studio 采样，不能追溯更改。
   界面用 "下一 Token 生效" 标注。

## 8. 透镜与内部张量

### 8.1 替代模型

默认 `Qwen/Qwen2.5-0.5B-Instruct`，CPU 2 线程，进程优先级低于正常。模型文件需用户执行
`scripts/fetch_lens_model.ps1` 获取；应用不自动下载。也可以指定任意本地 HF 因果语言模型路径。
若模型与 LM Studio 当前模型同名同权重，界面标记 `same-weights`。

### 8.2 注意力

用 eager attention + KV cache 逐 Token 增量前向。每个 Token 记录每个头对上下文的注意力行，
以及头类型：`previous-token`、`syntactic`、`entity`、`global`、`local`。分类依据统计量
（首行权重、熵、到最近标点的距离、跨实体重复），在前端用色相区分，文档说明这是启发式分类。
完整下三角矩阵由逐 Token 行拼合；默认显示第 12 层与最后一层，允许切换所有已捕获层。

### 8.3 Tuned Lens

为每层训练仿射 Translator：`h_l @ W_l + b_l -> final_hidden`，再应用最终 LayerNorm 与
`lm_head`。校准语料为项目文档与记忆库中的短句，关闭 dropout，使用岭回归闭式解。缓存到
`data/cache/tuned_lens_<model>.pt`。同时保留 Logit Lens 作为对照，并显示 top-1 一致率。
没有校准缓存时先显示 Logit Lens，后台校准完成后切换。

### 8.4 残差流贡献

对每层捕获 attention 分支输出与 MLP 分支输出，计算 L2 范数；把分支增量经最终 LayerNorm
的一阶近似投影到词表方向，得到对当前选中 Token 的 signed logit 贡献。前端以铜色（注意力）
与橄榄色（MLP）正负柱显示。

### 8.5 稀疏特征

默认 `feature-probe`：对中间层 MLP 激活取 top-k 神经元，用 `lm_head` 的对应方向解码出
最具代表性的 Token，作为可读标签；这是无外部 SAE 权重时的诚实替代。若用户安装并配置
`sae_lens` 与匹配 checkpoint，可通过适配器接口替换。界面必须标注 `probe`，不得标成 SAE。

### 8.6 UMAP

收集 Prompt + 生成 Token 的最终层隐状态（最多 256 点）。每 8 个 Token 在后台线程重新拟合
2D/3D 投影；不足 8 点用 PCA，再不足用确定性伪投影。输出归一化到 [-1,1]。不阻塞生成。
语义点只来自透镜模型。

## 9. RAG

- 记忆库：`memory/*.md`，按段落切块（约 320 字，48 字重叠）。
- 索引：字符 3-gram TF-IDF 做 BM25；embedding 使用 LM Studio
  `text-embedding-nomic-embed-text-v1.5`（`/v1/embeddings`，批量，仅索引/查询时调用）。
  embedding 不可用时纯 BM25。
- 混合分数：`0.55 * cosine_norm + 0.45 * bm25_norm`。召回结果展示两种原始分与混合分、命中词。
- 发送时可选把 top-k 记忆注入 system prompt；默认只展示，不注入，避免额外 Token 费用。

## 10. 性能预算

- 生成期间后端不做同步磁盘写入；会话先内存，完成后落盘。
- 指标采样 1.5s 一次；nvidia-smi 以子进程运行并带超时，失败返回 null。
- WebSocket 消息合并/节流：token 立即发送，probe 默认每 2 个 Token 一次，projection 每 8 个。
- 透镜 CPU 线程数 2、进程低于正常优先级；队列落后时丢弃旧探测、保留最新。
- UI 所有高频面板使用 Canvas 或 CSS transform，避免每 Token 触发整页 React 重渲染。

## 11. 协作边界

| 角色 | 写入范围 |
| --- | --- |
| lead | 根配置、`apps/desktop/src/App.tsx`、`src/state/*`、`src/components/*.tsx`（除 visuals）、`desktop/`、集成、Git |
| backend-core | `backend/xray/*.py`（除 `lens/`）、`backend/run.py`、`backend/requirements.txt`、`backend/tests/test_*.py`（除 lens） |
| lens-engine | `backend/xray/lens/*.py`、`backend/tests/test_lens*.py` |
| visuals | `apps/desktop/src/components/visuals/*`、`apps/desktop/src/styles/visuals.css` |
| docs-qa | `docs/*.md`（除 SPEC）、`README.md`、`CHANGELOG.md`、`scripts/*.ps1`、QA 报告 |

所有共享类型以 `apps/desktop/src/types.ts` 与本文档为准；修改契约必须通知 lead。
