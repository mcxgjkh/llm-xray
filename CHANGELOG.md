# 变更记录

本文件记录 LLM-Xray 的公开变更，格式参考 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.1.0/)，
版本号遵循[语义化版本](https://semver.org/lang/zh-CN/)。版本策略与发布流程见
[docs/VERSIONS.md](docs/VERSIONS.md)。

说明：1.0.0 是首个公开版本，1.1.0 是当前版本。下列条目描述各版本的发布范围，发布时以
`scripts/release.ps1` 实际通过的测试与构建结果为准；`0.x` 开发里程碑不在此列出。

`docs/SPEC.md` 是 v1.0.0 的实现契约；v1.1.0 的增量以本文件、`docs/API.md` 与
`docs/ARCHITECTURE.md` 为准。

## [Unreleased]

暂无。

## [1.2.0] - 2026-10-27

### Added

- 最大输出上限跟随模型：LM Studio 模式下取所选模型的 max_context_length，原生模式取实际上下文
  32768；调整滑块时实时显示 Estimated Memory Usage（权重 + KV 缓存估算）。
- CUDA/CPU 双运行时选择：设置页“原生精确模型 → 设备”可选 Auto/CUDA/CPU；新增
  scripts/setup-runtime.ps1 与 start-cuda.bat / start-cpu.bat。
- GitHub 发布准备：README 增加模型配置与 CUDA/CPU 说明。

### Changed

- 原生精确模式默认上下文从 4096 提升到 32768；Auto 优先使用 CUDA，不可用时回退 CPU。
- 最大输出上限不再固定 8192 或 4096；后端安全上限提升到 262144。

## [1.1.0] - 2026-10-23

在 v1.0.0 基础上增加“原生精确”运行时，修复用户反馈的截断、导出与概率云空白问题。

### Added

- 原生精确通道：在 LLM-Xray 进程内用 Transformers 运行本地 HF 因果模型，生成 Token 与
  probe 来自同一次前向，`probe.surrogate=false`、`same_weights=true`；默认模型为本地
  `Qwen/Qwen2.5-0.5B-Instruct`，CPU 推理，需显式加载且不会自动下载。
- 原生运行时接口与状态：`GET /api/native/status`、`POST /api/native/load`、
  `POST /api/native/unload`；顶栏“引擎”选择（LM Studio / 原生精确 / 演示）与设置页加载、
  卸载、状态显示。
- JSONL 导出原生保存：桌面版通过 pywebview 打开系统保存对话框并直接写文件；普通浏览器
  回退为 Blob 下载；成功或取消都会显示提示。对话窗标题栏新增导出按钮。
- 透镜引导（CTA）：对话窗在透镜未加载、加载中、加载失败时给出可直接操作的提示。
- Tuned Lens 校准缓存复用：校准结果写入并在下次启动时从
  `data/cache/tuned_lens_<model>.pt` 读取，不再每次重新训练。
- LM Studio no-logprobs 警告：前 5 个回答 Token 都没有 top-k 时发送
  `status(scope=runtime, state=no_logprobs)`，会话能力标记 `logprobs=false`，界面提示切换到
  原生精确模式。
- 模型下载脚本增加回退：`scripts/fetch_lens_model.ps1` 支持 `HF_ENDPOINT`、自动重试
  `https://hf-mirror.com`，并在 Hugging Face 不可达时通过 ModelScope 直接文件接口下载。

### Changed

- 默认 `max_tokens` 从 512 提升到 4096；采样面板与设置页新增“最大输出”滑块（256 到 8192），
  仍受模型上下文限制。
- v1.0 保存在 localStorage 中的旧 `max_tokens` 值（小于等于 512 或无值）会一次性迁移到 4096，
  之后以用户选择为准。
- 文档同步到 v1.1.0：README、使用说明、架构、可解释性、API、限制与 FAQ、版本表、暖色文档页。

### Fixed

- 生成长回答在 512 Token 处被截断：默认值与 UI 上限修正，并提供旧值迁移。
- WebView2 中点击 JSONL 导出没有反应：改为原生保存对话框，浏览器环境保留 Blob 下载回退。
- LM Studio 实际流不返回 `top_logprobs` 时概率云空白且无任何说明：新增检测、状态事件与
  切换建议。
- Qwen 等 byte-level BPE tokenizer 的中文 Token 显示乱码：统一改用 tokenizer `decode`
  解码，失败才回退原始 token。
- Tuned Lens 缓存未生效、每次启动重复校准：加载并复用已有校准缓存。

### Security

- 原生精确通道以 `local_files_only` 加载本地权重，不联网、不自动下载，也不修改 LM Studio
  的模型状态。
- 同一时间只允许一个原生生成会话，避免两个进程内前向同时占满 CPU/内存；冲突时返回 `BUSY`。
- 导出仍由用户选择保存位置；取消保存不会写入文件。

## [1.0.0] - 2026-10-21

首个正式版本：本地双通道 LLM 观测台。

### Added

- 真实遥测通道：对接 LM Studio `/v1/responses`（优先）、`/v1/chat/completions` 与原生
  `/api/v1/chat`，采集真实 Token 流、`top_logprobs`、TTFT、tok/s、结束原因与用量；连接超时不
  降级，避免模型加载期间反复请求。
- 可选替代透镜通道：默认 `Qwen/Qwen2.5-0.5B-Instruct`，懒加载、CPU 2 线程、低优先级；通道
  默认关闭，开启后不阻塞真实 Token 流。
- 概率云与采样重算：基于后端原始 top-k 在浏览器内重算 Softmax、Top-P、Top-K 与重复惩罚，
  显示熵、选中 Token 概率；标注“下一 Token 生效”。
- Tuned Lens 与 Logit Lens 对照：逐层仿射 translator、岭回归校准、校准缓存与 top-1 一致率。
- 注意力热图：eager attention + KV cache 的逐 Token 行捕获，多层/头切换，启发式头分类。
- 残差流贡献：attention/MLP 分支 L2 范数与一阶投影的正负柱状图。
- 稀疏特征探针：MLP top-k 神经元经 `lm_head` 方向解码，标注为 `probe` 而非 SAE。
- 3D 语义投影：最多 256 点的 UMAP/PCA/确定性伪投影，后台节流拟合。
- RAG 记忆库：`memory/*.md` 的字符 n-gram BM25 + LM Studio embedding 混合检索，召回结果可
  展示或注入 system prompt。
- 会话与回放：会话落盘 `data/sessions/`，时间穿梭逐 Token 回放，JSONL 导出（含 probe 快照）。
- 桌面外壳：pywebview + Edge WebView2；右侧对话窗可拖拽调宽、可折叠；reasoning 默认折叠。
- 暖色界面与文档站：应用主题与 `docs/index.html` 文档页使用深炭、羊皮纸、铜、橄榄配色，
  无 emoji、无渐变。
- 完整中文文档：使用说明、架构、可解释性方法、API、限制与 FAQ、版本管理；应用内“技术文档”页
  可直接读取并渲染 `docs/*.md`。
- 版本与发布工具：`VERSION`、`CHANGELOG.md`、`docs/VERSIONS.md`、
  `scripts/check-version.ps1`、`scripts/release.ps1`、`scripts/dev.ps1`、`scripts/build.ps1`、
  `scripts/fetch_lens_model.ps1`，以及 Windows CI 工作流。

### Changed

- 首版即确立“真实通道”与“替代透镜”双通道标注：所有替代结果带 `surrogate` 徽标，指向同一
  权重时标记 `same-weights`。
- 采样参数调整与已生成 Token 解耦：参数只影响显示重算与下一次请求，不再被误读为追溯修改。
- 透镜通道从“被观测模型”改为显式可选的独立模型，并允许替换为任意本地 HF 因果语言模型。

### Fixed

- 无：1.0.0 是首个公开版本，不包含对公开版本的修复记录。

### Security

- 默认仅监听 `127.0.0.1`，不提供公网访问、鉴权或多用户设计。
- 应用不包含云 API 客户端，不上传会话、遥测或指标；RAG embedding 走本机 LM Studio。
- 应用不自动下载模型；透镜模型获取需用户显式运行 `scripts/fetch_lens_model.ps1`。
- 会话、缓存、导出文件均落在本机 `data/`、`models/` 目录，`.gitignore` 排除运行期数据。
- 文档与界面统一提示：替代模型不是被观测模型，避免把近似结果当作真实内部状态。
