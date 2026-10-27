# LLM-Xray

LLM-Xray 是一个暖色界面的本地 LLM 观测桌面应用：在 Windows 上通过 pywebview + WebView2 打开，
后端是 FastAPI，前端是 React/TypeScript/Three.js。它把 LM Studio 的实时生成遥测、一个可选的
替代模型透镜通道，以及一个可选的进程内“原生精确”通道并排放在同一界面上，用来观察 Token 概率、
注意力、残差与语义投影。

项目定位是“观测台”，不是“解释器”。真实数据与近似数据严格分开显示：替代透镜的结果带
`surrogate` 徽标；原生精确通道的 probe 来自同一次前向（`surrogate=false`），但解释方法本身仍是
启发式的，不能被当作因果证明。

当前版本：`1.2.0`。`docs/SPEC.md` 是 v1.0.0 的实现契约；v1.1.0 的增量（原生精确通道、导出、
no-logprobs 警告等）见 [CHANGELOG.md](CHANGELOG.md)、[docs/API.md](docs/API.md) 与
[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)。

## 核心亮点

- 真实 LM Studio 遥测：Token 流、`top_logprobs`、TTFT、tok/s、结束原因、用量，全部来自
  LM Studio 本地服务器，而不是重放的假数据。
- 原生精确通道：在应用进程内用 Transformers 运行本地 HF 模型，生成与观测共用同一次前向，
  probe 标注 `surrogate=false`；默认本地 `Qwen/Qwen2.5-0.5B-Instruct`，CPU 推理，需显式加载。
- 替代透镜通道：一个可选的独立小模型，默认不启动，开启后不阻塞生成；用于 LM Studio 模式下
  观察近似内部状态。
- 概率云与解码树：后端给出原始 top-k 分布，前端用同一组数据在浏览器内重算 Softmax、Top-P、
  Top-K 和重复惩罚，拖动采样滑块不需要重新请求模型。
- 最大输出可调：默认 4096 Token（v1.0 固定为 512，长回答会被截断），采样面板与设置页提供
  256 到 8192 的滑块；旧版本保存的 512 值会一次性迁移。
- 内部状态可视化：Tuned Lens / Logit Lens、注意力热图、注意力与 MLP 的残差贡献、稀疏特征探针、
  UMAP/PCA 3D 语义投影、KV 缓存估算。
- 时间穿梭回放与 JSONL 导出：一次运行可以逐 Token 回看；桌面版用系统保存对话框导出，
  浏览器版回退为 Blob 下载，并显示导出结果提示。
- 本地优先：默认只监听 `127.0.0.1`，不使用云服务，不上传会话；记忆库是本地 `memory/*.md`。
- 低开销设计：Canvas/SVG 自绘图表、指标约 1.5 秒采样、probe 节流、透镜 CPU 2 线程与低优先级、
  队列落后时丢弃旧探测。
- 明确的缺失提示：LM Studio 实际流没有返回 `top_logprobs` 时，界面在若干 Token 后提示概率云为空，
  并建议切换到原生精确模式。

## 必须理解的边界

LM Studio 的 HTTP API 只暴露文本、采样参数、统计信息和可选的 `top_logprobs`，**不暴露隐藏状态、
逐层注意力或 MLP 输出**，部分版本/接口甚至不返回 `top_logprobs`。因此：

- 界面上的概率云、TTFT、tok/s 等来自你正在使用的 LM Studio 模型，是真实遥测；但概率云可能因为
  LM Studio 未返回 top-k 而为空。
- LM Studio 模式下的注意力热图、Tuned Lens、残差、稀疏特征、3D 向量云默认来自另一个替代小模型
  （默认 `Qwen/Qwen2.5-0.5B-Instruct`），它是**近似观察**，不是同一个模型的内部状态。
- 原生精确模式在应用进程内加载本地 HF 模型，生成与观测同一次前向，因此 probe 是**同一模型**的
  精确张量；但它是你加载的本地模型，不是 LM Studio 里的 8B 模型，除非你显式加载相同权重。
  0.5B 在 CPU 上可用但慢；8B 级别在普通 16GB 机器上不现实。
- `same-weights` 标记表示透镜或原生权重与 LM Studio 当前模型同名同权重，这是跨到 LM Studio 模型
  内部状态的唯一路径，但仍有量化差异。
- 探针结果是启发式近似（例如 attention head 分类、稀疏特征标签），即使张量精确，也不构成因果解释。

这些边界的完整清单见 [docs/LIMITATIONS.md](docs/LIMITATIONS.md)，透镜原理与正确解读方式见
[docs/INTERPRETABILITY.md](docs/INTERPRETABILITY.md)。


## 快速开始（Windows）

前置条件：

- Windows 10/11，Edge WebView2 运行时（安装 LM Studio 时通常已具备）。
- Python 3.11。
- Node.js 22（仅开发与构建前端时需要）。
- LM Studio 0.4.x，本地服务器默认端口 `1234`。

1. 准备 LM Studio：在 LM Studio 中加载 `deepseek-r1-distill-llama-8b`，打开
   `Developer` -> `Local Server`，确认地址为 `http://127.0.0.1:1234`。启动 LLM-Xray 或读取模型列表
   时不会替你加载模型，只有点击发送才会发起生成。
   如果 LM Studio 的流没有返回 `top_logprobs`，概率云会为空，界面会提示切换到原生精确模式。

2. 安装依赖（在项目根目录执行）：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r backend\requirements.txt
npm --prefix apps\desktop install
```

3. 启动开发环境（后端 `127.0.0.1:8765` + Vite `127.0.0.1:5173`）：

```powershell
powershell -ExecutionPolicy Bypass -File scripts\dev.ps1
```

4. 构建并运行桌面版：

```powershell
powershell -ExecutionPolicy Bypass -File scripts\build.ps1
python desktop\launcher.py
```

可选：启用替代透镜通道。透镜模型需要用户显式获取，应用不会自动下载：

```powershell
pip install -r backend\requirements-lens.txt
powershell -ExecutionPolicy Bypass -File scripts\fetch_lens_model.ps1
```

下载脚本会依次尝试 Hugging Face、`hf-mirror.com`，并可在 Hugging Face 不可达时回退到 ModelScope；
权重默认放入 `models/lens/`，应用会自动解析。然后在应用的设置页点“加载透镜”，再打开“允许透镜
采样”。应用以本地文件加载模型，不会自动联网。首次使用 Tuned Lens 需要一次后台校准，结果会缓存到
`data/cache/`，之后启动直接复用。

可选：使用原生精确模式（不经过 LM Studio）。它需要同一套 PyTorch/Transformers 依赖和一个本地
HF 因果模型；默认 `Qwen/Qwen2.5-0.5B-Instruct`：

```powershell
pip install -r backend\requirements-lens.txt
powershell -ExecutionPolicy Bypass -File scripts\fetch_lens_model.ps1
python desktop\launcher.py
```

启动后在顶栏把“引擎”切到“原生精确”，在设置页点“加载原生模型”。0.5B 在 CPU 上首次加载约
20-30 秒，生成速度明显慢于 LM Studio，但生成与观测来自同一次前向（`surrogate=false`）。
8B 级别模型在普通 16GB 内存机器上不现实，不建议用原生模式运行。

没有 LM Studio 或模型尚未加载时，可以在顶栏把引擎切到“演示”，或使用对话窗空状态的“演示模式”
按钮查看布局与图表；演示数据不是真实模型输出。详细步骤见 [docs/USAGE.md](docs/USAGE.md)。

## 目录结构

```text
backend/xray/            FastAPI、LM Studio 适配、原生精确桥接、会话、采样、RAG、回放、指标
backend/xray/lens/       透镜模型引擎、Tuned Lens、注意力、残差、稀疏特征、UMAP、demo 合成器
apps/desktop/            React 界面（Vite + TypeScript + Three.js）
desktop/                 pywebview 启动器与原生文件保存桥
docs/                    中文技术文档、暖色静态文档页（docs/index.html）
scripts/                 开发、构建、发布、模型准备脚本
memory/                  本地记忆库（Markdown，RAG 数据源）
data/                    运行期缓存与会话（git 忽略）
models/                  用户获取的本地模型（git 忽略）
```

## 文档索引

| 文档 | 内容 |
| --- | --- |
| [docs/USAGE.md](docs/USAGE.md) | 完整使用说明：启动、界面、采样、图表、回放、导出、性能建议 |
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | 运行时与观测通道、LM Studio 适配优先级、原生精确通道、事件流、性能预算、隐私 |
| [docs/INTERPRETABILITY.md](docs/INTERPRETABILITY.md) | Tuned Lens、Logit Lens、注意力、残差、稀疏特征、UMAP 的原理与误读风险 |
| [docs/API.md](docs/API.md) | REST、WebSocket、Probe 字段表、示例 JSON、错误码 |
| [docs/LIMITATIONS.md](docs/LIMITATIONS.md) | 限制、常见问题与排错 |
| [docs/VERSIONS.md](docs/VERSIONS.md) | 版本策略、发布流程、版本表 |
| [CHANGELOG.md](CHANGELOG.md) | 变更记录（Keep a Changelog 风格） |
| [docs/SPEC.md](docs/SPEC.md) | v1.0.0 实现契约与技术规格；v1.1.0 增量见 CHANGELOG、API 与 ARCHITECTURE |

静态文档页 `docs/index.html` 可离线直接用浏览器打开，只引用同目录的 `docs/docs.css`；应用内的
“技术文档”页会读取 `docs/*.md` 并在暖色阅读器中渲染。

## 开发与测试

```powershell
# 后端测试（不需要真实 8B 模型，使用 mock SSE 与 fixtures）
Push-Location backend; python -m pytest -q; Pop-Location

# 前端类型检查与构建
npm --prefix apps\desktop run typecheck
npm --prefix apps\desktop run build

# 版本一致性检查（CI 使用）
powershell -ExecutionPolicy Bypass -File scripts\check-version.ps1

# 发布（校验版本、干净工作区、测试与构建通过后创建 annotated tag）
powershell -ExecutionPolicy Bypass -File scripts\release.ps1 -DryRun
```

发布脚本默认只做校验与 dry-run；去掉 `-DryRun` 才会创建 `v<version>` annotated tag。

## 许可与第三方依赖

本仓库目前不附带项目级 `LICENSE` 文件；在对外分发或再发布前，请由维护者先确定项目许可证。
以下依赖遵循其各自的许可证，使用与分发时需保留对应声明：

- 前端：React（MIT）、Three.js（MIT）、marked（MIT）、DOMPurify（Apache-2.0 或 MPL-2.0，按上游声明）。
- 后端：FastAPI（MIT）、Uvicorn（BSD-3-Clause）、httpx（BSD-3-Clause）、NumPy（BSD-3-Clause）、
  scikit-learn（BSD-3-Clause）、psutil（BSD-3-Clause）、pywebview（BSD-3-Clause）。
- 可选透镜/原生精确：PyTorch（BSD-3-Clause）、Transformers（Apache-2.0）、UMAP（BSD-3-Clause）、
  tuned-lens 参考实现（MIT）。
- 模型权重归各自发布方所有：LM Studio 中使用的 `deepseek-r1-distill-llama-8b`、透镜与原生精确
  模式默认使用的 `Qwen/Qwen2.5-0.5B-Instruct` 均有各自的模型卡与许可证，下载和使用前请阅读
  对应条款。下载脚本只在你显式运行时联网，应用本身不会自动下载权重。
- LM Studio 是独立商业软件，本项目只调用其本地 HTTP API，不包含或再分发其程序。

文档页配色取自应用主题：深炭 `#17130f`、羊皮纸 `#f2e7d5`、铜 `#d08a4e`、橄榄 `#9aa36b`。


## CUDA 与 CPU 运行时

同一个安装包内含两种设备选择，不需要两个代码副本。打开“设置 → 原生精确模型 → 设备”，
选择 `Auto`、`CUDA` 或 `CPU`；`Auto` 在 CUDA 可用时优先使用 GPU，否则回退 CPU。
实际使用的设备显示在设置页状态与顶栏。

安装对应 PyTorch：

```powershell
# NVIDIA GPU（推荐）
powershell -ExecutionPolicy Bypass -File scripts\setup-runtime.ps1 -Device cuda

# 仅 CPU
powershell -ExecutionPolicy Bypass -File scripts\setup-runtime.ps1 -Device cpu
```

也可以使用 `start-cuda.bat` 或 `start-cpu.bat`；CPU 启动脚本会设置
`XRAY_NATIVE_DEVICE=cpu`。LM Studio 模式始终由 LM Studio 自己管理 GPU/CPU，不受此选择影响。
原生 CUDA 模式当前适合 0.5B–1.5B；8B 的 FP32 权重约 32 GB，需要 4bit 量化与更大显存。
