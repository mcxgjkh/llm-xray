# 版本管理

本页是 LLM-Xray 的版本策略与版本记录入口。变更明细见 [CHANGELOG.md](../CHANGELOG.md)，
实现契约见 [SPEC.md](SPEC.md)（v1.0.0）；v1.1.0 的增量以 CHANGELOG、API 与 ARCHITECTURE 为准。

## 1. 版本号规则

采用 [语义化版本](https://semver.org/lang/zh-CN/)：`MAJOR.MINOR.PATCH`。

| 变更类型 | 版本位 | 示例 |
| --- | --- | --- |
| 不兼容的协议、数据格式或配置变更 | MAJOR | 删除 WS 事件字段、改变 Probe 语义 |
| 向后兼容的新功能、新面板、新接口 | MINOR | 新增一个观测面板或 REST 端点 |
| 向后兼容的修复、文案、性能优化 | PATCH | 修复概率云渲染、修正文档命令 |

预发布版本使用 `MAJOR.MINOR.PATCH-rc.N`（如 `1.0.0-rc.1`），只用于发布候选，不改变
正式版本的兼容承诺。

## 2. 单一事实来源

版本号必须保持以下位置一致：

| 位置 | 内容 | 维护方式 |
| --- | --- | --- |
| `VERSION` | 纯文本版本号，例如 `1.1.0` | 手工修改，发布脚本校验 |
| `apps/desktop/package.json` | `"version"` 字段 | 手工修改，发布脚本校验 |
| `CHANGELOG.md` | `## [1.1.0] - YYYY-MM-DD` 标题 | 发布时补写 |
| `docs/VERSIONS.md` | 本页版本表 | 发布时补写 |
| Git 标签 | annotated tag `v1.1.0` | 发布脚本创建 |

`scripts/check-version.ps1` 同时校验前四项，供 CI 与本地检查使用。任何一项不一致都视为
版本错误，CI 会失败。

## 3. 发布流程

1. 确认 `VERSION` 与 `package.json` 中的版本一致，且已按计划递增。
2. 更新 `CHANGELOG.md`：新增 `## [版本] - 日期` 段落，按 Added / Changed / Fixed / Security
   分类，只写已实现且已验证的内容。
3. 更新本页版本表，写明版本类型与要点。
4. 运行本地校验：

```powershell
powershell -ExecutionPolicy Bypass -File scripts\check-version.ps1
```

5. 运行测试与构建（发布脚本也会执行）：

```powershell
python -m pytest backend -q
npm --prefix apps\desktop run build
```

6. 确认工作区干净、目标标签不存在，然后 dry-run 演练：

```powershell
powershell -ExecutionPolicy Bypass -File scripts\release.ps1 -DryRun
```

7. 去掉 `-DryRun` 执行发布，创建 annotated tag：

```powershell
powershell -ExecutionPolicy Bypass -File scripts\release.ps1
```

8. 需要推送时显式加 `-Push`，或手动 `git push origin v<version>`。

发布脚本在任一检查失败时立即退出，不会创建工作区修改或标签。

## 4. 标签规则

- 标签格式固定为 `v<MAJOR>.<MINOR>.<PATCH>`，与 `VERSION` 完全一致；预发布为 `v<version>-rc.N`。
- 必须使用 annotated tag，附带版本说明；禁止 lightweight tag。
- 标签一旦发布不得移动、删除或覆盖；发现错误时递增版本号重新发布，并在 CHANGELOG 中说明。
- 同一标签不得指向两个提交；发布前脚本会检查标签是否已存在。
- 分支与标签的关系：`main` 分支上的发布提交打标签；补丁分支合并后同样按新 PATCH 标签发布。

## 5. 兼容性承诺

在 v1 系列内：

- WebSocket 事件与 REST 响应**只增不改**：不删除已有字段，不改变字段语义；新增字段必须允许
  缺省，旧前端在字段缺失时不能崩溃。v1.1.0 的 `mode=native`、原生状态接口、
  `done.capabilities`、`done.exact` 均按此规则新增；旧客户端不传 `mode` 仍默认 `lmstudio`。
- `ReplayStep`、`Probe`、`TokenOption` 的结构保持向后兼容；破坏性调整只允许在 MAJOR 版本。
  `Probe.surrogate=false` 是 v1.1.0 新增的语义（原生精确通道），旧数据仍为 `true` 或缺失。
- `docs/SPEC.md` 是实现契约；修改契约必须先更新 SPEC 并同步前端 `types.ts`，再修改实现。
- 配置键名保持稳定；删除配置键需要一个新的 MINOR 版本的迁移说明。
- 会话与导出数据的 JSONL 格式在 v1 内可读；新增可选字段不视为破坏性变更。
- 版本号不承诺时间节奏，只承诺语义。

## 6. 版本表

| 版本 | 日期 | 类型 | 要点 |
| --- | --- | --- | --- |
| 2.0.0 | 未定 | 计划 | 占位：可能包含协议不兼容调整，具体内容未定 |
| 1.2.0 | 未定 | 计划 | 占位：新增能力与面板，具体内容未定 |
| 1.1.1 | 未定 | 计划 | 占位：v1.1.0 之后的兼容修复集合 |
| 1.1.0 | 2026-10-23 | 正式版 | 原生精确通道、导出原生保存、no-logprobs 警告、默认 4096、Tuned Lens 缓存、中文 Token 解码、ModelScope 回退 |
| 1.0.0 | 2026-10-21 | 正式版 | 首个正式版本：LM Studio 通道 + 替代透镜、概率云、透镜面板、回放与导出、暖色文档站、版本与发布脚本 |
| 0.9.0 | 开发里程碑 | 前史 | 视觉与回放：注意力热图、3D 向量云、时间回放；未对外发布、无公开标签 |
| 0.5.0 | 开发里程碑 | 前史 | 双通道：LM Studio 真实遥测 + 可选的替代透镜通道；未对外发布 |
| 0.1.0 | 开发里程碑 | 前史 | 脚手架：FastAPI 后端、React 外壳、事件协议与目录结构；未对外发布 |

说明：

- `0.x` 版本是开发过程中的里程碑，用于标记内部阶段，不是公开发行版，也没有对应的公开标签。
  它们不承诺兼容性。
- 计划中的版本只是占位，不代表已排期；实际版本号与内容以 `CHANGELOG.md` 和 Git 标签为准。
- 日期使用 ISO 8601（`YYYY-MM-DD`）。

## 7. 版本历史记录方式

- 每个正式版本在 `CHANGELOG.md` 中必须有对应段落，包含日期与 Added / Changed / Fixed /
  Security 分类。
- 已发布版本的 CHANGELOG 段落不再修改；补充说明以新版本或未发布段落记录。
- 版本表与 CHANGELOG 的分工：本页记录“版本是什么、策略是什么、有哪些版本”，CHANGELOG 记录
  “每个版本具体改了什么”。
- 任何 `0.x` 里程碑若要转为正式发布，必须重新编号为 v1 或更高，并补齐测试与版本校验。
