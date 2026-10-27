# 可解释性方法说明

本文解释 LLM-Xray 每个透镜面板背后的方法、数学直觉和误读风险。目标不是把模型“解释清楚”，
而是让你知道每个数字是怎么来的、能支持什么结论、不能支持什么结论。

先读这一段：

> LM Studio 不提供模型内部张量。LM Studio 模式下，除 `same-weights` 路径外，所有隐藏状态、
> 注意力、Tuned Lens、残差、稀疏特征与 UMAP 投影都来自一个**独立的小模型**（默认
> `Qwen/Qwen2.5-0.5B-Instruct`），它和你在对话的模型不是同一个。这些结果是“另一个模型如何处理
> 同一段文本”的近似观测，不能当作目标模型的内部真相，界面上的 `surrogate` 徽标就是在提醒这一点。
>
> v1.1.0 增加了原生精确模式：在应用进程内运行本地 HF 模型，生成与观测共用同一次前向，
> `probe.surrogate=false`、`same_weights=true`。这解决了“张量来源”问题，但**不解决解释问题**：
> 注意力权重、头分类、残差归因、稀疏特征标签仍然是启发式或近似方法，张量精确不等于因果结论。

## 1. Tuned Lens

### 1.1 原理

Transformer 各层的隐藏状态维度相同，但不同层的表示并不直接位于最终层的坐标系里。Tuned Lens
假设存在一个逐层仿射映射，可以把第 `l` 层的隐藏状态翻译到最终层：

```text
h'_l = h_l @ W_l + b_l
logits_l = lm_head( final_layer_norm(h'_l) )
```

其中 `W_l`、`b_l` 是为每一层单独训练的 translator。它不是简单地复用 `lm_head`，而是先学会
“如何把中间层的表示搬到最后一层”，再用与最终输出完全相同的 LayerNorm 和 `lm_head` 解码。
这样各层的预测词表与最终输出在同一空间里，层间比较才有意义。

参考：

- Belrose et al., *Eliciting Latent Predictions from Transformers with the Tuned Lens*
  (arXiv:2303.08112)
- 参考实现：[AlignmentResearch/tuned-lens](https://github.com/AlignmentResearch/tuned-lens)

### 1.2 本项目如何校准

- 校准语料：`docs/*.md`、`memory/*.md` 以及 README/CHANGELOG 中的短句（不调用任何云端服务）。
- 训练方式：把每层隐藏状态作为输入、最终层隐藏状态作为目标，关闭 dropout，用岭回归闭式解
  拟合 `W_l`、`b_l`。岭回归比梯度训练便宜，能在 CPU 上完成。
- 校准层：对模型的每一层分别拟合 translator；校准有总时间预算，超时后未完成的层回退到
  Logit Lens。结果缓存到 `data/cache/tuned_lens_<model>.pt`；缓存不存在时，面板先显示
  Logit Lens，后台校准完成后再切换。v1.1.0 起启动会直接加载已有缓存并复用，不再每次重训。
- 校准数据有限，translator 只是局部最优近似，不会修正目标模型与替代模型之间的差异。

### 1.3 agreement 是什么

面板中的 `agreement` 是 Tuned Lens 在该层的 top-1 预测与校准目标在最终输出空间中的 top-1 一致率
（在固定的校准语料上统计）。它衡量 translator 把该层表示翻译到最终输出空间后，与目标下一 Token
预测的接近程度；它不是准确率、不是模型置信度，也不能跨模型直接比较。agreement 低通常说明该层
表示离最终层较远，或校准语料不足；它不能证明该层“没有语义”。

### 1.4 如何使用

- 在面板提供的代表性层（3 个，包含最后一层）上观察“预测在哪些层开始收敛到最终答案”，适合看
  趋势，不适合看绝对数值。
- 对比 Tuned Lens 与 Logit Lens：如果两者差异很大，说明该层需要仿射校正，直接套 `lm_head`
  可能误导。
- 不要用某层的 top-1 词去解释最终输出；模型的输出是整条残差路径与采样共同作用的结果。

## 2. Logit Lens 对照

Logit Lens 直接把最终 LayerNorm 与 `lm_head` 作用到中间层隐藏状态：

```text
logits_l = lm_head( final_layer_norm(h_l) )
```

它假设中间层已经大致位于最终层的坐标系，早期层通常不成立。LLM-Xray 保留它作为对照基线：

- 计算成本低，不需要校准；
- 在较深层往往与 Tuned Lens 接近，在浅层经常给出无意义候选；
- 两个方法同时显示时，差异本身比任一方法的单点结果更有信息量。

## 3. 注意力头分类

### 3.1 捕获方式

透镜模型使用 eager attention 与 KV cache 做逐 Token 增量前向。每个 Token 记录每个头对上下文的
注意力行（形状为 `1 x 头数 x 上下文长度`），下三角矩阵由逐 Token 行拼合而成。默认在 2 个焦点层
捕获逐头数据（中间层与最后一层），其他层只保留平均行，以控制传输带宽。

### 3.2 头类型启发式

界面把每个头标成以下类别之一，依据是该头注意力行的统计量：

| 类型 | 启发式依据（近似） |
| --- | --- |
| `previous-token` | 大部分权重落在紧邻的前一个 Token |
| `local` | 权重集中在最近若干 Token |
| `syntactic` | 权重随标点、分隔符等结构位置增强 |
| `entity` | 对上下文里重复出现的实体片段有稳定权重 |
| `global` | 权重分散到长距离位置且熵较高 |
| `diffuse` | 无明显集中模式，接近均匀 |

这是**统计启发式分类**，不是人工标注的真值，也不是论文中的严格定义。同一个头在不同输入、
不同层下可能改变类别；分类器阈值也没有经过大规模验证。

### 3.3 注意力不等于重要性

注意力权重高只说明“这个头在这里读取得多”，不说明该 Token 对最终输出有因果影响。头可以被
后续层补偿，注意力也可以被冗余的头复制。不要从热图直接得出“模型关注了 X 所以输出了 Y”的
结论。可参考的经典观察：

- Voita et al., *Analyzing Multi-Head Self-Attention* (arXiv:1905.09418)
- Clark et al., *What Does BERT Look At?* (arXiv:1906.04341)
- Transformer Circuits Thread: <https://transformer-circuits.pub/>

## 4. 残差流贡献

### 4.1 方法

每层有两个分支：attention 分支和 MLP 分支。本项目分别捕获它们的输出，计算 L2 范数
（`attn_norm`、`mlp_norm`），并把分支增量经最终 LayerNorm 的一阶近似投影到词表方向，得到对
当前选中 Token 的 signed logit 贡献（`attn_logit`、`mlp_logit`）。

### 4.2 为什么只是近似

- 一阶近似忽略 LayerNorm 的非线性（缩放与平移依赖当前向量）。
- 忽略后续层的非线性交互：同一个 MLP 输出在不同上下文中的效果可能不同。
- 只投影到“当前选中 Token”的方向，不解释其他候选。

因此柱状图适合看同一步内各层的相对量级和正负方向，不适合把数值相加当成完整的 logit 分解，
也不能当作因果贡献。对因果结论，需要干预实验（消融、替换、patch），本项目不提供这些实验。

## 5. 稀疏特征探针（不是 SAE）

### 5.1 默认实现：feature-probe

没有外部 SAE 权重时，默认做法：

1. 取中间层 MLP 激活的 top-k 神经元（激活值排序）；
2. 用 `lm_head` 中与该神经元对应的方向解码出最有代表性的 Token 作为可读标签；
3. 记录激活值与相关 Token。

这是“神经元到词表方向”的投影，属于探针式近似。它给出的是**相关性线索**：某个神经元在当前
上下文里被激活，且它的输出方向偏向某个 Token。它不保证该神经元是单义的，也不保证标签词就是
该神经元的功能。

### 5.2 与 SAE 的区别

| 方面 | feature-probe（默认） | SAE |
| --- | --- | --- |
| 是否需要额外权重 | 不需要 | 需要与模型匹配的 SAE checkpoint |
| 特征是否稀疏解耦 | 否，神经元可以多义 | 训练目标就是稀疏、过完备特征 |
| 可读标签来源 | `lm_head` 方向解码 | 特征本身的激活样本、自动标注 |
| 可解释性强度 | 弱，相关线索 | 相对强，但仍非因果 |

界面把默认结果标注为 `probe`，不会标成 SAE。若用户安装并配置 `sae_lens` 与匹配的 checkpoint，
可以通过透镜引擎的适配器接口替换，届时界面才标注 `sae`。没有匹配 checkpoint 时不得宣称 SAE。

## 6. UMAP 语义投影

- 输入：替代透镜模型在 Prompt 与生成 Token 上的最终层隐状态（最多 256 点）。
- 降维：每 8 个 Token 在后台线程重新拟合 UMAP 到 2D/3D；点数不足 8 时用 PCA；再不足用确定性
  伪投影。结果归一化到 `[-1,1]`。
- UMAP 保留的是局部邻域结构，不是全局距离。点靠近只表示“在当前拟合的局部邻域中相似”，
  不表示语义等价，也不表示相似度可量化。
- 每次重新拟合都会改变坐标系，**不要跨刷新比较坐标**，也不要把 LM Studio 模型的输出投影到
  这些点上；点云只来自替代透镜模型。
- 参考：McInnes et al., *UMAP: Uniform Manifold Approximation and Projection*
  (arXiv:1802.03426)；文档 <https://umap-learn.readthedocs.io/>。

## 7. 如何正确解读

可以做：

- 在原生精确模式下，对同一个本地模型的注意力、透镜与残差做同模型对照。
- 在同一模型、同一校准缓存、同一会话内比较不同层的趋势。
- 用概率云判断解码分布的形状，用熵比较不同步骤的确定性。
- 用 JSONL 导出做离线统计，记录参数与可复现设置。
- 把透镜面板当作假设生成器：看到可疑模式后，设计对照实验（换 seed、换上下文、换层）。

不要做：

- 不要把替代模型的注意力或透镜结果说成目标模型的内部状态；原生精确模式也只代表你加载的本地
  模型，不是 LM Studio 中另一个模型的状态。
- 不要从注意力权重推断因果重要性。
- 不要用单层 top-1 词解释最终输出。
- 不要把 UMAP 坐标当作语义相似度的度量。
- 不要因为 `agreement` 高就认为解释“正确”；它只说明该层的 translator 在当前校准语料上接近
  最终预测。
- 不要跨模型、跨量化版本比较绝对值；量化和模型差异会让数字不可比。
- 不要因为张量来自“同一次前向”就跳过对照实验；精确的来源不等于因果解释。

## 8. 进一步阅读

- Tuned Lens 论文与实现：<https://github.com/AlignmentResearch/tuned-lens>
- Transformer Circuits Thread：<https://transformer-circuits.pub/>
- UMAP 文档：<https://umap-learn.readthedocs.io/>
- 注意力头分析：arXiv:1905.09418、arXiv:1906.04341
- Logit Lens 原始讨论：nostalgebraist, *interpreting GPT: the logit lens* (2020)

方法边界与已知失败模式见 [LIMITATIONS.md](LIMITATIONS.md)；数据字段见 [API.md](API.md)。
原生精确模式的张量来自与生成同一次前向（`surrogate=false`），适合做本模型内部的对照实验；
LM Studio 模式的替代透镜适合看趋势，两者不要混在一张图里比较。
