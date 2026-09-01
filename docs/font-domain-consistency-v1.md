# 回单字体域一致性 v1（独立实验）

## 目标和边界

本实验回答的是：一张回单里参与检查的多个文本区域，是否来自同一个已知的“渲染域”。渲染域应按业务可验证的组合定义，例如 `ios_alipay`、`android_alipay` 或更细的 `ios_18_alipay_10`；它不等同于某个 TTF/OTF 文件名。

`PASS` 只表示区域间字体域证据一致，不表示图片真实；任意一个已经通过逐区域门禁的异域结果都会使整图进入 `REVIEW`，高质量冲突另带强冲突原因；`UNKNOWN` 表示信息量、覆盖率或模型支持不足。所有输出固定包含 `authenticity: "not_assessed"`。

这条链路是独立 sidecar，不修改冻结的 v4 OCR、exact14 结果、模型或缓存。之前的金额减号漏识别已经由检测框裁切实验定位，仍应作为 OCR 几何问题单独处理，不能用字体域模型替代。

## 为什么按“整图单域”建模

“一张图只能出现一种字体”更准确的工程表达是：业务正文应有一个主导渲染域。粗体、字号、数字/中文 glyph、系统 fallback 可能在同图出现，因此不能要求每个像素都来自同一个字体文件。实现采用以下规则：

- 每个文本区域先输出已知域或 `unknown`；低质量/OOD 不硬判。
- 至少三个有效区域、至少两个角色才允许整图通过。
- 任意一个已经通过逐区域门禁的异域区域都会触发 `REVIEW`；高质量冲突会额外标记强冲突原因。
- `status_bar` 默认不参与一致性，因为它是弱设备先验，不是正文真值。
- 设备/系统信息只能作为可选先验；与正文冲突时送审，不能覆盖视觉证据。

## 数据集契约

数据根目录里放一个 JSONL 清单和区域图片。每行描述一张原始回单；同一原图、裁边变体、压缩变体和修图变体必须共用 `source_group_id`，并且只能出现在一个 split。

```json
{"schema_version":1,"kind":"receipt_font_domain_document_v1","id":"doc-0001","source_group_id":"source-0001","content_group_id":"content-0001","source_image_sha256":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa","split":"train","font_domain":"ios_alipay","label_source":"device-capture-log","device_prior_domain":"ios_alipay","regions":[{"id":"amount","role":"amount","image":"regions/doc-0001-amount.png","text":"-99.83"},{"id":"recipient","role":"recipient","image":"regions/doc-0001-recipient.png"},{"id":"time","role":"time","image":"regions/doc-0001-time.png"},{"id":"status","role":"status_bar","image":"regions/doc-0001-status.png","include_in_consistency":false}]}
```

示例中的 `source_image_sha256` 必须替换成原始整图的真实 SHA-256。训练清单使用 `train`、`calibration`、`test`；待分析清单使用 `inference` 且不带 `font_domain`/`label_source`。路径只能是清单目录内的 POSIX 相对路径。加载器会绑定同一份字节快照的原始字节 SHA-256、解码像素 SHA-256 和 pHash，并拒绝：

- `source_group_id` 或 `content_group_id` 跨 split；
- 同一解码图像跨 split；
- 图片路径逃逸、软链接逃逸、重复使用或哈希不符；
- 把 `unknown` 当训练类别；
- 非有限 JSON 数字、重复文档/区域 ID 和空参与区域。

`content_group_id` 用来绑定相同文字内容或同一合成模板，防止模型只记住文本。正式训练应让相同的金额、姓名和时间内容在各字体域中都有覆盖，并按原始来源分组切分。

## 数据采集建议

第一批不要追求很多字体名，先建立 2–4 个可靠的手机/应用渲染域：

1. 用可追溯的真实设备或模拟器截图，记录系统版本、应用版本、显示缩放、语言和截图/转发链。
2. 每个域使用相同的一组文本内容；金额、时间、中文和英文角色都要覆盖。
3. 每张图裁出金额、收款人、时间等 3 个以上正文区域；保留适度上下文，不能只裁一个减号。
4. JPEG/WebP、缩放和聊天软件转发版本与原图保持同一 `source_group_id`。
5. 按设备/原图来源切分 train/calibration/test，测试集保留未见设备和未见压缩链。
6. 字体文件和私人回单图片不提交 Git；仓库只保存程序、合约、合成测试与哈希证据。

## CLI 工作流

安装项目后可使用 `receipt-font-domain-consistency`；源码环境也可运行 `scripts/receipt-font-domain-consistency.py`。

```bash
# 1. 校验绑定和跨 split 泄漏；小数据集同时做 pHash 近重复审计
receipt-font-domain-consistency validate \
  --records /data/font-domain/train.jsonl

# 2. 训练 64 维经典特征 + 稳健 prototype 验证基线
receipt-font-domain-consistency fit \
  --records /data/font-domain/train.jsonl \
  --output /models/font-domain-prototype-v1.json

# 3. 独立分析待检区域，写入新目录
receipt-font-domain-consistency analyze \
  --model /models/font-domain-prototype-v1.json \
  --records /data/font-domain/inference.jsonl \
  --output /evidence/font-domain-run-001

# 4. 导出给后续 CNN/ONNX 训练器使用的扁平区域清单
receipt-font-domain-consistency export-classifier \
  --records /data/font-domain/train.jsonl \
  --output /data/font-domain/classifier.jsonl
```

训练校验、`fit` 和 `export-classifier` 默认要求 `content_group_id`、`source_image_sha256`，并运行跨 split pHash 启发式筛查（默认 64-bit Hamming 距离 8）。pHash 通过不等于证明无泄漏，仍需按来源审计。只有探索性实验可显式使用 `--allow-incomplete-leakage-metadata` 或 `--skip-near-duplicate-audit`；这些状态会写入模型自哈希字段。每个已知域默认至少要有 20 个独立 `calibration` source group，使默认 0.05 conformal 尾部确实有拒识分辨率；如果把尾部阈值设得更小，程序会自动提高最低独立校准组数，过小且无法在有界数据契约内分辨的阈值会被拒绝。`--allow-uncalibrated-model` 只保存研究模型，相关预测仍默认 `UNKNOWN`。

这里的模型自哈希用于发现文件被意外改写，并不是数字签名，也不能证明图片或模型真实可信。

`analyze` 默认拒绝缺少上述出版前置记录的模型。研究运行必须显式传 `--allow-experimental-model`，其 sidecar 会强制 `requires_manual_review: true` 并写入 `model_evidence.evaluation_status: "not_assessed"`。即使前置记录齐全，也仍需通过本文末尾的独立测试指标，不能称为生产可用。

导出的 classifier JSONL 必须写在原训练清单同一目录，以保持安全相对图片路径可直接解析。

模型文件、导出清单和分析目录默认禁止覆盖。`analyze` 产出：

- `font_domain.sidecar.jsonl`：每张图的整图决定和逐区域证据；
- `errors.jsonl`：成功运行时为空；输入或模型绑定失败时命令 fail closed；
- `run.json`：模型/清单 SHA-256、数量、决定分布和输出哈希。

## 当前基线和后续模型

v1 基线使用灰度/前景归一化后的 LBP、HOG、笔画宽度、形态学、连通域和边缘灰阶 64 维特征。它以训练集的稳健中位数/MAD 做缩放，学习通用域 prototype，并在角色样本足够时学习角色 prototype。calibration split 用于温度和域内距离校准；低质量、低支持、低 margin 或域外样本输出 `unknown`。

这个经典模型用于验证数据和指标是否真的可分，不是最终真实性模型。如果真实独立测试证明有效，可在不改变 sidecar 契约的前提下替换为轻量 CNN/ONNX：输出完整域概率、embedding 和校准后的 OOD 分数；不要复用普通 OCR 的 winner-only 置信度作为最终字体结论。当前导出的 classifier 清单会校验图片字节和解码像素绑定，但现有 OCR-lite checkpoint 格式不会持久化本实验的数据快照和泄漏审计状态，因此不能把该 checkpoint 当作字体域发布证据；启用 CNN 路径前必须补齐这层溯源。

## 上线前门禁

必须按文档级独立测试集报告，而不是只看训练准确率：

- 正常真实回单误送审率；
- 已知域整图 `PASS` 覆盖率；
- 跨域替换的 `REVIEW` 召回率；
- `UNKNOWN` 覆盖率和 coverage-risk 曲线；
- 各压缩链、字符高度、角色和设备版本分层结果；
- document bootstrap 95% 置信区间；
- exact/pHash 泄漏审计为零。

建议 PoC 的继续门槛是：真实正常图误送审率不高于 0.5%，信息充足样本中跨域替换召回率至少 90%，实际压缩链下降不超过 10 个百分点。达不到时应保留为研究 sidecar，不接生产判定。
