# 按系统/字体路由 OCR 验证 v1

## 要回答的问题

本实验验证：在相同字段、相同可见内容/格式、相同 source-group 安全切分下，把 OCR 分为 iOS 与 Android 两个专家，是否比通用模型更接近冻结的 Paddle 教师标签。

它对应“按字体分开训练”的快速验证，但当前没有字体文件真值或人工业务真值，因此两个专家只能称为**平台代理 OCR 专家**，不能称为某个具体字体模型。

`time` 是顶部状态栏时钟，单独参与本实验；它继续不进入支付宝正文字体一致性模型。`payment_method_field` 是优先字段，`amount` 是保护字段。

## 防止假阳性的对照

每个字段独立训练五个同结构模型：

1. `global`：两个平台的共同样本；
2. `ios`：iOS 平台代理样本；
3. `android`：Android 平台代理样本；
4. `random_a`、`random_b`：按 `group_id` 哈希随机分成的两个专家。

通用模型、平台专家和随机专家使用同一个 seed。随机双专家用于排除“只是模型数量翻倍/容量增加”的收益；还会把 iOS 测试集送入 Android 专家、把 Android 测试集送入 iOS 专家，形成 wrong-route 对照。

主实验只接受：

- `device.source == "resolution"`；
- `device.platform` 为 `ios` 或 `android`；
- `device.confidence >= 0.90`；
- `device_prior_conflict == false`。

不使用状态栏 CNN 触发主实验路由，因为 CNN 会读取顶部时间像素，拿它再证明 `time` 的系统字体收益可能形成循环验证。CNN、冲突、低置信、未知域在未来实际路由中必须回退 `global`。

## 数据控制

输入直接引用既有 `pseudo_labels.jsonl` 裁图，不复制图片。程序重新按 `group_id` 为本次从零训练的模型生成 train/val/test；同一 group 永不跨 split。当前代码还强校验 source 只属于一个 group、裁图路径不重复。

平台两边在每个 `field × split` 内按下列内容 strata 等量选择：

- `time`：严格相同的可见时间文本；
- `payment_method_field`：相同语义值；
- `amount`：相同正负号、币符、千分位、整数位数和小数位数格式。

这样平台专家不能仅依靠平台间不同的标签频率取得虚假优势。当前尚未运行近重复像素审计，报告会明确记录 `near_duplicate_pixel_audit: not_assessed`，因此结果不可发布。

## 初步通过条件

按 source group 做 2,000 次 bootstrap，并同时报告配对 McNemar 检验。方向性支持需要：

- 至少 200 个配对测试记录；少于 1,000/平台/字段标记为 underpowered；
- `time`、`payment_method_field` 的 semantic exact 至少提升 2 个百分点；`amount` 单字段支持门槛为 0.5 个百分点；
- group-bootstrap 95% 区间下界大于 0；
- 平台专家相对随机双专家至少多提升 1 个百分点；
- 正确专家相对错误平台专家至少提升 1 个百分点；
- 任一平台不得回退超过 0.5 个百分点；
- 最终方向性结论要求 `time` 或 `payment_method_field` 通过，且 `amount` 不得回退超过 1 个百分点。

即便通过，也只表示“值得制作人工真值集继续验证”，不是生产发布结论。

## Windows 隔离运行

PowerShell runner 使用独立的时间戳目录，拒绝覆盖旧输出：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File `
  \\tsclient\alipay-ai-inference-temp\font-routed-ocr-windows-pilot.ps1 `
  -SourceArchive \\tsclient\alipay-ai-inference-temp\<exact-commit>.zip `
  -RunId font-routed-ocr-<timestamp>-<commit> `
  -Epochs 5
```

默认读取：

```text
D:\alipay-ai-data\receipt-lite-teacher-120k-v1\paddle-teacher-labels-5field-recipient95-v12-r3-4090-r1\pseudo_labels.jsonl
```

输出只写入：

```text
D:\alipay-ai-data\experiments\font-routed-ocr-validation-v1\<RunId>
```

Mac 共享目录只回收 `prepare.json`、`runtime-evidence.json`、最终 A/B summary 和状态 JSON，
不回收图片或 checkpoint。

PyTorch 使用 CUDA 训练；ONNX A/B 评估明确使用 CPU provider。评估正确性不依赖 ONNX Runtime
CUDA DLL，避免把环境加速问题混入本轮字体方向判断。runner 会验证全部 30 个字段/对照评估
都只激活 `CPUExecutionProvider`，并把每个模型、输入清单、evaluation summary 和 comparisons 的
SHA-256 写入随最终结果一起发布的 `runtime-evidence.json`；最终 A/B summary 再绑定该证据文件。

如果训练阶段因 runner/环境问题中止，而上一个独立 RunId 已经完整生成
`prepared-resolution-primary`，可在新 RunId 中用 `-PreparedInput` 复用该只读清单。runner
仍会重新校验 resolution-only 契约、输入记录路径和每个平台/字段测试样本量，不覆盖旧目录：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File `
  \\tsclient\alipay-ai-inference-temp\font-routed-ocr-windows-pilot.ps1 `
  -SourceArchive \\tsclient\alipay-ai-inference-temp\<exact-commit>.zip `
  -RunId font-routed-ocr-<new-timestamp>-<commit> `
  -PreparedInput D:\alipay-ai-data\experiments\font-routed-ocr-validation-v1\<old-run>\prepared-resolution-primary `
  -Epochs 5
```

模型产物不会跨 RunId 复用：现有 checkpoint 没有绑定训练 manifest 的哈希，直接复制会削弱
A/B 实验的可审计性。续跑只复用经过契约校验的准备清单，所有模型仍按相同参数重新训练。

## 2026-09-02 Windows 实测结果

独立运行 `font-routed-ocr-20260902-020425-55a05690-r3` 已完整成功，源提交为 `55a05690`。
最终门禁是 `not_supported_in_pilot`，`supported_fields` 为空，金额护栏未通过。下表是
semantic exact 的配对结果；百分点变化均为平台代理专家相对同字段 global 模型：

| 字段 | 每平台测试数 | global | 正确路由专家 | 变化 | 95% group-bootstrap | 随机双专家变化 | 平台超额收益 | 正确路由相对错路由 | 支持 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `amount` | 549 | 4.83% | 0.00% | -4.83 pp | [-6.19, -3.64] pp | -4.83 pp | 0.00 pp | 0.00 pp | 否 |
| `time` | 436 | 0.00% | 0.00% | 0.00 pp | [0.00, 0.00] pp | 0.00 pp | 0.00 pp | 0.00 pp | 否 |
| `payment_method_field` | 1,500 | 94.90% | 2.40% | -92.50 pp | [-93.47, -91.50] pp | -49.70 pp | -42.80 pp | 0.00 pp | 否 |

证据审计同时通过：共 31,346 条选中记录、27,653 个 source group；时间路由只用
`device.source=resolution`，没有读取时间字形像素；15 个模型和 30 个评估全部完成；30/30
evaluation summary 都只激活 `CPUExecutionProvider`。最终 summary 绑定的 runtime evidence
SHA-256 为 `4e7e39bb4b2a1a163103d0a62f80dc57d6bf64b98f1547efcdcdced3cfe7e5ed`。
共享副本是哈希收据而不是可离线重算的完整 bundle：ONNX、逐评估 summary/comparisons 和 merged
JSONL 仍保留在该 RunId 的 Windows D 盘目录；离开该目录只能核对收据，不能重算全部指标。

这次结果否定的是“把当前 CTC 按平台各自从头训练 5 epochs 就能受益”，不是证明 iOS 与 Android
字体相同。`payment_method_field` 的 global 已有较强教师一致性，但平台代理专家严重退化；正确
路由与错路由又完全无差异，说明当前专家没有学到可用的平台特异信号。`amount` 和 `time` 的
global 基线本身过低，无法承担字体效应验证。这些退化值也可能包含训练配方、decoder 或评估
适配问题，不能外推成“平台路由天然有害”。不得据此拆分生产 OCR。

如果继续验证，下一次应先保住可用 global 基线，再从同一 global checkpoint 对 iOS/Android
做低学习率域微调，并保留本轮随机专家、错路由和金额护栏；在该门槛通过前不值得人工标大量
字体标签。即便后续教师一致性通过，生产准确率仍需要冻结的人工真值测试集。

## 负号边界

已有 Windows 实验证明，同一冻结 DET/CLS/REC 在 `det_db_unclip_ratio=1.5` 时读成 `99.83`，扩大到约 `2.4` 后恢复 `-99.83`。当前证据更支持检测框左边界裁掉负号，而不是字体识别失败。

“负号宽度 ÷ 邻近数字宽度”的真/改图判别需要真实/编辑真值；现有伪标签没有该真值，且现有金额裁图可能已把负号裁掉。本轮不把宽度比包装成准确率或真假结论。后续只能从原 result bbox 左扩后做独立几何实验。

## 结论边界

- 指标是 Paddle teacher parity，不是人工 OCR 准确率。
- 平台是字体/渲染代理，不是字体名称真值。
- 转发、缩放、压缩、设备分辨率和检测框差异仍可能参与信号。
- 不判断图片真假，不自动补负号，不修改冻结 v4 OCR 或生产路由。
