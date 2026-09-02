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

## 负号边界

已有 Windows 实验证明，同一冻结 DET/CLS/REC 在 `det_db_unclip_ratio=1.5` 时读成 `99.83`，扩大到约 `2.4` 后恢复 `-99.83`。当前证据更支持检测框左边界裁掉负号，而不是字体识别失败。

“负号宽度 ÷ 邻近数字宽度”的真/改图判别需要真实/编辑真值；现有伪标签没有该真值，且现有金额裁图可能已把负号裁掉。本轮不把宽度比包装成准确率或真假结论。后续只能从原 result bbox 左扩后做独立几何实验。

## 结论边界

- 指标是 Paddle teacher parity，不是人工 OCR 准确率。
- 平台是字体/渲染代理，不是字体名称真值。
- 转发、缩放、压缩、设备分辨率和检测框差异仍可能参与信号。
- 不判断图片真假，不自动补负号，不修改冻结 v4 OCR 或生产路由。
