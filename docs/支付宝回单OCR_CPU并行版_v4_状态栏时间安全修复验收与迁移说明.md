# 支付宝回单 OCR CPU 并行版 v4 状态栏时间安全修复验收与迁移说明

> 本版本是隔离实验候选，只验证状态栏时间安全修复；不替换、不覆盖现有正式交付、v2 性能实验或 v3 状态栏实验。未经单独发布批准，不得切换生产流量。

## 1. 本次结论

同一批 200 张固定验收图片已在远程 Windows CPU 服务器完成实跑和逐图对比：

| 验收项 | 实测结果 |
| --- | ---: |
| 输入 / 写入 / 错误 | 200 / 200 / 0 |
| 与旧结果成功配对 | 200 / 200 |
| 除 `status_bar_time` 外的 13 个字段 | 200 张逐字段完全一致，0 差异 |
| 修复前非标准状态栏时间 | 10 条 |
| 修复后非标准状态栏时间 | 0 条 |
| 受控状态栏时间变化 | 11 条 |
| 构建阶段关键保护文件 | 9 个，比对前后 SHA-256/大小完全一致 |
| 逐图比较阶段保护文件 | 414 个，比对前后 SHA-256/大小完全一致 |
| CPU 自动计划 | 4 workers × 5 threads/worker |
| 批处理 wall | 125.635414 秒 |
| 聚合吞吐率 | 1.591907833 张/秒 |
| `1.0 张/秒`门槛 | 通过 |

本次验证证明：固定 200 张结果闭合、其他 13 个字段不变、状态栏时间格式约束通过，并达到批量聚合每秒 1 张的门槛。它不等于“单张延迟低于 1 秒”，也不构成对任意图片或任意 CPU 的通用准确率、速度承诺。

## 2. 修复内容

`status_bar_time` 的非空值只允许：

```text
HH:mm
HH:mm:ss
```

本版处理以下情况：

- 时间与电量文字、百分号或电池区域相邻时，只提取唯一可信的时钟。
- 全角数字和冒号转换为 ASCII，一位小时补零。
- 正文日期时间、电量伪时间、多个候选或位置不安全时输出 JSON `null`，不猜测。
- OCR 时钟前只有拉丁字母残片时拒绝该候选。例如原始 OCR `L0:00` 不再被整理成看似合法的 `00:00`。
- 原始 OCR、置信度、位置和拒绝规则仍保留在诊断结果中，便于人工复核。

### 11 条变化的人工复核

以下结论来自独立人工查看原图，不是比较 JSON 自动证明的识别准确率：

- 9 条是一位小时的正确格式化：`0:04/0:05/0:06/0:07` 转为 `00:04/00:05/00:06/00:07`。
- `blue-055`：人工看图判断原结果 `00:04` 来自通知/正文，不是实际状态栏时钟；新结果为 `null`。自动比较证据本身只证明字段由 `00:04` 变为 `null`。
- `blue-008`：人工看图时原图可见时间为 `00:01`，但本次 PP-OCR 原始文本为 `L0:00`。程序不能从错误 OCR 安全推断 `00:01`，因此新结果为 `null`，并记录规则 `top_band_clock_latin_prefix_rejected`。自动 rich JSON 证明的是原始 OCR、拒绝规则和安全空值；原图真值属于人工复核结论。

## 3. 版本与隔离路径

本次远程实测使用：

```text
源码 ZIP：receipt-pp-clsbatch-auto-statusbar-time-experiment-v4-src-20260828.zip
源码 ZIP SHA-256：4ad7bc55404741d3583f6b709fd0e4dcf886995b37e1fafb53a66520a578666f

源码解压目录：D:\f3-pp-clsbatch-auto-statusbar-time-experiment-v4-src-20260828-c
实验包目录：  D:\alipay-ai-data\experiments\pp-clsbatch-auto-statusbar-time-experiment-v4-c
固定输入目录：D:\alipay-ai-data\acceptance\pp-v4-r2-200-input-20260820-202315
本次结果目录：D:\alipay-ai-data\experiments\acceptance\pp-clsbatch-auto-statusbar-time-v4-resultset-c
```

源码 ZIP 共 44 个文件：43 个受控源码文件加 `SOURCE-MANIFEST.sha256`。为保持已验证归档不变，ZIP 内置的 `docs\DELIVERY-GUIDE-ZH-CN.md` 仍保留旧 v3 标题和路径，不可作为 v4 验收说明；v4 应以本文档为准。若以后重打源码 ZIP，必须更新内置说明并重新生成清单、重新构建和重新验收，不能原地修改当前 ZIP。

实验包配置明确为：

```text
release_status = experimental_candidate
supersedes_existing_delivery = false
production_approval_granted = false
```

若重跑，`-OutputDirectory` 必须使用尚不存在的全新目录。不要删除、覆盖或复用上述已验收结果。

## 4. 运行环境与模型

- Windows x64。
- Microsoft .NET 8 x64 Runtime。
- CPU-only；不需要 GPU、CUDA、cuDNN、Python、PaddlePaddle 或 PaddleOCR runtime。
- 包内通过 ONNX Runtime CPU 运行设备模型和 PP-OCR `det + cls + rec` 三个 ONNX 模型。
- 蓝图和白图不再分流，全部使用同一套全页 PP-OCR 和相同字段映射。
- `-Workers Auto` 会根据当前进程可见的有效逻辑处理器数自动规划并行度。

确认 .NET 8：

```powershell
dotnet --list-runtimes | Select-String 'Microsoft.NETCore.App 8\.'
```

## 5. 输入说明

批量入口递归扫描 `-InputDirectory`，支持 `.png`、`.jpg`、`.jpeg`、`.bmp`、`.webp`。

- 输入目录、实验包和输出目录必须彼此分离，不能相互嵌套。
- 批次运行期间不要增删、替换、修改或重命名输入图片。
- 输入目录及内容不要使用符号链接、junction 或其他 reparse point。
- 程序只读取输入；启动前、worker 前后会校验输入快照和 SHA-256。

## 6. 输出说明

成功的批量输出根目录包含：

- `batch-input-manifest.json`：确定性输入快照与哈希。
- `batch-manifest.json`：每张输入和结果 JSON 的严格对应关系。
- `batch-summary.json`：批次计数、耗时和吞吐摘要。
- `batch-report.json`：CPU 规划、完整性、worker、耗时和吞吐证据。
- `batch-errors.jsonl`：成功批次必须没有 JSONL 错误记录。
- `worker-input-lists\`：worker 分片清单。
- `workers\worker-XX\results\`：每张图片对应的结果 JSON。
- `workers\worker-XX\`：每个 worker 的 manifest、summary、日志和错误文件。

失败批次可能留下部分结果和 `batch-failure.json`，只能用于排障，不能作为成功产物。

## 7. 每张图片的 exact14 JSON

每张结果有且只有以下 14 个顶层字段，并保持此顺序：

| JSON key | 含义 | 取值 |
| --- | --- | --- |
| `input_image` | 输入图片绝对路径 | 非空字符串 |
| `device` | 设备 | `ios`、`android` 或 `uncertain` |
| `voucher_type` | 凭证类型 | 字符串或 `null` |
| `transfer_status` | 转账状态 | 字符串或 `null` |
| `amount` | 金额 | 字符串或 `null` |
| `recipient_name` | 收款方姓名 | 字符串或 `null` |
| `recipient_account` | 收款方账号 | 字符串或 `null` |
| `payer_name` | 付款方姓名 | 字符串或 `null` |
| `payer_account` | 付款方账号 | 字符串或 `null` |
| `transfer_time` | 转账时间 | 字符串或 `null` |
| `transfer_note` | 转账附言/备注 | 字符串或 `null` |
| `voucher_number` | 凭证编号/订单号 | 字符串或 `null` |
| `status_bar_time` | 手机状态栏时间 | `HH:mm`、`HH:mm:ss` 或 `null` |
| `payment_method` | 交易/付款方式 | 字符串或 `null` |

未识别字段保留 key，值为 JSON `null`；不会删除 key，也不会用空字符串代替。

## 8. 远程 Windows 批量验收命令

以下命令可直接在当前远程 Windows PowerShell 使用。每次自动创建带时间戳的新输出目录：

```powershell
$pkg = 'D:\alipay-ai-data\experiments\pp-clsbatch-auto-statusbar-time-experiment-v4-c'
$input = 'D:\alipay-ai-data\acceptance\pp-v4-r2-200-input-20260820-202315'
$output = 'D:\alipay-ai-data\experiments\acceptance\pp-statusbar-v4-' + (Get-Date -Format 'yyyyMMdd-HHmmss')

if (-not (Test-Path -LiteralPath $pkg -PathType Container)) {
    throw "实验包不存在：$pkg"
}
if (-not (Test-Path -LiteralPath $input -PathType Container)) {
    throw "输入目录不存在：$input"
}
if (Test-Path -LiteralPath $output) {
    throw "输出目录必须是全新目录：$output"
}

powershell.exe -NoLogo -NoProfile -NonInteractive -ExecutionPolicy Bypass `
  -File (Join-Path $pkg 'run-receipt-pp-batch-cpu.ps1') `
  -InputDirectory $input `
  -OutputDirectory $output `
  -Workers Auto `
  -MinimumThroughput 1.0

$rc = $LASTEXITCODE
if ($rc -ne 0) {
    throw "PP 批量验收失败 rc=$rc；请保留输出目录排障：$output"
}

"批量验收成功，结果目录：$output"
```

读取并严格检查报告：

```powershell
$report = Get-Content -LiteralPath (Join-Path $output 'batch-report.json') -Raw -Encoding UTF8 |
  ConvertFrom-Json

[pscustomobject]@{
    Status = $report.status
    Attempted = $report.attempted
    Written = $report.written
    Errors = $report.errors
    Skipped = $report.accounting.skipped
    ExactInputResultClosure = $report.accounting.exact_input_result_closure
    Workers = $report.worker_plan.workers
    ThreadsPerWorker = $report.worker_plan.threads_per_worker
    WallSeconds = $report.execution.batch_wall_seconds
    ImagesPerSecond = $report.throughput.written_images_per_wall_second
    ThroughputGatePassed = $report.throughput.gate_passed
} | Format-List

if ($report.status -ne 'completed' `
    -or $report.attempted -ne $report.written `
    -or $report.errors -ne 0 `
    -or $report.accounting.skipped -ne 0 `
    -or -not $report.accounting.exact_input_result_closure `
    -or -not $report.throughput.gate_passed) {
    throw '批次报告未通过严格验收'
}
```

## 9. 单图抽查命令

```powershell
$pkg = 'D:\alipay-ai-data\experiments\pp-clsbatch-auto-statusbar-time-experiment-v4-c'
$img = 'D:\download2\OtherImages\s3_voucher_GWCZ2071987206628708352_20260701000147.png'
$output = 'D:\alipay-ai-data\delivery-validation\pp-statusbar-v4-single-' + (Get-Date -Format 'yyyyMMdd-HHmmss')

if (-not (Test-Path -LiteralPath $img -PathType Leaf)) {
    throw "输入图片不存在：$img"
}
if (Test-Path -LiteralPath $output) {
    throw "输出目录必须是全新目录：$output"
}

powershell.exe -NoLogo -NoProfile -NonInteractive -ExecutionPolicy Bypass `
  -File (Join-Path $pkg 'run-receipt-pp-cpu.ps1') `
  -InputImage $img `
  -OutputDirectory $output

$rc = $LASTEXITCODE
if ($rc -ne 0) {
    throw "PP 单图抽查失败 rc=$rc"
}

"单图抽查成功，结果目录：$output"
```

单图成功后会在控制台打印设备和全部字段，并给出结果 JSON 路径。单图输出目录严格包含错误、manifest、summary 和 1 个结果 JSON，共 4 个文件。

## 10. 自动并行和性能解释

当前 24 个有效逻辑处理器主机自动得到：

```text
effective=24, reserved=2, usable=22
workers=4, threads/worker=5, allocated worker threads=20
```

`1.591907833 张/秒`是 4 个 worker 的聚合吞吐率。单张整体推理平均耗时为 `2429.368619 ms`，PP-OCR 阶段平均为 `2381.933246 ms`；多张图片并行处理，所以聚合产能仍超过 1 张/秒。

旧 v2 基线同批吞吐为 `1.731165493 张/秒`，本次为 `1.591907833 张/秒`，本次运行低约 `8.04%`，但通过 `1.0 张/秒`门槛。该差异同时出现在 PP-OCR 阶段，不能据此把状态栏映射规则解释为固定性能回退；也不能宣称 v4 比 v2 更快。换服务器后必须重新运行代表性批次。

## 11. 迁移步骤

1. 保持正式交付、v2、v3 和现有任务不变。
2. 将 `pp-clsbatch-auto-statusbar-time-experiment-v4-c` 整个目录复制到目标服务器的独立实验目录；不要拆分、重组或向包内增加文件。
3. 安装或确认 Microsoft .NET 8 x64 Runtime。
4. 准备包外输入目录和全新输出目录，先做单图抽查。
5. 用接近实际图片尺寸、比例和类型的代表性批次运行 `-Workers Auto`。
6. 核对退出码、`batch-report.json`、结果数量、空错误记录、exact14 和新机吞吐率。
7. 实验通过只形成候选结论；生产升级仍需单独批准。

## 12. 本次可复核证据

共享目录：

```text
\\tsclient\alipay-ai-inference-temp\router-calibration-20260814
```

关键文件：

```text
statusbar-time-v4-phase1-summary-c.json
statusbar-time-v4-batch-report-c.json
statusbar-time-v4-build-and-200-c.log
statusbar-time-v4-comparison-evidence-a.json
statusbar-time-v4-compare-a.log
statusbar-time-v4-blue008-rich-c.json
statusbar-time-v4-package-config-c.json
```

其中构建阶段摘要记录 9 个关键保护文件（正式包、v2/v3 证据和源码 ZIP）前后未变；逐图比较证据记录 200 对输入、其他 13 字段 0 差异、11 条状态栏时间变化、非标准非空时间 0 条，以及 414 个比较输入/结果/报告文件前后未变。

## 13. 验收检查清单

- [ ] 当前目录明确是隔离实验包，没有替换正式包。
- [ ] Windows x64 能看到 `Microsoft.NETCore.App 8.x`。
- [ ] 输入、包和输出目录彼此分离，输出目录启动前不存在。
- [ ] 批量性能验收使用 `run-receipt-pp-batch-cpu.ps1 -Workers Auto`，不是外部单图循环。
- [ ] 退出码为 0，报告为 `completed`，`attempted=written`，`errors=0`，`skipped=0`。
- [ ] `exact_input_result_closure=true`，错误 JSONL 没有错误记录。
- [ ] 每张结果严格符合 exact14，未识别字段保留为 `null`。
- [ ] 非空 `status_bar_time` 都是 ASCII `HH:mm` 或 `HH:mm:ss`。
- [ ] `throughput.gate_passed=true`，并明确这是批量聚合吞吐，不是单图小于 1 秒。
- [ ] CPU、虚拟化配额、磁盘或安全软件变化后，已在新环境重新验收。

---

本说明记录的是一次可复核的固定批次实验结果；不构成对任意硬件、任意图片集的通用性能或字段准确率承诺。
