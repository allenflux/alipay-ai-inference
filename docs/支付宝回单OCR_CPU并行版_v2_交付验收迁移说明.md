# 支付宝回单 OCR CPU 并行版 v2 交付、验收与迁移说明

## 1. 交付版本

- 交付包名称：`pp-final-v4-r2-perf-v2`
- 建议部署目录：`D:\alipay-ai-data\delivery\pp-final-v4-r2-perf-v2`
- 生产批量入口：`run-receipt-pp-batch-cpu.ps1`
- 单图检查入口：`run-receipt-pp-cpu.ps1`
- 运行方式：Windows x64、CPU-only、.NET 8

本版本使用包内的 PP-OCR `det + cls + rec` ONNX 模型和设备识别模型，通过 .NET 8 与 ONNX Runtime CPU 执行。运行时不需要 Python、PaddlePaddle/PaddleOCR runtime、CUDA、cuDNN 或 GPU ONNX Runtime，也不使用 GPU。

> 生产处理必须使用批量并行入口。单图入口只用于抽查和问题定位，不用它代替批量处理。

## 2. 运行前提

1. Windows x64 服务器已安装 Microsoft .NET 8 x64 Runtime。
2. 执行账号对交付包和输入目录有读权限，对输出父目录有写权限。
3. 交付包、输入目录和输出目录必须彼此分离，不能相互嵌套。
4. 交付包必须整个目录保持原样；不要只拷贝脚本或模型，不要向包内增加日志、结果或其他文件。每次启动都会校验包内文件的完整性和 SHA-256 绑定。

在 PowerShell 中确认 .NET 8 Runtime：

```powershell
dotnet --list-runtimes | Select-String 'Microsoft.NETCore.App 8\.'
```

能看到 `Microsoft.NETCore.App 8.x.x` 即可继续。

## 3. 输入约定

批量入口会递归扫描 `-InputDirectory` 下的普通图片文件，支持：

- `.png`
- `.jpg`
- `.jpeg`
- `.bmp`
- `.webp`

程序启动时会对输入目录生成一次确定性快照，并记录每个输入文件的规范化路径、大小、最后修改时间和 SHA-256。

- 启动快照之后新增的图片不属于当前批次。
- 从开始到结束，不要增、删、改、替换或重命名已纳入快照的图片。
- 程序会在 worker 启动前和结束后重新校验输入；任何变化都会使批次失败关闭。
- 输入目录及其内容不应使用符号链接、junction 或其他 reparse point。
- 程序只读取和校验输入，不会改写输入图片。

## 4. 输出约定

`-OutputDirectory` 必须是一个尚不存在的全新目录。程序不会覆盖、清空或复用旧目录；重跑时请换一个新路径。

成功的批量输出根目录包含：

- `batch-input-manifest.json`：启动时输入快照与分片绑定。
- `batch-manifest.json`：输入与结果的严格对应记录。
- `batch-summary.json`：批次状态、计数、耗时和吞吐率摘要。
- `batch-report.json`：完整验收报告，包含 CPU 自动规划、输入完整性、worker、分阶段耗时和吞吐率证据。
- `batch-errors.jsonl`：错误记录；成功批次中必须为空文件。
- `worker-input-lists\`：确定性 worker 分片清单。
- `workers\worker-XX\results\`：每张输入图片对应的紧凑结果 JSON。
- `workers\worker-XX\`：worker 的摘要、manifest、空错误文件和日志证据。

如果批次失败，输出目录可能保留部分结果、诊断证据和 `batch-failure.json`。这些只能用于排障，不能当作成功产物。

## 5. exact14 结果 JSON

每张图片的结果 JSON 有且仅有下列 14 个顶层字段，顺序如下：

| JSON key | 含义 | 取值约定 |
| --- | --- | --- |
| `input_image` | 输入图片位置 | 规范化的 Windows 绝对路径，非空字符串 |
| `device` | 设备 | `ios`、`android` 或 `uncertain` |
| `voucher_type` | 凭证类型 | 非空字符串或 `null` |
| `transfer_status` | 转账状态 | 非空字符串或 `null` |
| `amount` | 金额 | 非空字符串或 `null` |
| `recipient_name` | 收款方姓名 | 非空字符串或 `null` |
| `recipient_account` | 收款方账号 | 非空字符串或 `null` |
| `payer_name` | 付款方姓名 | 非空字符串或 `null` |
| `payer_account` | 付款方账号 | 非空字符串或 `null` |
| `transfer_time` | 转账时间 | 非空字符串或 `null` |
| `transfer_note` | 转账附言/备注 | 非空字符串或 `null` |
| `voucher_number` | 凭证编号/订单号 | 非空字符串或 `null` |
| `status_bar_time` | 手机状态栏时间 | 非空字符串或 `null`，与转账时间分开 |
| `payment_method` | 交易/付款方式 | 非空字符串或 `null` |

未识别出的业务字段保留 key，值写为 JSON `null`，不会删掉 key，也不会用空字符串代替。结果不包含嵌套 `transaction_info`、中间 OCR 候选、判断或调试字段。

## 6. 未知 CPU 上的自动并行策略

客户无需预先知道服务器 CPU 型号或手工填线程数。使用 `-Workers Auto` 时，程序通过 `[Environment]::ProcessorCount` 获取当前进程可用的有效逻辑处理器数，再自动计算：

```text
reserved = max(1, min(2, floor(effective_processors / 8)))
# 极小 CPU 配额下仍至少留出 1 个处理器给 OCR：
if reserved >= effective_processors: reserved = max(0, effective_processors - 1)
usable = effective_processors - reserved
workers = min(4, max(1, floor(usable / 5)))
threads_per_worker = floor(usable / workers)
```

程序始终校验：

```text
workers * threads_per_worker <= effective_processors - reserved
```

在本次验证的 24 有效逻辑处理器主机上：

```text
effective=24, reserved=2, usable=22, workers=4, threads/worker=5
```

即 4 个独立 worker 并行，每个 worker 使用 5 个 OCR CPU 线程。不建议在未重新基准测试前手工固定 worker 数。

每个 .NET 8 worker 启动后还会回报自己的有效处理器数和实际 OCR 线程计划；父进程与 worker 口径不一致时，批次会失败关闭。因此，`Auto` 不等于对所有硬件拓扑作无条件性能承诺。超过 64 个逻辑处理器、跨 Windows processor group、受 affinity/Job Object/容器配额限制，或人为设置 `DOTNET_PROCESSOR_COUNT`、`COMPlus_ProcessorCount` 等覆盖变量的主机，应先清理非预期覆盖，并用本机代表性批次重新验收。本文的性能基线对应普通单 processor group 的 24 有效逻辑处理器 Windows x64 主机。

## 7. 批量生产/验收命令

先将待验收图片放入一个独立目录。下面命令默认使用本次远程电脑上保留的 200 张验收输入，并自动为每次执行生成新输出目录，可直接复制到 Windows PowerShell。迁移到其他电脑后，只需把 `$input` 换成客户的实际输入目录：

```powershell
$pkg = 'D:\alipay-ai-data\delivery\pp-final-v4-r2-perf-v2'
$input = 'D:\alipay-ai-data\acceptance\pp-v4-r2-200-input-20260820-202315'
$output = 'D:\alipay-ai-data\acceptance\pp-final-v4-r2-perf-v2-' + (Get-Date -Format 'yyyyMMdd-HHmmss')

if (-not (Test-Path -LiteralPath $pkg -PathType Container)) {
    throw "交付包不存在：$pkg"
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
    throw "PP 批量验收失败 rc=$rc，请保留输出目录用于排障：$output"
}

"批量验收成功，结果目录：$output"
```

`-MinimumThroughput 1.0` 是本次验收的吞吐率门槛。如果客户的正式生产环境在另一层系统中监控性能，可在完成该服务器的基准验证后省略此参数，或显式传入 `-MinimumThroughput 0`。

成功后可直接读取批次报告：

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

## 8. 单图抽查命令

单图命令可直接复制到本次远程电脑。迁移到其他电脑后，只需替换 `$img`：

```powershell
$pkg = 'D:\alipay-ai-data\delivery\pp-final-v4-r2-perf-v2'
$img = 'D:\download2\OtherImages\s3_voucher_GWCZ2071987206628708352_20260701000147.png'
$output = 'D:\alipay-ai-data\delivery-validation\pp-v2-single-' + (Get-Date -Format 'yyyyMMdd-HHmmss')

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

单图成功后会在控制台打印中文字段表和结果 JSON 路径。单图输出目录应严格包含 4 个文件：`inference_errors.jsonl`、`inference_manifest.json`、`inference_summary.json` 和 1 个结果 JSON。

## 9. 失败关闭规则

下列任一情况都会返回非 0 退出码，整个批次不算成功：

- 交付包缺文件、多文件，或者大小/SHA-256 绑定不一致。
- 输出目录已经存在，或包/输入/输出路径相互嵌套。
- 输入文件在运行期间发生变化。
- 任一 worker 启动、加载模型、推理或退出异常。
- 任一图片失败、遗漏、重复，或者结果 JSON 不符合 exact14 结构。
- 实际成功写入吞吐率低于 `-MinimumThroughput` 门槛。

成功的严格批次必须同时满足 `status=completed`、`attempted=written`、`errors=0`、`accounting.skipped=0`、`accounting.exact_input_result_closure=true`，以及启用吞吐率门槛时 `throughput.gate_passed=true`。

## 10. 本版 200 张验收结果

在固定的同一批 200 张验收图片（蓝图 100 张 + 白图 100 张）上，24 有效逻辑处理器主机使用 `-Workers Auto` 得到 4 worker × 5 threads/worker，结果如下：

| 指标 | v2 结果 |
| --- | ---: |
| 尝试 / 写入 / 错误 | 200 / 200 / 0 |
| exact14 与 r2 基线一致 | 200 / 200 |
| worker 批次 wall | 110.48873 s |
| 成功写入吞吐率 | 1.810139 张/s |
| r2 基线吞吐率 | 1.049016 张/s |
| 相对 r2 吞吐率提升 | 72.56% |
| PP-OCR 阶段平均耗时 | 2073.390738 ms/张 |
| 整体推理平均耗时 | 2119.15053 ms/张 |

`exact14 与 r2 基线一致 200/200` 表示这 200 张的用户可见 JSON 字段与 r2 基线逐图一致，用于确认本次性能优化没有改变该验收集的输出。这不等于对未标注数据声明或推导出字段识别“准确率”，本文档不虚构准确率数字。

### 性能指标的正确理解

1.810139 张/s 是多 worker 并行处理的聚合吞吐率，不是“每张图的单图延迟小于 1 秒”。本次整体推理平均耗时为 2119.15053 ms/张，但多张图在不同 worker 中并行运行，所以整体产能可超过 1 张/s。

上述数字只对当次验证的 CPU、固定 200 张数据、存储和并发设置成立。迁移到其他 CPU、虚拟化配额、磁盘或杀毒软件环境后，必须在新机器上用代表性批次重新验证吞吐率。

## 11. 整目录迁移步骤

1. 停止使用旧包的新任务，已在执行的批次应自然结束。
2. 将 `pp-final-v4-r2-perf-v2` 整个目录复制到新服务器；不要拆分、重组或向包内增加文件。
3. 在新服务器上确认 Microsoft .NET 8 x64 Runtime。
4. 准备包外的输入目录和全新输出目录，先执行单图抽查。
5. 使用与实际生产图片尺寸和比例接近的代表性批次，执行 `-Workers Auto` 批量验收。
6. 核对 `batch-report.json`、结果数、空错误文件、exact14 结构和新机吞吐率。
7. 通过后再将上游输入与下游 JSON 消费者切换到新目录。旧包保留一段回退期，但不要在新旧包之间复用同一输出目录。

## 12. 客户验收检查清单

- [ ] 交付目录名为 `pp-final-v4-r2-perf-v2`，且是完整目录复制。
- [ ] Windows x64 已安装 `Microsoft.NETCore.App 8.x`。
- [ ] 未安装 Python、Paddle runtime、CUDA 或 GPU 也可正常运行。
- [ ] 交付包、输入和输出目录彼此分离。
- [ ] 输出目录在启动前不存在。
- [ ] 批次运行期间没有改动输入目录和图片。
- [ ] 生产/性能验收使用 `run-receipt-pp-batch-cpu.ps1 -Workers Auto`，而非外部单图循环。
- [ ] 命令退出码为 0，终端显示 `PP CPU BATCH COMPLETE`。
- [ ] `batch-report.json` 中 `status=completed`、`attempted=written`、`errors=0`、`accounting.skipped=0`、`accounting.exact_input_result_closure=true`。
- [ ] 启用吞吐率门槛时，`throughput.gate_passed=true`。
- [ ] `batch-errors.jsonl` 为空文件，没有 `batch-failure.json`。
- [ ] 结果数量与启动快照输入数量一致，每个结果都符合 exact14。
- [ ] 业务系统能正确接受可空业务字段的 JSON `null`。
- [ ] 已明确“超过 1 张/s”是批量聚合吞吐率，不是单图延迟小于 1 秒。
- [ ] 如果 CPU、虚拟化配额或存储环境变更，已在新环境重新执行代表性批量验收。

---

文档中的验收数字是一次可复核的固定批次性能证据，不构成对任意硬件、任意图片集的通用性能或准确率承诺。
