# PP-OCR CPU v4 客户交付、验收与迁移指南

> 适用版本：`receipt_full_page_ppocr_cpu_delivery_v4`  
> 运行方式：Windows x64、纯 CPU、批量并行  
> 发布状态：客户验收版；按本文完成远程验收并保存报告后转为正式交付

## 1. 交付概述

PP-OCR CPU v4 是一套无需 GPU 的通用中文 OCR 交付包。蓝色凭证、白色凭证及其他支持格式的图片均走同一套全页识别流程，不需要先判断“蓝图/白图”。

交付包配置以下 4 个 ONNX 模型；设备模型和文本检测模型按图片运行，方向分类与文本识别模型对检测出的文本区域运行：

1. 状态栏设备识别模型；
2. PP-OCR 文本检测模型（det）；
3. PP-OCR 方向分类模型（cls）；
4. PP-OCR 文本识别模型（rec）。

正式交付目录建议统一命名为：

```text
D:\alipay-ai-data\delivery\pp-final-v4
```

交付包包含两个可用入口，其中批量脚本是唯一正式生产入口：

| 使用场景 | 正式入口 | 说明 |
|---|---|---|
| 单张图片 | `run-receipt-pp-cpu.ps1` | 验收、人工抽查和故障定位入口 |
| 批量并行 | `run-receipt-pp-batch-cpu.ps1` | 正式生产入口；自动识别 CPU 并并行处理 |

批量模式中，每个 worker 只加载一次设备模型和三套 PP-OCR Session，并在该 worker 的全部图片之间复用；不会每张图片重新加载模型。

## 2. 交付验收目标

客户验收需要同时满足以下条件：

- 最终包完整性检查通过；
- 运行环境为 CPU-only；
- 结果 JSON 固定为 14 个顶层字段；
- 批量输入数量、成功数量和结果数量完全一致；
- `errors = 0`；
- 批量并行聚合吞吐量不低于 `1.0 张/秒`；
- 保存整个批量输出目录作为验收证据，其中 `batch-report.json` 是汇总索引。

> **重要口径：**“每秒 1 张”指多个 worker 并行后的整体吞吐量，不代表任意单张图片的响应时间一定小于 1 秒。正式验收以 `written / worker-batch wall` 的未舍入值为准。

## 3. 运行环境

### 3.1 必需环境

- Windows Server 或 Windows 10/11 x64；
- Windows PowerShell 5.1 或更高版本；
- .NET 8 x64 Runtime；
- 普通本地磁盘或稳定的高速数据盘；
- 对交付目录具有读取权限，对输出目录具有创建和写入权限。

检查 .NET 8：

```powershell
dotnet --list-runtimes
```

输出中必须存在类似内容：

```text
Microsoft.NETCore.App 8.x.x
```

### 3.2 不需要的环境

运行时不需要：

- GPU；
- CUDA、cuDNN；
- Python；
- PaddlePaddle；
- PaddleOCR Python 运行时；
- GPU 版本 ONNX Runtime。

交付包使用 PP-OCR 的 ONNX 模型，但推理运行时是 `.NET 8 + ONNX Runtime CPU`。

## 4. 输入说明

### 4.1 支持的图片格式

```text
.png
.jpg
.jpeg
.bmp
.webp
```

### 4.2 单图输入

单图入口接收一个图片文件绝对路径。

### 4.3 批量输入

批量入口接收一个目录，并递归扫描目录中的支持格式图片。

批量输入规则：

- 输入目录必须真实存在；
- 交付包目录、输入目录和输出目录必须两两分离；三者不能相同，也不能互相嵌套；
- 输出目录必须是尚不存在的全新路径；
- 输入目录及子目录不能包含符号链接、junction 或其他 reparse point；
- 启动时对目录成员做一次固定快照；运行开始后新增的文件不属于本批；
- 已进入本批的文件在运行期间不得修改、替换或删除；
- 程序会在初始扫描、worker 启动前、worker 退出后复核文件路径、大小、修改时间和 SHA-256；
- 检测到输入变化时，整批失败并保留诊断证据。

为避免误判，正式运行期间应冻结输入目录，不要让上传程序继续向同一目录写入文件。

## 5. 结果 JSON 说明

每张图片生成一份结果 JSON，固定包含以下 14 个顶层字段，名称和顺序均受程序校验：

| JSON 字段 | 中文含义 | 值类型 |
|---|---|---|
| `input_image` | 输入图片绝对路径 | 非空字符串 |
| `device` | 设备类型 | `ios`、`android`、`uncertain` |
| `voucher_type` | 凭证类型 | 字符串或 `null` |
| `transfer_status` | 转账状态 | 字符串或 `null` |
| `amount` | 金额 | 字符串或 `null` |
| `recipient_name` | 收款方姓名 | 字符串或 `null` |
| `recipient_account` | 收款方账号 | 字符串或 `null` |
| `payer_name` | 付款方姓名 | 字符串或 `null` |
| `payer_account` | 付款方账号 | 字符串或 `null` |
| `transfer_time` | 转账时间 | 字符串或 `null` |
| `transfer_note` | 转账附言 | 字符串或 `null` |
| `voucher_number` | 凭证编号 | 字符串或 `null` |
| `status_bar_time` | 手机状态栏时间 | 字符串或 `null` |
| `payment_method` | 交易方式 | 字符串或 `null` |

无法由当前映射规则形成值时，该字段输出 JSON `null`。非空值仍属于 review-only OCR/规则结果，可能发生误识别；客户应按自身业务风险要求进行抽查或复核。

示例：

```json
{
  "input_image": "D:\\input\\receipt-001.jpg",
  "device": "android",
  "voucher_type": "转账凭证",
  "transfer_status": "转账成功",
  "amount": "￥99.97",
  "recipient_name": "宾森(**森)",
  "recipient_account": null,
  "payer_name": null,
  "payer_account": null,
  "transfer_time": null,
  "transfer_note": null,
  "voucher_number": null,
  "status_bar_time": "00:01:09",
  "payment_method": "余额宝(转出资金付款)"
}
```

注意：`input_image` 会记录服务器上的绝对路径。如需向外部传输结果，应先确认该路径信息符合客户的数据安全要求。

## 6. 输出产物说明

### 6.1 单图输出

单图成功时，输出目录严格包含 4 个文件：

```text
inference_errors.jsonl
inference_manifest.json
inference_summary.json
<单图结果>.json
```

### 6.2 批量输出

批量输出目录主要包含：

```text
batch-input-manifest.json     # 本批输入快照和 worker 分配
batch-errors.jsonl            # 批次级保留文件；成功批次为空
batch-manifest.json           # 每张输入与结果文件的映射
batch-summary.json            # 数量、耗时、吞吐量和验收门结果
batch-report.json             # 客户验收汇总报告
worker-input-lists\           # 每个 worker 的固定输入清单
workers\worker-XX\           # 每个 worker 的结果、日志和报告
```

单张图片的实际错误记录位于：

```text
workers\worker-XX\results\inference_errors.jsonl
```

排障时还应优先查看对应 worker 的 `stdout.log` 和 `stderr.log`，不能只查看顶层 `batch-errors.jsonl`。`worker-report.json` 仅在该 worker 已完成输出验收时存在；worker 提前异常退出时可能没有该文件。

批量成功必须满足：

```text
attempted = written
errors = 0
throughput.gate_passed = true   # 使用吞吐门时
```

任一图片、进程、结果结构、数量、输入文件、交付包或吞吐门异常，批量入口均以非零退出码失败关闭，不会伪装成成功。进入正式批处理阶段后的失败会尽力写入 `batch-failure.json` 并保留已有日志；若在输出目录创建前的预检阶段失败，则可能没有 `batch-failure.json`，应以控制台错误和退出码为准。

## 7. 远程电脑验收步骤

以下命令均在客户远程 Windows PowerShell 中执行。

### 7.1 设置交付包路径

```powershell
$pkg = 'D:\alipay-ai-data\delivery\pp-final-v4'

if (-not (Test-Path -LiteralPath $pkg -PathType Container)) {
    throw "找不到交付包：$pkg"
}

dotnet --list-runtimes
```

不要在 `$pkg` 内新增说明文件、日志或结果目录，否则严格包完整性检查会拒绝运行。

### 7.2 单图功能验收

```powershell
$pkg = 'D:\alipay-ai-data\delivery\pp-final-v4'
$img = 'D:\download2\BlueImages\s3_voucher_GWCZ2071987126068711424_20260701000133.jpg'
$out = 'D:\alipay-ai-data\delivery-validation\pp-v4-single-a'

if (Test-Path -LiteralPath $out) {
    throw "输出目录必须是全新目录：$out"
}

powershell.exe -NoLogo -NoProfile -NonInteractive -ExecutionPolicy Bypass `
  -File (Join-Path $pkg 'run-receipt-pp-cpu.ps1') `
  -InputImage $img `
  -OutputDirectory $out

if ($LASTEXITCODE -ne 0) {
    throw "单图验收失败 rc=$LASTEXITCODE"
}
```

验收要求：

- 命令退出码为 0；
- 控制台打印设备类型和交易字段；
- 输出目录只有 4 个文件；
- 结果 JSON 恰好包含 14 个字段；
- `device` 为 `ios`、`android` 或 `uncertain`。

### 7.3 准备 100 张蓝图 + 100 张白图

以下命令会把两类图片各选取 100 张，复制到一个全新的验收输入目录。原始图片不会被修改。

```powershell
$blueRoot = 'D:\download2\BlueImages'
$whiteRoot = 'D:\download2\OtherImages'
$acceptInput = 'D:\alipay-ai-data\acceptance\pp-v4-200-input-a'
$extensions = @('.png', '.jpg', '.jpeg', '.bmp', '.webp')

if (Test-Path -LiteralPath $acceptInput) {
    throw "验收输入目录必须是全新目录：$acceptInput"
}
if (-not (Test-Path -LiteralPath $blueRoot -PathType Container)) {
    throw "找不到蓝图目录：$blueRoot"
}
if (-not (Test-Path -LiteralPath $whiteRoot -PathType Container)) {
    throw "找不到白图目录：$whiteRoot"
}

$blue = @(
    Get-ChildItem -LiteralPath $blueRoot -Recurse -File -Force |
        Where-Object { $extensions -contains $_.Extension.ToLowerInvariant() } |
        Sort-Object FullName |
        Select-Object -First 100
)
$white = @(
    Get-ChildItem -LiteralPath $whiteRoot -Recurse -File -Force |
        Where-Object { $extensions -contains $_.Extension.ToLowerInvariant() } |
        Sort-Object FullName |
        Select-Object -First 100
)

if ($blue.Count -ne 100) { throw "蓝图不足100张，实际=$($blue.Count)" }
if ($white.Count -ne 100) { throw "白图不足100张，实际=$($white.Count)" }

$null = New-Item -ItemType Directory -Path $acceptInput
$blueOut = New-Item -ItemType Directory -Path (Join-Path $acceptInput 'blue')
$whiteOut = New-Item -ItemType Directory -Path (Join-Path $acceptInput 'white')

for ($i = 0; $i -lt 100; $i++) {
    $name = 'blue-{0:D3}{1}' -f ($i + 1), $blue[$i].Extension.ToLowerInvariant()
    Copy-Item -LiteralPath $blue[$i].FullName -Destination (Join-Path $blueOut.FullName $name)
}
for ($i = 0; $i -lt 100; $i++) {
    $name = 'white-{0:D3}{1}' -f ($i + 1), $white[$i].Extension.ToLowerInvariant()
    Copy-Item -LiteralPath $white[$i].FullName -Destination (Join-Path $whiteOut.FullName $name)
}

$actual = @(
    Get-ChildItem -LiteralPath $acceptInput -Recurse -File -Force |
        Where-Object { $extensions -contains $_.Extension.ToLowerInvariant() }
).Count

if ($actual -ne 200) { throw "验收输入不是200张，实际=$actual" }
'PP_V4_ACCEPTANCE_INPUT_200_OK'
```

该命令按完整路径排序后固定选取前 100 张，因此同一目录内容不变时可以复现同一验收样本；它用于功能和性能验收，不代表随机抽样，也不能单独作为整体识别准确率结论。

### 7.4 批量并行性能验收

```powershell
$pkg = 'D:\alipay-ai-data\delivery\pp-final-v4'
$acceptInput = 'D:\alipay-ai-data\acceptance\pp-v4-200-input-a'
$acceptOutput = 'D:\alipay-ai-data\acceptance\pp-v4-200-output-a'

if (Test-Path -LiteralPath $acceptOutput) {
    throw "验收输出目录必须是全新目录：$acceptOutput"
}

powershell.exe -NoLogo -NoProfile -NonInteractive -ExecutionPolicy Bypass `
  -File (Join-Path $pkg 'run-receipt-pp-batch-cpu.ps1') `
  -InputDirectory $acceptInput `
  -OutputDirectory $acceptOutput `
  -Workers Auto `
  -MinimumThroughput 1.0

if ($LASTEXITCODE -ne 0) {
    throw "批量验收失败 rc=$LASTEXITCODE"
}

$summary = Get-Content -LiteralPath (Join-Path $acceptOutput 'batch-summary.json') -Raw -Encoding UTF8 |
    ConvertFrom-Json

$requiredEvidence = @(
    'batch-input-manifest.json',
    'batch-report.json',
    'batch-manifest.json',
    'batch-errors.jsonl'
)
foreach ($name in $requiredEvidence) {
    if (-not (Test-Path -LiteralPath (Join-Path $acceptOutput $name) -PathType Leaf)) {
        throw "验收证据缺失：$name"
    }
}
foreach ($name in @('worker-input-lists', 'workers')) {
    if (-not (Test-Path -LiteralPath (Join-Path $acceptOutput $name) -PathType Container)) {
        throw "验收证据目录缺失：$name"
    }
}

if ([string]$summary.status -ne 'completed' `
    -or [int]$summary.attempted -ne 200 `
    -or [int]$summary.written -ne 200 `
    -or [int]$summary.errors -ne 0 `
    -or -not [bool]$summary.throughput.gate_passed `
    -or [double]$summary.throughput.written_images_per_wall_second_unrounded -lt 1.0) {
    $summary | ConvertTo-Json -Depth 20
    throw '批量验收报告未达到200张、零错误、每秒1张的要求'
}

$summary | ConvertTo-Json -Depth 20
'PP_V4_CUSTOMER_ACCEPTANCE_PASS'
```

验收完成后必须保存整个 `$acceptOutput` 目录，不能只复制顶层报告。完整目录包含输入快照、worker 固定清单、逐图结果、worker 报告和原始日志，是复核报告的证据闭包。至少应确认以下内容存在：

```text
batch-input-manifest.json
batch-report.json
batch-summary.json
batch-manifest.json
batch-errors.jsonl
worker-input-lists\
workers\
```

建议验收后将整个目录压缩并记录 SHA-256：

```powershell
$acceptOutput = 'D:\alipay-ai-data\acceptance\pp-v4-200-output-a'
$acceptEvidenceZip = 'D:\alipay-ai-data\acceptance\pp-v4-200-output-a.zip'

if (Test-Path -LiteralPath $acceptEvidenceZip) {
    throw "验收证据压缩包已存在：$acceptEvidenceZip"
}

Compress-Archive -LiteralPath $acceptOutput -DestinationPath $acceptEvidenceZip -CompressionLevel Optimal
Get-FileHash -LiteralPath $acceptEvidenceZip -Algorithm SHA256 | Format-List
```

## 8. 正式批量生产命令

正式批量生产建议继续保留吞吐门：

```powershell
$pkg = 'D:\alipay-ai-data\delivery\pp-final-v4'
$input = 'D:\customer-data\receipt-input-20260818'
$output = 'D:\customer-data\receipt-output-20260818-a'

powershell.exe -NoLogo -NoProfile -NonInteractive -ExecutionPolicy Bypass `
  -File (Join-Path $pkg 'run-receipt-pp-batch-cpu.ps1') `
  -InputDirectory $input `
  -OutputDirectory $output `
  -Workers Auto `
  -MinimumThroughput 1.0

if ($LASTEXITCODE -ne 0) {
    throw "正式批处理失败 rc=$LASTEXITCODE"
}
```

`Workers Auto` 会读取当前服务器可用处理器数量，并自动分配最多 4 个 worker。迁移到不同 CPU 的服务器后，不应沿用旧服务器的固定线程数。

`-MinimumThroughput 1.0` 是服务器级批量 SLA 验收门：聚合吞吐低于 1.0 张/秒时，即使图片均已生成，整批仍返回非零退出码。正式交付建议保留该值；只有吞吐由外部调度系统单独监控并获得客户书面确认时，才可将该参数设为 `0`，此时程序只报告吞吐而不以速度判失败。

## 9. 迁移到新服务器

### 9.1 源服务器打包

迁移前确保没有 OCR 任务正在运行，然后执行：

```powershell
$sourcePackage = 'D:\alipay-ai-data\delivery\pp-final-v4'
$zip = 'D:\alipay-ai-data\delivery\pp-final-v4.zip'

if (-not (Test-Path -LiteralPath $sourcePackage -PathType Container)) {
    throw "找不到源交付包：$sourcePackage"
}
if (Test-Path -LiteralPath $zip) {
    throw "压缩包已经存在，请使用新文件名：$zip"
}

Compress-Archive -LiteralPath $sourcePackage -DestinationPath $zip -CompressionLevel Optimal
Get-FileHash -LiteralPath $zip -Algorithm SHA256 | Format-List
```

将生成的 ZIP 和 SHA-256 通过不同渠道交给目标服务器管理员。

### 9.2 目标服务器恢复

```powershell
$zip = 'D:\transfer\pp-final-v4.zip'
$expectedSha256 = '由交付方提供的64位SHA256'
$deliveryParent = 'D:\alipay-ai-data\delivery'
$targetPackage = Join-Path $deliveryParent 'pp-final-v4'

if ($expectedSha256 -notmatch '^[0-9a-fA-F]{64}$') {
    throw 'expectedSha256 必须是交付方提供的64位十六进制SHA256'
}

$actualSha256 = (Get-FileHash -LiteralPath $zip -Algorithm SHA256).Hash.ToLowerInvariant()
if ($actualSha256 -ne $expectedSha256.ToLowerInvariant()) {
    throw "迁移包SHA256不一致：$actualSha256"
}
if (Test-Path -LiteralPath $targetPackage) {
    throw "目标目录必须不存在：$targetPackage"
}

Expand-Archive -LiteralPath $zip -DestinationPath $deliveryParent

if (-not (Test-Path -LiteralPath $targetPackage -PathType Container)) {
    throw "迁移后找不到交付包：$targetPackage"
}

dotnet --list-runtimes
'PP_V4_MIGRATION_EXTRACT_OK'
```

迁移后必须在新服务器重新执行第 7.2 节单图验收和第 7.4 节 200 张批量验收。不同 CPU、磁盘和系统负载下的性能必须以新服务器生成的 `batch-report.json` 为准。

## 10. 完整性与安全边界

- `SHA256SUMS.json` 会约束最终包内部文件和目录；
- 单图和批量入口都会在运行前后复核交付包；
- 不要在交付包目录内新增、删除或替换任何文件；
- 输入目录在任务运行期间必须冻结；
- 包内 SHA-256 是完整性清单，不是数字签名；对外交付时仍需提供 ZIP 的外部 SHA-256；
- 字段输出属于 review-only OCR 结果，不代表 builder 已对每个业务字段作人工准确率认证；
- `status_bar_time` 只是手机状态栏时间，不能当作 `transfer_time`；
- 聚合吞吐量达到 1 张/秒，不代表单张延迟低于 1 秒；
- 极端的恶意瞬时替换并完全恢复不属于普通迁移和验收场景保证范围。

## 11. 常见问题

### 11.1 提示输出目录已存在

所有正式输出目录必须全新。不要删除旧证据后复用原路径，改用新的 `-a`、`-b` 或时间戳后缀。

### 11.2 提示缺少 .NET 8

安装 Microsoft .NET 8 x64 Runtime，然后重新打开 PowerShell 验证。

### 11.3 吞吐量低于 1 张/秒

依次检查：

1. 是否使用 `run-receipt-pp-batch-cpu.ps1`，而不是循环调用单图脚本；
2. 是否使用 `-Workers Auto`；
3. 服务器是否同时运行其他高 CPU 任务；
4. 杀毒软件是否实时扫描模型和大量结果文件；
5. 输入和输出是否位于速度过慢的网络盘；
6. CPU 是否被虚拟机配额、亲和性或节能策略限制；
7. 查看 `batch-report.json` 中的 worker/thread 计划、stage p50/p95 和实际吞吐量。

### 11.4 提示输入文件发生变化

停止向输入目录上传或改写文件，重新建立一个冻结的输入目录，并使用全新的输出目录重新运行。

### 11.5 批量任务中一张图片损坏

正式批量入口默认失败关闭：任一图片失败则整批返回非零退出码。优先查看对应 `workers\worker-XX\results\inference_errors.jsonl`、`workers\worker-XX\stderr.log` 和 `stdout.log`；`worker-report.json` 仅在该 worker 已完成输出验收时存在。进入正式批处理阶段后的失败通常还会留下 `batch-failure.json`。若在输出目录创建前预检即失败，可能没有该文件，应直接依据控制台错误和非零退出码处理。修复或隔离损坏图片后，必须使用全新输出目录重新运行。

## 12. 客户验收记录

建议客户在完成验收后记录以下信息：

| 项目 | 记录值 |
|---|---|
| 交付版本 | PP-OCR CPU v4 |
| 交付目录 | `D:\alipay-ai-data\delivery\pp-final-v4` |
| 迁移 ZIP SHA-256 | 由交付方与客户共同记录 |
| 服务器 CPU | 从 `batch-report.json` 记录 |
| 自动 worker/thread 方案 | 从 `batch-report.json` 记录 |
| 验收图片数 | 200 |
| 成功数 | 200 |
| 错误数 | 0 |
| 聚合吞吐量 | 不低于 1.0 张/秒 |
| 验收报告 | `batch-report.json` 的实际路径 |
| 验收日期 | 客户填写 |
| 验收人员 | 客户填写 |

完成以上步骤并取得 `PP_V4_CUSTOMER_ACCEPTANCE_PASS` 后，本版本可作为该服务器上的正式交付版本。
