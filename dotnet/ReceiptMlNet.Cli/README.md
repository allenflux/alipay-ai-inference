# Receipt ML.NET ONNX CLI

This .NET 8 command runs the receipt detector and optional status-bar device
classifier through **ML.NET `ApplyOnnxModel`**. It validates the ONNX SHA-256
against the required adjacent `.contract.json` sidecar before loading a model.
It has two mutually exclusive OCR paths:

- `--ocr onnx --ocr-bundle <delivery-directory>` loads the frozen PP-OCR
  `det + cls + rec` ONNX bundle through direct ONNX Runtime sessions.
- `--ocr unified --ocr-model <receipt_unified_field_reader_v12.onnx>` loads one
  v12 five-field unified reader through one direct ONNX Runtime Session.

The OCR delivery target is Windows x64 because the PP-OCR adapter uses the
Windows OpenCvSharp native runtime.

It is intentionally a model-layer runner, not a silent replacement for the
Python pipeline:

- Included: EXIF orientation, RGB letterbox to detector tensor `image`
  `[3,1536,864]`, model execution, box restoration, score filtering, per-class
  best selection, device status-bar classification, inspection JPGs, JSON and
  batch manifests.
- Included when `--ocr onnx` is selected: DB text detection, text perspective
  crops, angle classification, CTC decoding, OCR text, and the same receipt
  field normalisation rules as the Python pipeline.
- Included when `--ocr unified` is selected: frozen detector-field crop
  geometry plus v12 contract preprocessing, two static inputs in one ONNX
  graph, one `InferenceSession` reused for the CLI run, and one `Run` per
  receipt. `field_images` contains amount/time/status/payment plus a white
  reserved fifth slot; `recipient_value_image` is the dedicated high-resolution
  recipient value crop. Candidate text is diagnostic; the current v12 contract
  delivers every text/status business value as `review`.
- Included for the production wrapper: EXIF-upright full-image OpenCV cubic
  normalization with longest side 1600 (`--rectification max-side-1600`),
  matching the Python direct-screenshot path. Detected boxes are projected
  back to the upright source coordinates.
- Not included: automatic receipt screen/quad detection. Perspective photos
  still need an externally rectified input.

## 当前三模型交付（Windows x64，CPU 正式生产）

完整的 unified v12 交付由 **3 个神经网络 ONNX** 组成，不是每个字段各放一个
OCR 模型：

| 阶段 | 文件 | 是否必需 | 作用 |
| --- | --- | --- | --- |
| 回单字段检测 | `receipt_lrcnn_v1.onnx` + `.contract.json` | 必需 | 检测并还原金额、时间、状态、付款方式、收款方等字段框 |
| 设备识别 | `statusbar_device_v1.onnx` + `.contract.json` | 可选；完整交付建议包含 | 根据状态栏识别 iOS/Android |
| 统一字段 OCR | `best.onnx` + `best.labels.json` + `best.contract.json` | 必需 | 一张图一次 unified `Run`，同时识别金额、时间、状态、付款方式和收款方 |

`*.contract.json` 和 `best.labels.json` 是模型契约、字符表及哈希证据，不是额外
神经网络。运行顺序是：输入已摆正/已透视纠正的回单图 → 检测字段框 → 裁出字段图 →
一次 unified OCR → 规范化并写 JSON；设备模型可在同一流程中附加设备结果。当前 .NET
程序会执行与 Python 直接截图口径一致的长边 1600 全图 OpenCV 规范化，但还不做票面
四边形自动检测；带透视的拍照原图必须先完成外部矫正。

Paddle 只在训练准备阶段生成教师标签。发布目录中的 .NET 8 程序、检测器、设备模型和
unified OCR ONNX 在生产推理时都不加载 Paddle、PaddleOCR 或 Python。下面的打包脚本会用
项目 Python 环境整理验证输入和计算候选一致率，但这属于交付验收，不是生产运行依赖。

正式生产基线固定为 **CPU**：`OnnxRuntimeFlavor=cpu`、`--device cpu`。4090/GPU
用于训练或可选性能检查，不能代替 CPU 正式验收。打包脚本使用 Release publish，整个批次
只启动一次进程并复用模型 Session；批量验收使用 `-Annotate none` 可避免 JPG 绘制开销。
正式 wrapper 同时固定 `--rectification max-side-1600`，避免训练 crop 与生产 crop 走不同
的重采样路径。

当前已固定上述 wide1536 产物和数据路径的机器可直接使用一键入口；它自动创建带 UTC
时间戳的新输出目录，完整包始终包含并实际运行设备模型：

```powershell
Set-Location D:\alipay-ai-data\alipay-ai-inference
& .\scripts\receipt-mlnet-production-cpu-validate.ps1 -Mode smoke
& .\scripts\receipt-mlnet-production-cpu-validate.ps1 -Mode pilot -PilotLimit 100
& .\scripts\receipt-mlnet-production-cpu-validate.ps1 -Mode formal
```

第一条只跑 CPU 接线与速度冒烟；第二条按 manifest 顺序跑前 100 张并评分，但报告会固定
标为 `partial_pilot`、`formal_delivery_gate=false`，只能用于决定是否值得跑全量；第三条才
执行完整 val、四字段评分和正式原子发布。下面是
等价的展开命令，便于改路径或接入 CI。

### 1. CPU 单图冒烟（只验证接线，不是正式门槛）

以下路径对应当前 wide1536 训练产物。`-Output` 和 `-DeliveryDir` 必须是尚不存在且
互不嵌套的新目录；重跑时更换目录后缀。

```powershell
Set-Location D:\alipay-ai-data\alipay-ai-inference

$run = "D:\alipay-ai-data\receipt-lite-teacher-120k-v1\unified-run-v12-r3-4090-paddle-fit-open-text-joint-wide1536-20260806-114954"
$sample = "D:\download\TempFakeImages\s3_voucher_GWCZ2071991511234514944_20260701001815.png"
$smokeOutput = "D:\alipay-ai-data\delivery-validation\mlnet-wide1536-cpu-smoke-1"
$smokePackage = "D:\alipay-ai-data\delivery\ReceiptMlNet-wide1536-cpu-smoke-1"

& .\scripts\receipt-mlnet-unified-package-validate-4090.ps1 `
  -RunDirectory $run `
  -InputPath $sample `
  -Output $smokeOutput `
  -DeliveryDir $smokePackage `
  -Limit 1 `
  -RuntimeFlavor cpu `
  -Rectification max-side-1600 `
  -IncludeDeviceModel `
  -Annotate all
```

没有同时传入 `-Records` 与 `-EndToEndEvaluationDir` 时，脚本会明确把包标为
`candidate_smoke_only`。`-Limit` 也只能用于冒烟，不能产生正式验收包。

### 2. CPU 全量端到端正式验收与打包

先从与 `onnx-val/summary.json` 绑定的同一份 v12 manifest 生成完整 `val` 输入列表，
再运行 CPU 正式门槛。正式命令不能带 `-Limit`；脚本还会重新生成 canonical val 列表，
拒绝缺图、多图或不属于同一 val split 的列表。

```powershell
Set-Location D:\alipay-ai-data\alipay-ai-inference

$run = "D:\alipay-ai-data\receipt-lite-teacher-120k-v1\unified-run-v12-r3-4090-paddle-fit-open-text-joint-wide1536-20260806-114954"
$records = "D:\alipay-ai-data\receipt-lite-teacher-120k-v1\unified-manifest-v12-r3-4090-r1\unified_fields.jsonl"
$inputList = "D:\alipay-ai-data\delivery-validation\mlnet-wide1536-val-inputs.txt"
$cpuOutput = "D:\alipay-ai-data\delivery-validation\mlnet-wide1536-cpu-full"
$cpuEvaluation = "D:\alipay-ai-data\delivery-validation\mlnet-wide1536-cpu-full-e2e"
$cpuPackage = "D:\alipay-ai-data\delivery\ReceiptMlNet-wide1536-cpu-production"

& .\.venv-cu126\Scripts\python.exe `
  .\scripts\receipt_mlnet_unified_evaluate.py prepare `
  --records $records `
  --output $inputList `
  --split val

& .\scripts\receipt-mlnet-unified-package-validate-4090.ps1 `
  -RunDirectory $run `
  -InputList $inputList `
  -Records $records `
  -EndToEndEvaluationDir $cpuEvaluation `
  -Output $cpuOutput `
  -DeliveryDir $cpuPackage `
  -RuntimeFlavor cpu `
  -Rectification max-side-1600 `
  -IncludeDeviceModel `
  -Annotate none
```

正式门槛为：完整 val 回单覆盖率为 100%，manifest 中实际带真值的四字段候选覆盖率各为
100%（不要求一张只标了部分字段的回单凭空补齐未标字段），没有推理错误或混用模型哈希；
金额 exact match ≥ **78.85%**，时间 ≥ **98.40%**，付款方式 ≥ **93.25%**，
收款方 ≥ **90.00%**。脚本还会核对 `best.onnx`、labels、contracts、检测器和设备
模型哈希，以及既有 `onnx-val/summary.json` 的 val split、manifest、字段样本数和保护线。
任一条件失败时不会发布 `$cpuPackage`。

CPU 性能证据写在 `$cpuOutput\inference_summary.json`，其中
`inference_latency_ms.mean/p50/p95` 分别是本次 CPU 全量运行的平均、P50 和 P95
单图模型推理耗时；`stage_latency_ms` 进一步拆出图像加载、设备识别、检测预处理/推理/
后处理、统一 OCR 预处理/推理/后处理与结果组装。相同文件会复制到
`$cpuPackage\evidence\inference_summary.json`；控制台末尾也会打印 `mean-ms`、
`p50-ms` 和 `p95-ms`。正式报告必须引用这组 CPU 数值，不能引用 GPU 数值替代。

### 3. 正式包结构与生产运行命令

正式验收成功后，主要输出如下：

- `$cpuOutput`：逐图 JSON、`inference_manifest.json`、`inference_errors.jsonl`、
  `inference_summary.json`；若启用标注还会包含 JPG。
- `$cpuEvaluation\summary.json`：四字段端到端候选一致率、覆盖率、门槛与失败项；
  `comparisons.jsonl` 是逐字段比对明细。
- `$cpuPackage\app\ReceiptMlNet.Cli.exe`：Release、framework-dependent、win-x64、
  CPU 版生产程序。
- `$cpuPackage\models`：检测器、可选设备模型，以及 `unified\best.*` 三件套。
- `$cpuPackage\evidence`：`validation-input-list.txt`、`console.log`、
  `inference_summary.json`、`package_validation.json`、
  `end-to-end-evaluation-summary.json`、`end-to-end-comparisons.jsonl`、
  `result_evidence_sha256.json`、`onnx-validation-summary.json` 以及 manifest/errors；
  根目录 `SHA256SUMS.json` 可用于整包校验。

生产机器只需 .NET 8 x64 Runtime；完整三模型单图命令如下：

```powershell
$package = "D:\alipay-ai-data\delivery\ReceiptMlNet-wide1536-cpu-production"
$receiptInput = "D:\input\one-receipt.png"
$output = "D:\output\one-receipt-result"

& "$package\app\ReceiptMlNet.Cli.exe" `
  --detector "$package\models\receipt_lrcnn_v1.onnx" `
  --device-model "$package\models\statusbar_device_v1.onnx" `
  --ocr unified `
  --ocr-model "$package\models\unified\best.onnx" `
  --input $receiptInput `
  --output $output `
  --device cpu `
  --rectification max-side-1600 `
  --annotate all `
  --require-complete
```

批量生产把 `--input` 改为图片目录，或改用 `--input-list <txt>`。完整交付包含设备
模型；不需要识别设备时，可同时从包和命令中去掉 `statusbar_device_v1.*`/
`--device-model`，其余流程不变。

当前 artifact policy 仍然 fail closed：真正的识别文本在
`fields.amount/time/payment_method/recipient.candidate`，业务 `value` 仍为
`review`。上述门槛验证的是 ML.NET 端到端候选与 Paddle 教师标签的一致率，不把教师标签
宣称为独立人工真值，也不会自动改变交付策略。

### 4. 可选 4090/GPU 性能检查

GPU 只用于冒烟或 benchmark，因此不要传 `-Records`/
`-EndToEndEvaluationDir`，也不要把生成的 `candidate_smoke_only` 包作为正式包：

```powershell
& .\scripts\receipt-mlnet-unified-package-validate-4090.ps1 `
  -RunDirectory $run `
  -InputPath $sample `
  -Output "D:\alipay-ai-data\delivery-validation\mlnet-wide1536-gpu-smoke-1" `
  -DeliveryDir "D:\alipay-ai-data\delivery\ReceiptMlNet-wide1536-gpu-smoke-1" `
  -Limit 1 `
  -RuntimeFlavor gpu `
  -IncludeDeviceModel `
  -Annotate none
```

GPU 模式要求 CUDA 12.x、cuDNN 9.x 运行库，并验证 `cuda:0`；CPU 正式包不需要
这些 NVIDIA DLL。

## Annotated output

`--annotate all` is the default and writes the same compatibility names used by
the Python pipeline beside each result JSON:

- `<stem>_rectified_annotated.jpg`
- `<stem>_original_annotated.jpg`

The image uses the same field colors, expanded ellipse outlines, fixed label
order, score legend, and red device row. `--annotate flagged` writes JPGs only
when one of the four core transfer fields is missing; `--annotate none` writes
JSON only. The JSON is written after the derived JPGs, and `--skip-existing`
still treats JSON as the completion marker.

`--require-complete` only requires the detector to find its core field boxes.
It does not turn v12 candidates into delivered business values: current v12
`fields.*.value` remains `review` until a separately calibrated artifact policy
permits delivery.

With `--rectification max-side-1600`, this runner ports Python's deterministic
portrait rule (EXIF-upright landscape inputs rotate 90 degrees clockwise),
full-image OpenCV cubic normalization, and homography projection for the
direct-screenshot path. Detection and OCR run on that normalized image; JSON boxes and both
compatibility-named JPGs are projected back to EXIF-upright source coordinates.
Automatic phone/screen quadrilateral detection is intentionally not included,
so perspective photos still require external rectification before inference.

## PP-OCR CPU

```powershell
dotnet restore .\dotnet\ReceiptMlNet.Cli\ReceiptMlNet.Cli.csproj
dotnet build .\dotnet\ReceiptMlNet.Cli\ReceiptMlNet.Cli.csproj -p:OnnxRuntimeFlavor=cpu
$ocrBundle = ".\artifacts\paddle_ocr_ppocrv4_v1_delivery"

dotnet run --no-build --project .\dotnet\ReceiptMlNet.Cli\ReceiptMlNet.Cli.csproj -p:OnnxRuntimeFlavor=cpu -- `
  --detector .\artifacts\receipt_lrcnn_v1.onnx `
  --device-model .\artifacts\statusbar_device_v1.onnx `
  --ocr onnx `
  --ocr-bundle $ocrBundle `
  --input D:\input\one-receipt.png `
  --output D:\output\mlnet-onnx-ocr-cpu-1 `
  --device cpu `
  --annotate all `
  --require-complete
```

## v12 unified CPU

The v12 reader is not a PP-OCR bundle. Deliver these three adjacent files with
their names unchanged:

```text
receipt_unified_field_reader_v12_*.onnx
receipt_unified_field_reader_v12_*.labels.json
receipt_unified_field_reader_v12_*.contract.json
```

The CLI verifies their hashes, v12 two-input ABI, 15 output names/shapes,
charsets, recipient preprocessing, and review-only policy before opening the
session. It intentionally rejects v3-v11 readers and never accepts this ONNX
through `--ocr-bundle`.

```powershell
$unifiedModelV12 = "D:\alipay-ai-data\receipt-lite-teacher-120k-v1\models\receipt_unified_field_reader_v12_120k_r1.onnx"

dotnet build .\dotnet\ReceiptMlNet.Cli\ReceiptMlNet.Cli.csproj -p:OnnxRuntimeFlavor=cpu
dotnet run --no-build --project .\dotnet\ReceiptMlNet.Cli\ReceiptMlNet.Cli.csproj -p:OnnxRuntimeFlavor=cpu -- `
  --detector .\artifacts\receipt_lrcnn_v1.onnx `
  --device-model .\artifacts\statusbar_device_v1.onnx `
  --ocr unified `
  --ocr-model $unifiedModelV12 `
  --input D:\input\one-receipt.png `
  --output D:\output\mlnet-unified-v12-cpu-1 `
  --device cpu `
  --annotate all `
  --require-complete
```

## CUDA

GPU uses the `Microsoft.ML.OnnxRuntime.Gpu` package selected by the build
property below. Start by forcing `cuda:0`; use `auto` only after it succeeds.

The NuGet package contains the ONNX Runtime CUDA provider but does **not**
include NVIDIA's CUDA/cuDNN runtime DLLs. On Windows, provide compatible CUDA
12.x and cuDNN 9.x DLLs on `PATH` before starting `dotnet`. When the adjacent
Python environment has a CUDA 12 PyTorch build, its `torch\lib` directory is
usually enough for a development test:

```powershell
$torchLib = python -c "import os, torch; print(os.path.join(os.path.dirname(torch.__file__), 'lib'))"
Test-Path "$torchLib\cublasLt64_12.dll"
Test-Path "$torchLib\cudnn64_9.dll"
$env:Path = "$torchLib;$env:Path"
```

Both checks should be `True`. This only changes the current PowerShell
session. For a production .NET-only delivery, install CUDA 12.x and cuDNN 9.x
for Windows x64 and add their `bin` directories to the system `PATH`; do not
copy individual DLLs into the project.

```powershell
dotnet build .\dotnet\ReceiptMlNet.Cli\ReceiptMlNet.Cli.csproj -p:OnnxRuntimeFlavor=gpu

dotnet run --no-build --project .\dotnet\ReceiptMlNet.Cli\ReceiptMlNet.Cli.csproj -p:OnnxRuntimeFlavor=gpu -- `
  --detector .\artifacts\receipt_lrcnn_v1.onnx `
  --device-model .\artifacts\statusbar_device_v1.onnx `
  --ocr onnx `
  --ocr-bundle $ocrBundle `
  --input D:\input\one-receipt.png `
  --output D:\output\mlnet-onnx-ocr-gpu-1 `
  --device cuda:0 `
  --require-complete
```

## v12 unified CUDA

After the CUDA/cuDNN `PATH` preparation above, build the GPU flavor and use
the same three-file v12 artifact:

```powershell
dotnet build .\dotnet\ReceiptMlNet.Cli\ReceiptMlNet.Cli.csproj -p:OnnxRuntimeFlavor=gpu

dotnet run --no-build --project .\dotnet\ReceiptMlNet.Cli\ReceiptMlNet.Cli.csproj -p:OnnxRuntimeFlavor=gpu -- `
  --detector .\artifacts\receipt_lrcnn_v1.onnx `
  --device-model .\artifacts\statusbar_device_v1.onnx `
  --ocr unified `
  --ocr-model $unifiedModelV12 `
  --input D:\input\one-receipt.png `
  --output D:\output\mlnet-unified-v12-gpu-1 `
  --device cuda:0 `
  --annotate all `
  --require-complete
```

`--device cuda:0` is the strict GPU acceptance mode: the detector and device
classifier request CUDA device 0; PP-OCR additionally requests it for its
three sessions, while v12 requests it for its single unified session. It fails
instead of silently falling back when the GPU runtime is unavailable. ONNX
Runtime can still keep small shape/metadata operators on CPU. `--device auto`
allows fallback and is not a GPU acceptance command. Build the matching
`OnnxRuntimeFlavor` immediately before `--no-build`; a CPU build cannot be
reused for a GPU command, or vice versa. See the repository README for the
100 / 10,000-image CPU and GPU commands.
