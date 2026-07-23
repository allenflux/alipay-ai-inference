# Receipt ML.NET ONNX CLI

This .NET 8 command runs the current receipt detector and optional status-bar
device classifier through **ML.NET `ApplyOnnxModel`**. It validates the ONNX
SHA-256 against the required adjacent `.contract.json` sidecar before loading a
model.

It is intentionally a model-layer runner, not a silent replacement for the
Python pipeline:

- Included: EXIF orientation, RGB letterbox to detector tensor `image`
  `[3,1536,864]`, model execution, box restoration, score filtering, per-class
  best selection, device status-bar classification, JSON and batch manifests.
- Not included: receipt screen/quad detection and perspective correction,
  annotated images, PaddleOCR, and field-value normalization. Use already
  rectified images when perspective correction is needed.

## CPU

```powershell
dotnet restore .\dotnet\ReceiptMlNet.Cli\ReceiptMlNet.Cli.csproj
dotnet run --project .\dotnet\ReceiptMlNet.Cli\ReceiptMlNet.Cli.csproj -- `
  --detector .\artifacts\receipt_lrcnn_v1.onnx `
  --device-model .\artifacts\statusbar_device_v1.onnx `
  --input D:\input\one-receipt.png `
  --output D:\output\mlnet-cpu `
  --device cpu `
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
dotnet run --project .\dotnet\ReceiptMlNet.Cli\ReceiptMlNet.Cli.csproj -p:OnnxRuntimeFlavor=gpu -- `
  --detector .\artifacts\receipt_lrcnn_v1.onnx `
  --device-model .\artifacts\statusbar_device_v1.onnx `
  --input D:\input\one-receipt.png `
  --output D:\output\mlnet-gpu `
  --device cuda:0 `
  --require-complete
```

`--device auto` requests CUDA device 0 with ML.NET's CPU fallback enabled.
`--device cpu` forces CPU. `--device cuda:0` fails instead of silently falling
back when the GPU runtime is unavailable.
