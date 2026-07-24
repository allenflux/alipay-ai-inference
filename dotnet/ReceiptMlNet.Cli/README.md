# Receipt ML.NET ONNX CLI

This .NET 8 command runs the receipt detector and optional status-bar device
classifier through **ML.NET `ApplyOnnxModel`**. It validates the ONNX SHA-256
against the required adjacent `.contract.json` sidecar before loading a model.
When passed `--ocr onnx --ocr-bundle <delivery-directory>`, it additionally
loads the frozen PP-OCR `det + cls + rec` ONNX bundle through direct ONNX
Runtime sessions. The OCR delivery target is Windows x64 because the adapter
uses the Windows OpenCvSharp native runtime.

It is intentionally a model-layer runner, not a silent replacement for the
Python pipeline:

- Included: EXIF orientation, RGB letterbox to detector tensor `image`
  `[3,1536,864]`, model execution, box restoration, score filtering, per-class
  best selection, device status-bar classification, inspection JPGs, JSON and
  batch manifests.
- Included when `--ocr onnx` is selected: DB text detection, text perspective
  crops, angle classification, CTC decoding, OCR text, and the same receipt
  field normalisation rules as the Python pipeline.
- Not included: receipt screen/quad detection and source-image perspective
  correction. Use already rectified images when perspective correction is
  needed.

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

This runner has not ported Python's OpenCV perspective rectification or
homography projection. Therefore, for now both compatibility-named JPGs show
the same EXIF-upright source-coordinate annotation. They visually match the
Python rectified annotation when the input is already rectified, but are not a
pixel-for-pixel substitute for Python's original-photo projection.

## CPU

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

`--device cuda:0` is the strict GPU acceptance mode: the detector, device
classifier, and all three PP-OCR sessions request CUDA device 0. It fails
instead of silently falling back when the GPU runtime is unavailable. ONNX
Runtime can still keep small shape/metadata operators on CPU. `--device auto`
allows fallback and is not a GPU acceptance command. Build the matching
`OnnxRuntimeFlavor` immediately before `--no-build`; a CPU build cannot be
reused for a GPU command, or vice versa. See the repository README for the
100 / 10,000-image CPU and GPU commands.
