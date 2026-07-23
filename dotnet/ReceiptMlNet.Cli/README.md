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
