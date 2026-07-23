# Alipay AI Inference

独立的转账回执模型推理项目。它不依赖训练项目 `/alipay-ai`，项目内已经包含图片
方向/透视纠正、LRCNN、OCR、结果渲染和批量 manifest 所需的运行时代码。

第一阶段只注册 `receipt_lrcnn_v1`，并且不加载 `status_style_v1`。后续增加其他模型时，
在 `src/receipt_inference/models.py` 注册模型目录，再按模型输入输出差异增加 runner。
可选的状态栏设备识别模型不在注册表中；通过 `--platform-checkpoint` 或
`--platform-onnx-model` 显式启用。

## 目录结构

```text
alipay-ai-inference/
  checkpoints/
    receipt_lrcnn_v1/
      best.pt                 # 部署时复制进来，不提交 Git
    statusbar_device_v1/
      best.pt                 # 可选：Android/iOS 状态栏设备识别
  artifacts/                  # 建议放导出的 ONNX 与 contract JSON
  src/
    receipt_inference/        # 多模型入口与模型注册表
    transfer_receipt_ai/      # 当前回执模型运行时代码
  tests/
  pyproject.toml
  requirements.txt
  requirements-ocr.txt
  requirements-export.txt
  requirements-dev.txt
```

## Windows + CUDA 环境配置

推荐 Python 3.10–3.12。先用 `nvidia-smi` 确认 NVIDIA 驱动，然后根据
[PyTorch 官方安装器](https://pytorch.org/get-started/locally/)选择与服务器匹配的
Windows CUDA wheel。不要直接让普通 PyPI 自动选择 PyTorch，否则可能安装 CPU 版。

在 PowerShell 中：

```powershell
cd D:\path\alipay-ai-inference
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip

# 示例版本仅适用于对应 CUDA 环境；以 PyTorch 官方安装器给出的命令为准。
# 本机 CUDA 12.6 时可使用 cu126 wheel index。
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
python -m pip install -r requirements.txt

# 先按 PaddlePaddle 官方说明安装匹配 CUDA 的 paddlepaddle-gpu，再安装 OCR。
# 示例为 CUDA 12.6；请按实际 CUDA wheel index 调整。
python -m pip install paddlepaddle-gpu==3.2.0 -i https://www.paddlepaddle.org.cn/packages/stable/cu126/
python -m pip install -r requirements-ocr.txt
python -m pip install -e . --no-deps
```

Windows 上若 LRCNN 和 PaddleOCR 都使用 GPU，必须使用 `paddleocr==2.10.0`（已由
`requirements-ocr.txt` 固定）。如果之前安装过 PaddleOCR 3.x，先清理其会隐式导入
Torch 的依赖链：

```powershell
python -m pip uninstall -y paddleocr paddlex modelscope albumentations albucore numpy opencv-python opencv-contrib-python opencv-python-headless
python -m pip install --no-cache-dir --force-reinstall -r requirements.txt -r requirements-ocr.txt
python -m pip install -e . --no-deps
python -m pip check
python -c "import importlib.metadata as m; print(m.version('paddleocr'), m.version('albumentations'), m.version('albucore'))"
```

预期版本为 `2.10.0 1.4.10 0.0.13`。先确认 Albumentations 没有偷偷导入 Torch：

```powershell
python -c "import sys, albumentations as A; print(A.__version__, 'torch' in sys.modules); assert A.__version__ == '1.4.10'; assert 'torch' not in sys.modules"
```

然后单独验证 PaddleOCR GPU（首次会下载 OCR 模型）：

```powershell
python -c "from transfer_receipt_ai.ocr import PaddleOCRReader; PaddleOCRReader(device='cuda', require_v2=True); import sys; assert 'torch' not in sys.modules; print('PaddleOCR 2 GPU OK')"
```

开发或运行测试时再安装：

```powershell
python -m pip install -r requirements-dev.txt
python -m pytest
```

检查 CUDA：

```powershell
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CUDA unavailable')"
```

PaddlePaddle/PaddleOCR 的 GPU wheel 必须和 Windows、Python、CUDA 版本匹配。如果暂时
安装 Paddle CPU 版，LRCNN 仍可使用 CUDA，但 OCR 会走 CPU。

## 放置模型

主检测器复制训练阶段选出的 `best.pt`：

```text
checkpoints\receipt_lrcnn_v1\best.pt
```

不需要复制 `last.pt`、训练数据、标注或训练日志。`best.pt` 已包含 LRCNN 模型结构配置、
固定类别顺序和模型权重。程序从项目安装位置解析默认 checkpoint，因此无需在命令里传
`--checkpoint`。

如果要输出 Android/iOS 设备识别字段，再额外复制：

```text
checkpoints\statusbar_device_v1\best.pt
```

它不是默认模型；运行时必须明确传入 `--platform-checkpoint`。

## 单图验证

以下命令已在 Windows CUDA 环境验证通过；项目目录是
`D:\alipay-ai-data\alipay-ai-inference`：

```powershell
python -m receipt_inference.cli `
  --input "D:\download\TempFakeImages\s3_voucher_GWCZ2071991511234514944_20260701001815.png" `
  --output "D:\download\TempFakeResults_v1" `
  --platform-checkpoint "D:\alipay-ai-data\alipay-ai-inference\checkpoints\statusbar_device_v1\best.pt" `
  --device cuda `
  --ocr paddle `
  --require-complete
```

## 批量推理

将 `--input` 指向图片目录即可递归处理目录下的所有受支持图片。先用 `--limit 100` 做
100 张试跑；输出目录应放在输入目录之外。`--continue-on-error` 会把坏图或缺少核心字段的
图片记录到 `inference_errors.jsonl`，但不阻断其他图片：

```powershell
python -m receipt_inference.cli `
  --input "D:\download\TempFakeImages" `
  --output "D:\download\TempFakeResults_batch100" `
  --platform-checkpoint "D:\alipay-ai-data\alipay-ai-inference\checkpoints\statusbar_device_v1\best.pt" `
  --device cuda `
  --ocr paddle `
  --require-complete `
  --continue-on-error `
  --limit 100
```

任务中断后，以相同的输入、输出和模型参数重新运行，并追加 `--skip-existing`，即可跳过已经
完成 OCR 的结果：

```powershell
python -m receipt_inference.cli `
  --input "D:\download\TempFakeImages" `
  --output "D:\download\TempFakeResults_batch100" `
  --platform-checkpoint "D:\alipay-ai-data\alipay-ai-inference\checkpoints\statusbar_device_v1\best.pt" `
  --device cuda `
  --ocr paddle `
  --require-complete `
  --continue-on-error `
  --limit 100 `
  --skip-existing
```

在 Windows GPU 环境中，以上 `--ocr paddle` 命令会自动使用**同一个虚拟环境**启动两个
顺序子进程：第一个只加载 PyTorch GPU 来检测，退出后第二个只加载 PaddleOCR GPU 来识别。
因此不会在同一 Windows Python 进程中混用两套 cuDNN DLL；输入路径、输出目录、JSON、标注图
和 manifest 的格式都不变。

`--ocr none` 的命令和行为保持不变：它只运行检测模型，不会启动 PaddleOCR。例如：

```powershell
python -m receipt_inference.cli `
  --input "D:\download\TempFakeImages\s3_voucher_GWCZ2071991511234514944_20260701001815.png" `
  --output "D:\download\TempFakeResults_v1" `
  --device cuda `
  --ocr none `
  --require-complete
```

要得到 GPU OCR 完整字段，只需将同一条命令中的 `--ocr none` 改为 `--ocr paddle`；其余参数不变。
不要在 Windows GPU 环境直接运行 `python -m transfer_receipt_ai.infer --ocr paddle`，该底层入口会在
一个进程中加载两套框架。

如果 checkpoint 缺失，命令会打印必须复制到的完整路径。临时验证另一份权重时，可以用
`--checkpoint D:\path\other.pt` 覆盖默认位置。

若没有部署设备识别模型，删除整段 `--platform-checkpoint ...` 即可。设备识别使用原图
顶部状态栏和分辨率规则，结果会写到每张结果 JSON 的 `device` 字段。

每张成功输入会生成结构化 JSON、纠正图圈选结果、原图透视回投圈选结果，以及批量
`inference_manifest.json`。启用 `--require-complete` 后，未检测齐金额、转账状态、收款方和
付款方式这四个核心字段的图片会失败；顶部 `time` 字段允许缺失。

## ONNX 交付与推理

ONNX 是标准交付物，不是 ML.NET 专属格式。当前项目可导出并运行：

- `receipt_lrcnn_v1` 主检测器；
- `statusbar_device_v1` 可选设备识别 CNN。

先在能够加载原始 `.pt` 的服务器安装导出验证依赖。`requirements-export.txt` 安装的是用于
导出校验的 CPU 版 `onnxruntime`。最终服务器若需要 CUDA ONNX 推理，必须将它替换为与 CUDA
匹配的 `onnxruntime-gpu`，不要同时保留两个 ONNX Runtime 包。

```powershell
python -m pip install -r requirements-export.txt
# Pull 到包含 ONNX 命令的代码后，刷新 editable console script。
python -m pip install -e . --no-deps
```

### Windows CUDA 12 ONNX 推理

当前部署环境若使用 CUDA 12.x（例如 `venv-cu126`），安装 CUDA 12 构建的 ONNX Runtime GPU
包。先移除 CPU/GPU 两种旧包，再限定到 1.26.x；这样不会误装从 1.27 起默认 CUDA 13 的 wheel：

```powershell
python -m pip uninstall -y onnxruntime onnxruntime-gpu
python -m pip install "onnxruntime-gpu>=1.21,<1.27"
python -c "import onnxruntime as ort; print(ort.__version__); print(ort.get_available_providers())"
```

最后一条必须显示 `CUDAExecutionProvider`。运行时会自动预加载 PyTorch 或 NVIDIA pip 包中的
CUDA/cuDNN DLL；如果未显示该 provider，不要传 `--device cuda`，先解决环境安装问题。

主检测器是 TorchVision Faster R-CNN。为保证 ONNX Runtime / .NET 的稳定性，导出固定的
单图输入画布，默认协议为 `image: float32 [3, 1536, 864]`（CHW、RGB、像素 `0..1`）。
ONNX 图内部保留 Faster R-CNN 的归一化；调用方不要再次做 ImageNet normalize。运行时将
透视纠正后的图片等比缩放并黑边 letterbox 到该画布，然后把检测框映射回纠正图坐标。

导出主检测器时需要一张真实回执图片用于跟踪检测图，并在相同固定输入上做 PyTorch
wrapper 与 ONNX Runtime CPU 对齐验证：

```powershell
receipt-model-export-onnx `
  --kind detector `
  --checkpoint "D:\path\alipay-ai-inference\checkpoints\receipt_lrcnn_v1\best.pt" `
  --sample-image "D:\input\one-receipt.png" `
  --output "D:\path\alipay-ai-inference\artifacts\receipt_lrcnn_v1.onnx" `
  --input-width 864 `
  --input-height 1536 `
  --resize-mode letterbox
```

导出设备识别模型：

```powershell
receipt-model-export-onnx `
  --kind statusbar `
  --checkpoint "D:\path\alipay-ai-inference\checkpoints\statusbar_device_v1\best.pt" `
  --output "D:\path\alipay-ai-inference\artifacts\statusbar_device_v1.onnx"
```

每个 ONNX 旁边都会生成 `.contract.json`，其中固定记录了来源 checkpoint SHA-256、输入输出
节点、类别顺序、画布大小、预处理、阈值、opset 与导出运行时版本；交付时两个文件必须成对
交付。

导出后，仍用同一个公共入口运行完整的 OpenCV / ONNX / PaddleOCR 流水线：

```powershell
python -m receipt_inference.cli `
  --onnx-model "D:\path\alipay-ai-inference\artifacts\receipt_lrcnn_v1.onnx" `
  --platform-onnx-model "D:\path\alipay-ai-inference\artifacts\statusbar_device_v1.onnx" `
  --input "D:\input\one-receipt.png" `
  --output "D:\output\receipt-onnx-test" `
  --device cuda:0 `
  --ocr paddle `
  --score-threshold 0.50 `
  --max-side 1600 `
  --require-complete
```

若不需要设备识别，删除 `--platform-onnx-model ...`。若运行的 ONNX 与 contract 中的
`resize_mode` 不同，必须显式传入 `--onnx-resize-mode stretch`；默认是 `letterbox`。
建议先用同一批图片分别跑 PyTorch 与 ONNX，比较每类最佳框、类别与分数，再把 ONNX 和
contract JSON 交付给 .NET。对于 Faster R-CNN 的变长 `boxes / labels / scores` 输出，C# 使用
ONNX Runtime 直接 API 通常比强行套 ML.NET 的 `IDataView` 更直接；如交付方明确要求 ML.NET，
则以同一份 ONNX contract 中的节点名和 tensor shape 接入 `ApplyOnnxModel`。

## CPU / macOS 开发环境

CPU 环境可以直接安装 PyTorch CPU wheel，并使用 `--device cpu`。macOS Apple Silicon
可以使用 `--device mps`；生产验收仍应在最终 Windows CUDA 服务器上进行。

## 后续增加模型

模型名称和默认 checkpoint 路径集中在：

```text
src\receipt_inference\models.py
```

新增模型建议保持以下约定：

```text
checkpoints\<model_name>\best.pt
```

如果新模型与 `receipt_lrcnn_v1` 使用相同输入输出协议，只需注册模型并选择对应 predictor；
如果协议不同，则新增独立 runner，避免把多个模型的条件判断堆进一个推理函数。
