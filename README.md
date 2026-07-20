# Alipay AI Inference

独立的转账回执模型推理项目。它不依赖训练项目 `/alipay-ai`，项目内已经包含图片
方向/透视纠正、LRCNN、OCR、结果渲染和批量 manifest 所需的运行时代码。

第一阶段只注册 `receipt_lrcnn_v1`，并且不加载 `status_style_v1`。后续增加其他模型时，
在 `src/receipt_inference/models.py` 注册模型目录，再按模型输入输出差异增加 runner。

## 目录结构

```text
alipay-ai-inference/
  checkpoints/
    receipt_lrcnn_v1/
      best.pt                 # 部署时复制进来，不提交 Git
  src/
    receipt_inference/        # 多模型入口与模型注册表
    transfer_receipt_ai/      # 当前回执模型运行时代码
  tests/
  pyproject.toml
  requirements.txt
  requirements-ocr.txt
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
python -m pip install -e .
```

Windows 上若 LRCNN 和 PaddleOCR 都使用 GPU，必须使用 `paddleocr==2.10.0`（已由
`requirements-ocr.txt` 固定）。如果之前安装过 PaddleOCR 3.x，先清理其会隐式导入
Torch 的依赖链：

```powershell
python -m pip uninstall -y paddleocr paddlex modelscope albumentations albucore numpy opencv-python opencv-contrib-python opencv-python-headless
python -m pip install --no-cache-dir --force-reinstall -r requirements.txt -r requirements-ocr.txt
python -m pip install -e .
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

只复制训练阶段选出的 `best.pt`：

```text
checkpoints\receipt_lrcnn_v1\best.pt
```

不需要复制 `last.pt`、训练数据、标注或训练日志。`best.pt` 已包含 LRCNN 模型结构配置、
固定类别顺序和模型权重。程序从项目安装位置解析默认 checkpoint，因此无需在命令里传
`--checkpoint`。

## 单图验证

```powershell
receipt-model-infer `
  --input "D:\download\TempFakeImages\s3_voucher_GWCZ2071991511234514944_20260701001815.png" `
  --output "D:\download\TempFakeResults_v1" `
  --device cuda `
  --ocr paddle `
  --require-complete
```

也可以使用模块入口：

```powershell
python -m receipt_inference.cli `
  --input "D:\path\receipt.png" `
  --output "D:\path\results" `
  --device cuda `
  --ocr paddle `
  --require-complete
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

每张成功输入会生成结构化 JSON、纠正图圈选结果、原图透视回投圈选结果，以及批量
`inference_manifest.json`。启用 `--require-complete` 后，未检测齐五个字段的图片会失败，
不会写成完整结果。

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
