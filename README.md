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
  requirements-train-ocr.txt
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

## 纯 ONNX OCR 交付（保持当前 PaddleOCR 效果的路线）

前面的自训练 CTC 是一个实验候选，不应直接替换当前 PaddleOCR：本次 1000 张伪标签的保留集验收只得到
`raw_exact_match=52.65%`、`micro_cer=0.1302`，且收款方有大量未覆盖中文字符。因此交付版本应冻结**当前
实际运行成功**的 PaddleOCR 资产并导出 ONNX，而不是继续加大自训练模型。

当前 Windows 服务器已确认模型类型是 PP-OCRv4 det/rec + mobile v2.0 cls。默认缓存通常是以下布局（最终以
冻结出的 contract 中 `source_directory` 和 `rec_char_dict_path` 为准；不要硬编码 Windows 用户名）：

```text
%USERPROFILE%\.paddleocr\whl\det\ch\ch_PP-OCRv4_det_infer
%USERPROFILE%\.paddleocr\whl\rec\ch\ch_PP-OCRv4_rec_infer
%USERPROFILE%\.paddleocr\whl\cls\ch_ppocr_mobile_v2.0_cls_infer
%VIRTUAL_ENV%\Lib\site-packages\paddleocr\ppocr\utils\ppocr_keys_v1.txt
```

前三项分别是文字检测、中文识别、方向分类；最后一项是与识别模型严格匹配的字符表。不要改用列表中其他
`*_dict.txt`，尤其不要用 `chinese_cht_dict.txt`。也不要只导出识别模型：当前生产逻辑是
`PaddleOCR(..., use_angle_cls=True).ocr(crop, cls=True)`，需要完整的 det + cls + rec 三段。

先从**已成功运行当前 PaddleOCR 的同一个虚拟环境**冻结资产。此命令以 CPU 初始化，只读取相同缓存，不会影响
现有 GPU 推理；默认若缓存缺失会报错而不会暗中下载新模型：

```powershell
$project = "D:\alipay-ai-data\alipay-ai-inference"
$bundle = "$project\artifacts\paddle_ocr_ppocrv4_v1"

python -m transfer_receipt_ai.paddle_ocr_bundle snapshot `
  --output $bundle `
  --device cpu

Get-Content "$bundle\paddle_ocr_bundle.contract.json" -Encoding UTF8
```

输出是不可覆盖的审计快照，内含三个 Paddle 静态模型、字符表、有效 PaddleOCR 参数、版本、文件大小和 SHA-256。
冻结命令也会记录当前 Python 流水线的关键兼容规则：输入是直接传给 Paddle v2 的 RGB crop；多行文本清洗后以单个
空格连接，置信度取行级置信度均值。C# 端必须照此实现，不能自行加 RGB/BGR 交换。

然后在当前 Windows venv 中安装适用于 Windows/Python 3.10 的转换器。使用 `--no-deps`，避免它改动已经能运行的
Paddle/PaddleOCR/CUDA 依赖；不要使用 Paddle2ONNX 2.1.x：它针对 Paddle 3 依赖链，不是本项目的锁定转换环境。

```powershell
python -m pip install --no-deps paddle2onnx==1.3.0
$converter = "$env:VIRTUAL_ENV\Scripts\paddle2onnx.exe"
& $converter --version

python -m transfer_receipt_ai.paddle_ocr_bundle export-onnx `
  --bundle $bundle `
  --paddle2onnx $converter

python -m transfer_receipt_ai.paddle_ocr_bundle verify `
  --bundle $bundle `
  --require-onnx
```

它会从冻结副本而不是 live cache 导出，并对每个 ONNX 再运行 ONNX checker 和 CPU ONNX Runtime 加载检查；输出为：

```text
artifacts\paddle_ocr_ppocrv4_v1\
  onnx\paddle_ocr_det.onnx
  onnx\paddle_ocr_rec.onnx
  onnx\paddle_ocr_cls.onnx
  charset\ppocr_keys_v1.txt
  paddle_ocr_bundle.contract.json
  paddle\...                    # 只用于审计/可复现转换，不放最终交付包
```

转换成功后，先在**同一批字段裁图**上验证“Paddle 原生推理 vs PaddleOCR 2.10 的 `use_onnx=True` 推理”。两侧复用
官方完全相同的 det/cls/rec 前后处理，只替换底层推理图，因此这是转换是否保持效果的第一道门槛。先抽 100 张，再跑全部：

```powershell
$crops = "D:\download\ReceiptOcrPseudoV3\images"

python -m transfer_receipt_ai.paddle_ocr_bundle validate-onnx `
  --bundle $bundle `
  --input $crops `
  --output "D:\download\PaddleOcrOnnxParity100" `
  --limit 100 `
  --min-text-exact-match 1.0 `
  --max-confidence-delta 0.01
```

成功后将输出目录换成 `D:\download\PaddleOcrOnnxParityAll`、删除 `--limit 100` 再跑全量。输出的
`comparisons.jsonl` 包含逐 crop 文本、行序列、置信度和耗时，`summary.json` 包含精确匹配率与 p50/p95。它只验证
Paddle → ONNX 的转换一致性；通过后才进入 Python ONNX → C# ONNX 的逐字段、JSON 和性能回归。

不要传 `--input_shape_dict` 或进行静态 shape / INT8 量化；OCR 检测模型的高宽和识别模型的宽度必须先保持动态，才能做
当前效果的一致性基线。审计 bundle 必须保留 `paddle/`，不要从它手动删除源模型；完成逐字段回归后，用以下命令创建
单独的、轻量的纯 ONNX 交付目录：

```powershell
$delivery = "$project\artifacts\paddle_ocr_ppocrv4_v1_delivery"

python -m transfer_receipt_ai.paddle_ocr_bundle package-delivery `
  --bundle $bundle `
  --output $delivery

python -m transfer_receipt_ai.paddle_ocr_bundle verify-delivery `
  --delivery $delivery
```

该目录只带 `onnx/`、`charset/` 和 `paddle_ocr_delivery.contract.json`，不带 `.pdmodel`、Paddle 或 Python。

### CPU 是正式交付目标

CPU 不是 GPU 不可用时的隐式降级，而是独立验收目标：`export-onnx` 对每份 ONNX 强制用
`CPUExecutionProvider` 做一次加载检查；`validate-onnx` 也固定在 CPU 上运行原生 Paddle 与 ONNX 的文本对比。
因此上面的 100 张和全量命令成功，才说明这三个 OCR ONNX 在 CPU 可运行且保持转换前效果；它不需要 CUDA、cuDNN 或
Paddle GPU。

`.NET OCR CLI` 的 `--device cpu` 会固定创建 CPU ONNX Runtime Session；CUDA 只是同一套 ONNX 的可选
加速路径，不改变 CPU 的结果契约。C# 侧仍须用下面的 CPU / GPU 命令对同一批回单做字段、JSON 和性能回归。

### 速度与轻量化策略

| 阶段 | 运行内容 | 目的 |
| --- | --- | --- |
| 基线 | 每个字段执行 OCR det + cls + rec | 先证明文字和字段值与当前 Paddle 一致 |
| 后续快路径 | 对正常横向、单行且高置信字段直接 cls + rec | 仅在完整基线回归通过后实施，跳过文字检测以降低多数回单耗时 |
| 后续回退 | 快路径低置信、旋转、多行或字段规则异常时回到 det + cls + rec | 不用速度换取错误结果 |
| 压缩 | 只在一致性回归通过后测试 CPU INT8 | 量化一旦影响中文/金额/时间任一门槛即不采用 |

三个 OCR ONNX Session 会在进程启动时只加载一次；基线按字段顺序处理，并在识别 ONNX 的 batch 轴可变时按
Paddle 的文本框宽高比分组批量识别，否则安全地逐条识别。GPU 使用 CUDA Execution Provider，CPU 使用 CPU
Execution Provider。OCR 的 det/rec 有动态 shape、文本框后处理和 CTC 解码，因此 .NET 端会直接用
`Microsoft.ML.OnnxRuntime.InferenceSession`，不强行塞进固定张量的 `ApplyOnnxModel`；这仍然是纯 ONNX/.NET 交付。
验收同时记录全量 1000 张的逐字段文本/归一化值、p50/p95 单图耗时、总吞吐、显存/内存和最终包体积。

## 自训练 OCR（不在交付时依赖 Paddle）

已有的 Python `--ocr paddle` 结果可以作为**伪标签**，用来训练独立的 PyTorch CTC 文字识别器。
训练和导出过程不会导入 Paddle；最终产物是 ONNX、字符表和 contract，供后续 C# / ONNX Runtime
OCR 接入使用。

不要直接拿带红圈的 `*_annotated.jpg` 训练。下面的脚本读取结果 JSON 中的 `source`、几何和
`detections[].ocr`，重建干净的矫正图，并用与当前 OCR 相同的 8% 边距规则裁出字段图。最终 JSON 的
几何坐标经过序列化取整，因此裁图是可复现的近似重建；它不使用带红圈预览图，也不是私有全精度
Paddle 阶段裁图的逐像素副本。默认只接受
检测置信度 `>= 0.90`、Paddle OCR 置信度 `>= 0.98` 且能通过字段语义检查的样本；低质量样本会进入
`rejected.jsonl`，不会混入训练集。

原始图片在生成这批 Paddle 结果后不要再覆盖或编辑；导出器会检查修改时间并拒绝明显已变更的图片，
但旧结果 JSON 没有保存原图内容哈希，因此最稳妥的做法仍是保持原始图片目录只读。
只有在迁移文件后已人工确认内容完全相同、仅修改时间改变时，才可以显式加
`--allow-source-newer` 跳过这一层保护。

例如，已在服务器生成的结果目录为
`D:\download\TempFakeResults_onnx_cpu_ocr_10000` 时：

```powershell
python -m transfer_receipt_ai.ocr_pseudolabels `
  --results "D:\download\TempFakeResults_onnx_cpu_ocr_10000" `
  --output "D:\download\ReceiptOcrPseudoV1" `
  --min-detector-score 0.90 `
  --min-ocr-confidence 0.98 `
  --min-ocr-confidence-override "amount=0.90" `
  --test-ratio 0.10 `
  --review-ratio 0.10 `
  --continue-on-error
```

如果某字段因全局 `0.98` 门槛而几乎没有样本，可以只降低该字段，例如上面的 `amount=0.90`；其余字段
仍保持 `0.98`。不要为了补金额而降低全部字段的阈值。

输出目录包含：

- `images/<field>/*.png`：没有标注框的干净字段裁图；
- `pseudo_labels.jsonl`：通过筛选的训练记录；
- `review_candidates.jsonl`：稳定抽取的人工复核样本；
- `rejected.jsonl` / `build_errors.jsonl`：被拒绝或无法读取的原因；
- `splits/train.jsonl`、`val.jsonl`、`test.jsonl`：按同一回单分组的无泄漏切分；
- `charset.txt`、`character_coverage.json`、`dataset_config.json`：构建期字符覆盖与筛选条件。

先抽查 `review_candidates.jsonl` 指向的图片和文字。Paddle 伪标签适合扩大训练样本，**不能**作为
最终准确率证明；应另留人工修正、且不参与训练的验证/测试样本。1000 张回单适合先跑通流程，
本版本的训练、ONNX contract 和验收都包含收款方。收款方包含自由中文姓名/商户名，因此 1000 张
只能先产生候选模型；是否可替代当前 Paddle，必须由下面的逐字段保留集评估决定。

构建完成后，先确认没有读取错误、并且每个准备训练的字段都有足够样本；命令会把相同统计也打印出来：

```powershell
$summary = Get-Content "D:\download\ReceiptOcrPseudoV1\dataset_config.json" | ConvertFrom-Json
$summary.counts
```

要决定能否加入收款方，先查看 train 中最少出现的收款方字符；`Count=1` 的字通常还需要补更多样本，
完全没出现的目标字则无法由当前模型输出：

```powershell
$coverage = Get-Content "D:\download\ReceiptOcrPseudoV1\character_coverage.json" | ConvertFrom-Json
$coverage.recipient_field.train.characters.psobject.Properties |
  ForEach-Object { [PSCustomObject]@{ Character = $_.Name; Count = [int]$_.Value } } |
  Sort-Object Count, Character |
  Select-Object -First 50
```

训练会强制要求每个 `--fields` 字段都至少有一条 train 和 val 样本；否则不会生成一个“实际上没学过
某字段”的 ONNX。为了防止验证集字符泄漏，字符表只从 train 生成；若报出验证集有未见字符，不要忽略。
这通常只会发生在你手工编辑 JSONL 后；请用一个新的空输出目录重建伪标签切分，或增加该字符的
人工确认训练样本。
构建器默认会将包含这类字符的**整笔验证回单**移入 train，保持同组不泄漏；test 回单绝不因此移动，
其 OOV 统计会在 ONNX 验收中如实报告。

训练机器需先安装对应 CPU/CUDA 的 PyTorch wheel（不要让普通 PyPI 覆盖已验证的 CUDA wheel），然后：

```powershell
python -m pip install -r requirements-train-ocr.txt

python -m transfer_receipt_ai.ocr_train `
  --records "D:\download\ReceiptOcrPseudoV1\pseudo_labels.jsonl" `
  --output "D:\download\ReceiptOcrCtcV1" `
  --fields "amount,time,transfer_status,recipient_field,payment_method_field" `
  --device cuda:0 `
  --epochs 30 `
  --batch-size 32 `
  --onnx-output ".\artifacts\receipt_ocr_ctc_v1.onnx"
```

`requirements-train-ocr.txt` 只安装 ONNX 导出所需的 `onnx`，不会触碰运行环境已有的
`onnxruntime-gpu`。若要在单独的 CPU 环境执行 ONNX Runtime 对齐检查，再额外安装
`onnxruntime`；CUDA 环境只保留与 CUDA 版本匹配的 `onnxruntime-gpu`，不要同时安装两者。

这会写出 `best.pt`、`last.pt`、训练历史，以及：

```text
artifacts\receipt_ocr_ctc_v1.onnx
artifacts\receipt_ocr_ctc_v1.charset.json
artifacts\receipt_ocr_ctc_v1.contract.json
```

模型固定输入为灰度白底 letterbox 的 `[1, 1, 48, 768]`，输出为 CTC logits；字符表和 contract
必须与 ONNX 一起交付。当前 .NET CLI 接入的是冻结的 PP-OCR `det + cls + rec` 交付包，**尚未接入这一份
自训练 CTC ONNX**；只有当该候选通过下方严格验收后，才应单独接入并替换 Paddle OCR 阶段。

训练命令只会用 train/val；`--test-ratio 0.10` 留出的 test 不参与训练或调参。导出后必须运行
ONNX 实际推理，与同一张字段裁图的 Paddle OCR 结果逐项对比：

```powershell
python -m transfer_receipt_ai.ocr_evaluate `
  --model ".\artifacts\receipt_ocr_ctc_v1.onnx" `
  --records "D:\download\ReceiptOcrPseudoV1\pseudo_labels.jsonl" `
  --split test `
  --output "D:\download\ReceiptOcrEvalV1" `
  --fields "amount,time,transfer_status,recipient_field,payment_method_field" `
  --device cuda:0 `
  --min-raw-exact-match 0.99 `
  --min-semantic-exact-match 0.99 `
  --max-micro-cer 0.005 `
  --max-oov-reference-rate 0
```

该命令会对每个字段（包括 `recipient_field`）分别执行验收；任一字段没有达到门槛则返回失败，但仍写出
`summary.json`、`comparisons.jsonl` 和 `disagreements.jsonl` 供查看。它会同时报告模型字符表外的
测试字符、训练中见过/未见过的文本，以及 ONNX 实际 provider 和延迟。这个比较证明的是“对保留的
Paddle 输出的一致性”；要证明真实业务准确率仍需一批人工标注的独立回单。

## 单模型轻量字段识别 v8（当前实验路线；v3–v7 只保留兼容）

最终交付目标不是四个小模型，而是 **一个 OCR ONNX、一个 ONNX Runtime Session、每张回单一次 `Run`**。
检测器 `receipt_lrcnn_v1.onnx` 和设备识别器仍是独立模型；本节只替代字段 OCR。

v4/v5 的 1 万张试验已经证明：把金额、时间和卡尾号都当作普通整行 CTC 不够安全。例如 CTC 可以把
`900.00` 识别成 `300.00`，同时给出很高的 token confidence。v8 保留 CTC 作为教师一致性候选，
并把金额显示样式、时间模板和付款方式银行规则写进**同一张 ONNX 图**的轻量有限分类头。金额数字仍完全由
CTC 决定；样式头只能在高置信时恢复 `¥`、千分位逗号和负号，绝不能编造或改写数字。

```text
field_images [4,1,80,512]
  0=amount / 1=time / 2=transfer_status / 3=payment_method_field
        ↓
共享轻量 CNN 主干 + 每字段轻量 CTC / 校验 GRU
  ├─ amount: 规范金额 CTC（例如 -1234.56）+ 货币/千分位/负号位置三项有限样式头
  ├─ time: 可见 CTC（:、-、空格）+ 时钟/日期时间模板与固定数字位
  ├─ payment: 原文 CTC + 银行前缀分类 + 四位卡尾/括号样式
  └─ transfer_status: 三分类（类别覆盖不完整时固定为 review）
```

它仍是一个模型文件、一个输入和一次推理；v8 保持 14 个输出节点：前四个为 CTC/status，后十个为同一图内的
金额样式、时间和付款方式辅助头，不是额外模型或额外 Session。金额、时间和付款方式裁图在 v8 统一右对齐
白底 letterbox；交付端必须严格复用 sidecar contract 的 `[4,1,H,W]` slot 顺序和预处理。

v8 的时间格式头直接复用时间 CTC 的 GRU 状态：逐字符 CTC 损失会同时训练数字与 `:`/`-`/空格等细小符号，
避免 v6 中独立格式分支在右对齐空白前缀下学不到冒号的问题。金额显示头同样复用金额 CTC 的状态；任一样式
头低于 `--amount-format-min-confidence` 时，运行时只保留规范 CTC 候选。旧 v3–v7 checkpoint 仍按原路径
加载和导出，不会被新代码改写。

v8 的文本字段目前仍是**固定 review-only**：CTC 与格式/银行校验头即使渲染出相同可见文本，也只作为
诊断候选，不会自动交付。训练标签仍来自同一 Paddle 伪标签；二者
不一致、金额不符合严格 CNY 格式、时间不符合已知时钟/日期模板、或银行前缀为 `__other__` 时都必须返回
`review`。CTC 和结构化候选会写进评估报告；v6/v7/v8 的时间主候选优先使用有效的结构化模板，例如原始 CTC
`0240` 会显示为 `02:40`，但原始 CTC 仍保留在 JSON 中供诊断。v8 的金额主候选也只会用高置信样式渲染
规范 CTC 数字。这只改变评估报告的候选展示，不改变任何
已生成 ONNX 的权重、输入或输出协议。只有在按回单分组隔离的人工真值集上
证明“零错误放行 + 非零覆盖率”后，才可以把文本改为自动交付。

金额严格接受 `99.99`、`¥1,234.56`、`-¥1,234.56`、`¥-1,234.56` 等明确形式；错误千分位、错误小数位、
重复负号与负零不进入结构化监督。时间仅接受有效 `H:MM` / `HH:MM` / `HH:MM:SS` 和
`YYYY-MM-DD HH:MM(:SS)`；日期会进行真实日历校验。付款方式的银行分类词表**仅从 train split 生成**；
低频或未见银行映射为 `__other__`，不允许由 val/test 泄漏进类表，也不能被自动放行。

v5 有意不包含自由中文 `recipient_field`：此前小识别器的收款方精度不合格；要接近 Paddle 的任意中文
原文，需要独立宽字符模型和更多人工真值。付款方式仍以画面可见的原文训练，例如
`建设银行储蓄卡（3667）`；不能把 `bank_card` / `balance` 这类归一化类别当成原文 OCR 标签。

已生成的 v3–v7 checkpoint、ONNX 和 sidecar 仍可被 `export`、`evaluate` 加载，便于历史回归；但不能与
v8 sidecar 或权重混用。新的训练一律显式使用 `--architecture v8`。

### 1. 将已经生成的 Paddle 教师标签聚合为“一张回单一行” manifest

这一步不训练、不复制图片、也不再调用 Paddle。它只把平铺的四类字段裁图按同一张回单组合，并隔离
train/val/test；同一字段出现相互矛盾的教师文本时，会移出训练而不是凭置信度强行选一个。

```powershell
$outputRoot = "D:\alipay-ai-data\receipt-lite-teacher-10k-v1"
$teacherLabels = "$outputRoot\paddle-teacher-labels"
$unifiedManifest = "$outputRoot\unified-manifest-v7"

python -m transfer_receipt_ai.ocr_unified_dataset `
  --records "$teacherLabels\pseudo_labels.jsonl" `
  --output $unifiedManifest
```

成功后查看 `$unifiedManifest\dataset.contract.json`。除四个 slot 的 train/val/test 数量、完整回单数和
冲突裁图数外，manifest 还会写出 `structured_target_counts`：旧兼容目标以及 `amount_display`、`time_display`、
`payment_bank_prefix` 和各自未解析数量。训练器会再次检查 train/val 中是否同时有严格可见金额、严格可见时间
和精确银行前缀；缺任何一种会拒绝训练，而不是静默降级成旧 CTC。

现有的 v7 manifest 已经包含 v8 所需的 `visible_text` 与规范金额信息，因此升级 v8 **不需要重新运行 Paddle、
不需要重新裁图，也不需要人工标注**：直接用既有 `unified_fields.jsonl` 训练即可。整个蒸馏循环仍是 Paddle
伪标签，不能把 teacher-parity 当作人工业务真值。

### 2. 训练并直接导出一个 ONNX

训练阶段可以使用 PyTorch/CUDA；运行交付物只包含 ONNX 和两个 JSON sidecar，不包含 Paddle、Python 或网络依赖。
`--dataset-root` 必须指向教师标签目录，因为 manifest 不复制 `images/`。
若训练环境尚未有 ONNX 导出依赖，先执行 `python -m pip install -r requirements-train-ocr.txt`；保持已有的
`onnxruntime-gpu`，不要再额外安装 CPU 版 `onnxruntime`。导出会用已安装的 ONNX Runtime 做一次 CPU 图加载和
Torch/ORT 输出对齐验证。

```powershell
$unifiedRun = "$outputRoot\unified-run-v8"
$unifiedModel = "$outputRoot\models\receipt_unified_field_reader_v8_120k.onnx"

python -m transfer_receipt_ai.ocr_unified train `
  --records "$unifiedManifest\unified_fields.jsonl" `
  --dataset-root $teacherLabels `
  --output $unifiedRun `
  --device cuda:0 `
  --architecture v8 `
  --image-height 80 `
  --image-width 512 `
  --base-channels 32 `
  --numeric-hidden-size 96 `
  --payment-hidden-size 128 `
  --epochs 80 `
  --batch-size 24 `
  --ctc-loss-weight 0.35 `
  --structured-loss-weight 1.0 `
  --amount-format-min-confidence 0.90 `
  --payment-bank-prefix-min-support 3 `
  --num-workers 0 `
  --onnx-output $unifiedModel
```

输出包括：

- `$unifiedRun\best.pt`：只用于后续再导出；不是交付物；
- `$unifiedModel`：一个固定输入 `[4,1,80,512]`、一次推理返回 14 个 v8 输出的 ONNX；
- `$unifiedModel` 同名的 `.labels.json` 和 `.contract.json`：字符表、slot 顺序、SHA-256、预处理和输出协议，必须与 ONNX 一起交付。

v8 的 `best.pt` 按验证集实际候选文本（金额/时间/付款方式）的宏平均 teacher-parity 优先选择，随后才比较
结构辅助头和 loss，避免校验头看似提高但最终候选文本退化。它仍是教师标签上的训练选择，不是业务上线验收。

检查 `training_summary.json` 的 `status_head_policy`。若 `runtime_policy` 为 `review_only`（例如当前 1 万张中
状态全是 `success`），训练会主动关闭状态损失以免浪费共享主干容量；ONNX 仍保留稳定的 `status_logits` 输出协议，
但交付运行时必须把 `transfer_status` 返回为 `review`，绝不能把全成功数据伪装成安全的状态分类器。只有
`success`、`pending`、`failed` 都在 train / val / test 出现时，`runtime_policy=classify` 才允许使用状态头。

### 3. 用 ONNX 实际运行做保留集 teacher-parity 验收

这是 CPU 验证命令，确保最终没有 GPU 时也能跑；若只验证 GPU，将 `--device cpu` 改成 `cuda:0`。每张回单
只调用一次 ONNX Session，然后同时报告“候选 teacher-parity”与“实际可交付 delivery”指标；状态头只有在
delivery contract 的 `status_head_policy.runtime_policy=classify` 时才可参与状态及 `pending/failed -> success`
的验收。

```powershell
python -m transfer_receipt_ai.ocr_unified evaluate `
  --model $unifiedModel `
  --records "$unifiedManifest\unified_fields.jsonl" `
  --dataset-root $teacherLabels `
  --split test `
  --output "$outputRoot\unified-eval-v8-cpu" `
  --device cpu `
  --max-delivery-false-accepts 0
```

先阅读 `summary.json`、`comparisons.jsonl`、`disagreements.jsonl`，尤其是每个字段的
`raw_exact_match`、`delivery_coverage`、`delivery_exact_match`、`delivery_false_accepts` 和付款方式的
`oov_reference_rate`。`raw_exact_match` 只是候选与教师的一致性；只有 delivery 指标才代表 v8 当前会实际
输出的值。只设置 `--max-delivery-false-accepts 0` 的首轮运行也**不能**判定模型合格：若所有结果都变成
`review`，它仍可能是零错误交付但零业务覆盖。

达到预先确认的业务门槛后，再加上下面三个 delivery gate。数值应来自独立人工真值保留集，而不是把未验证的
阈值当作精度承诺；`0` 个错误交付是金额/时间/卡尾号的最低安全要求。

```powershell
  --min-delivery-coverage <已确认的覆盖率门槛> `
  --min-delivery-exact-match <已确认的交付准确率门槛> `
  --max-delivery-false-accepts 0
```

v8 可以直接在已生成的 12 万张 manifest 上训练和评估；教师一致性并不等于真实业务准确率。

### 可选人工复核候选差异（不修改原图）

当 `comparisons.jsonl` 中出现 Paddle 参考文字与学生候选不一致时，不要直接认定任意一方为真。下面的
本地页面会显示**字段裁图**，可用键盘快速选择 Paddle、学生候选、手工编辑真实文字，或标记不确定。它只
绑定 `127.0.0.1`，不访问网络；每一次选择都会原子写入 JSONL，因此可随时关闭、重新运行后继续。

```powershell
$eval = "$outputRoot\unified-eval-v8-cpu"

python -m transfer_receipt_ai.ocr_review `
  --input "$eval\comparisons.jsonl" `
  --output "$eval\manual-truth.jsonl"
```

这一步**不参与自动教师蒸馏**；若目标仅是复现 Paddle，直接跳过并继续使用 `pseudo_labels.jsonl` 即可。
浏览器会自动打开。快捷键：`1`=Paddle 参考标签，`2`=学生候选，`E`=编辑真实文字，`Ctrl+Enter`=保存编辑，
`0`=不确定，`B`/`N`=上一张/下一张。输出中的 `truth_text` 才是后续校准和人工真值验收可使用的标签；
`unclear` 不会被当作训练真值。当前这批差异来自 `test` 保留集，确认后的 `manual-truth.jsonl` 只能用于
验收/校准，不能回灌到下一轮训练。若浏览器没有自动打开，终端会打印形如 `http://127.0.0.1:8765/` 的本地地址。

### 历史分模型 v2（仅保留为对照，不是本次交付路线）

旧版 `receipt_ocr_ctc_v1` 把金额、时间、状态、付款方式和自由中文收款方放进同一个 CTC 字符表，
不能作为小模型交付路线。早期 v2 改成分别建模：

| 字段 | v2 模型 | 输出策略 |
| --- | --- | --- |
| `amount` | 数字 CTC | 仅输出合法金额；无效或低置信进入 `review` |
| `time` | 数字/冒号 CTC | 仅输出合法时钟时间 |
| `transfer_status` | 三分类 | `success` / `pending` / `failed`；不确定进入 `review` |
| `payment_method_field` | 五分类 | `bank_card` / `balance` / `yuebao` / `huabei` / `other` |
| `recipient_field` | 本地交易索引优先；可选已知对象分类 | 未命中统一为 `unknown/review`，首版不伪造任意中文原文 |

它的训练、导出和运行都不导入 Paddle、不访问网络。最推荐的标签来源是本地交易真值表，而不是历史
OCR 伪标签：12 万原图只有在能关联真实交易字段时，才是高质量监督数据。

### 1. 只跑检测，生成本地训练所需的几何 JSON

下面的命令使用既有收据/设备 ONNX，不执行 OCR，也不写标注图片。第一轮只跑 **1000 张**；确认
`receipt_key` 关联率、字段漏检率和裁图无误后，再移除 `--limit 1000` 跑全量。`--continue-on-error`
会把坏图保留在错误报告中而不中断批次。

```powershell
$project = "D:\alipay-ai-data\alipay-ai-inference"
$rawImages = "D:\download\OriginalImages120K"
$detectorResults = "D:\download\ReceiptDetectorNoOcr120K"

python -m receipt_inference.cli `
  --onnx-model "$project\artifacts\receipt_lrcnn_v1.onnx" `
  --platform-onnx-model "$project\artifacts\statusbar_device_v1.onnx" `
  --input $rawImages `
  --output $detectorResults `
  --device cpu `
  --ocr none `
  --annotate none `
  --require-complete `
  --continue-on-error `
  --limit 1000
```

这里必须使用 Python `receipt_inference.cli` 的结果目录：其中保存了透视矫正几何，后续才能从原图重建
同样的字段裁图。当前 ML.NET CLI 的 JSON 不含这套几何，不能直接作为这一步的 `--results`。

### 2. 准备离线交易真值 JSONL

真值文件每行一个 JSON，对应的 `receipt_key` 必须能由原图文件名抽出。当前 `s3_voucher_GWCZ...png`
可使用默认正则；金额可提供 `amount`（如 `100.00`）或整数分 `amount_fen`。类别可使用中文原文，也可
直接使用规范值。

```json
{"receipt_key":"GWCZ2071991511234514944","amount_fen":10000,"time":"12:06","transfer_status":"success","payment_method":"bank_card","recipient":"交易商家"}
```

请保留一部分完整回单作为独立 test；不要把同一 `receipt_key` 或同一张回单的多次截图分到不同 split。
构建器会按照收据组稳定切分 train / val / test：

```powershell
$truth = "D:\secure-data\receipt_truth_120k.jsonl"
$truthDataset = "D:\download\ReceiptTruthDataset120K"

python -m transfer_receipt_ai.ocr_truth_dataset `
  --results $detectorResults `
  --truth $truth `
  --output $truthDataset `
  --source-key-regex '(?P<key>GWCZ[0-9A-Za-z]+)' `
  --fields amount,time,transfer_status,payment_method_field `
  --min-detector-score 0.80 `
  --validation-ratio 0.10 `
  --test-ratio 0.10 `
  --continue-on-error
```

输出的 `$truthDataset\pseudo_labels.jsonl` 名称为兼容旧工具而保留；其中每条记录的
`label_source` 是 `transaction_truth`，并不是 Paddle 标签。`rejected.jsonl` 会记录缺少真值键、
漏检字段或坏裁图；先检查它再训练。构建器默认拒绝“原图修改时间晚于检测 JSON”的情况，以免拿旧框裁
新图；只有审计过变化时才加 `--allow-source-newer`。构建过程会先写入临时同级目录，全部完成后才原子发布
`$truthDataset`，因此中断不会留下看似可用的半成品。

### 3. 从真值裁图创建轻量任务

该步骤不复制 12 万张裁图，只在 `$truthDataset` 上生成轻量 manifest。收款方只保留训练集出现次数
达到门槛的前 N 个已知对象；其余一律映射到 `unknown`，避免长尾中文把模型和字符表重新做大。

```powershell
$lite = "D:\download\ReceiptLiteV2"

python -m transfer_receipt_ai.ocr_lite_dataset `
  --records "$truthDataset\pseudo_labels.jsonl" `
  --output $lite `
  --recipient-top-k 200 `
  --recipient-min-train-count 25 `
  --recipient-unknown-to-known-ratio 2.0
```

会生成 `amount_ctc.jsonl`、`time_ctc.jsonl`、状态/付款方式/收款方分类 manifest、
`recipient_catalog.json` 和 `dataset.contract.json`。初版建议只训练前四项；收款方优先由本地交易索引
返回，未命中即 `unknown/review`。

### 4. 训练并导出小 ONNX

先安装与训练机 CUDA 匹配的 PyTorch，再安装导出依赖。以下是 GPU 示例；将 `--device cuda:0` 改成
`--device cpu` 即可用 CPU 训练（会慢很多）。

```powershell
python -m pip install -r requirements-train-ocr.txt

python -m transfer_receipt_ai.ocr_train `
  --records "$lite\amount_ctc.jsonl" `
  --dataset-root $truthDataset `
  --fields amount `
  --output "D:\download\ReceiptLiteAmountRun" `
  --device cuda:0 `
  --image-height 32 `
  --image-width 256 `
  --base-channels 16 `
  --hidden-size 64 `
  --lstm-layers 1 `
  --epochs 30 `
  --batch-size 128 `
  --onnx-output "$project\artifacts\receipt_lite_amount_v2.onnx"

python -m transfer_receipt_ai.ocr_train `
  --records "$lite\time_ctc.jsonl" `
  --dataset-root $truthDataset `
  --fields time `
  --output "D:\download\ReceiptLiteTimeRun" `
  --device cuda:0 `
  --image-height 32 `
  --image-width 192 `
  --base-channels 16 `
  --hidden-size 64 `
  --lstm-layers 1 `
  --epochs 30 `
  --batch-size 128 `
  --onnx-output "$project\artifacts\receipt_lite_time_v2.onnx"

python -m transfer_receipt_ai.ocr_lite_classifier train `
  --records "$lite\transfer_status_classifier.jsonl" `
  --dataset-root $truthDataset `
  --output "D:\download\ReceiptLiteStatusRun" `
  --device cuda:0 `
  --epochs 20 `
  --batch-size 128 `
  --onnx-output "$project\artifacts\receipt_lite_status_v2.onnx"

python -m transfer_receipt_ai.ocr_lite_classifier train `
  --records "$lite\payment_method_classifier.jsonl" `
  --dataset-root $truthDataset `
  --output "D:\download\ReceiptLitePaymentRun" `
  --device cuda:0 `
  --epochs 20 `
  --batch-size 128 `
  --onnx-output "$project\artifacts\receipt_lite_payment_v2.onnx"
```

上面的 `pip install` 只发生在训练/导出准备阶段；最终 ONNX 推理包不含 Paddle，也不需要网络。若训练机本身
也不允许联网，请由受批准的机器预先准备 wheel 目录，再离线安装，例如
`python -m pip install --no-index --find-links D:\offline-wheels -r requirements-train-ocr.txt`；PyTorch/TorchVision
同样应从对应 CUDA 的离线 wheel 安装，而不是在线临时换版本。

每个分类器会输出 `.onnx`、`.labels.json` 与 `.contract.json`；数值 CTC 会输出 `.onnx`、
`.charset.json` 与 `.contract.json`。三份文件都必须一起交付。

### 5. 先用独立 test 验收，再接入 .NET

以下命令的 reference 是本地交易真值，不是 Paddle。金额和时间必须先达到验收门槛；状态重点检查
`failed/pending -> success` 的混淆记录，低置信度在运行时应进入 `review`，而不是猜成成功。

```powershell
python -m transfer_receipt_ai.ocr_evaluate `
  --model "$project\artifacts\receipt_lite_amount_v2.onnx" `
  --records "$lite\amount_ctc.jsonl" `
  --dataset-root $truthDataset `
  --split test `
  --output "D:\download\ReceiptLiteAmountEval" `
  --fields amount `
  --device cpu `
  --min-semantic-exact-match 0.995 `
  --max-micro-cer 0.005 `
  --max-oov-reference-rate 0

python -m transfer_receipt_ai.ocr_evaluate `
  --model "$project\artifacts\receipt_lite_time_v2.onnx" `
  --records "$lite\time_ctc.jsonl" `
  --dataset-root $truthDataset `
  --split test `
  --output "D:\download\ReceiptLiteTimeEval" `
  --fields time `
  --device cpu `
  --min-semantic-exact-match 0.995 `
  --max-micro-cer 0.005 `
  --max-oov-reference-rate 0

python -m transfer_receipt_ai.ocr_lite_classifier evaluate `
  --model "$project\artifacts\receipt_lite_status_v2.onnx" `
  --records "$lite\transfer_status_classifier.jsonl" `
  --dataset-root $truthDataset `
  --split test `
  --output "D:\download\ReceiptLiteStatusEval" `
  --device cpu `
  --min-accuracy 0.99 `
  --min-macro-recall 0.98 `
  --min-confidence 0.99 `
  --max-non-success-to-success 0

python -m transfer_receipt_ai.ocr_lite_classifier evaluate `
  --model "$project\artifacts\receipt_lite_payment_v2.onnx" `
  --records "$lite\payment_method_classifier.jsonl" `
  --dataset-root $truthDataset `
  --split test `
  --output "D:\download\ReceiptLitePaymentEval" `
  --device cpu `
  --min-accuracy 0.99 `
  --min-macro-recall 0.98 `
  --min-confidence 0.99
```

当前 .NET CLI 尚未加载这些 v2 小模型；先以以上 CPU test 的真实精度、p50/p95、ONNX 文件大小为准。
通过后再新增 `--ocr lite`，不会改坏既有 `--ocr onnx` 的 PP-OCR 命令。

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
contract JSON 交付给 .NET。

## .NET / ML.NET ONNX 命令

[`dotnet/ReceiptMlNet.Cli`](dotnet/ReceiptMlNet.Cli) 是独立于 Python 的 .NET 8 控制台项目，当前 OCR
运行时交付目标是 **Windows x64**（OpenCvSharp Windows 原生库）。
它实际使用 ML.NET 的 `ApplyOnnxModel` 加载当前 ONNX：主检测模型输出动态长度的
`boxes / labels / scores`，设备模型输出 Android/iOS 概率。它会校验两个 `.contract.json` 的
模型 SHA-256，防止模型和交付说明混用。传入 `--ocr onnx --ocr-bundle <交付目录>` 后，它还会
直接用 ONNX Runtime 加载冻结的 PP-OCR `det + cls + rec` 三个模型；OCR 不依赖 Python、Paddle
或网络下载。OCR 交付目录必须是前文 `package-delivery` 产生的目录，CLI 会校验三份 ONNX、字符表和
`paddle_ocr_delivery.contract.json` 的 SHA-256。

上面的 `--ocr onnx --ocr-bundle` 仅适用于冻结的 PP-OCR bundle，**不能**把单模型
`receipt_unified_field_reader_v3`、`receipt_unified_field_reader_v4` 或
`receipt_unified_field_reader_v5`、`receipt_unified_field_reader_v6` 或 `receipt_unified_field_reader_v7` 传给它。统一 reader 目前仍未接入 .NET CLI；只有 v7 同时通过 Python 的
保留集 delivery gate 和人工真值验收后，才会以独立的 `--ocr unified` 路径接入，并复用同一份 `[4,1,H,W]`
预处理、slot 顺序、结构化解码规则和 `status_head_policy`。

该项目覆盖**模型层**：EXIF 摆正、letterbox、检测框坐标还原、阈值/每类最佳框、设备识别规则，
并写出 JSON / manifest / 标注 JPG。默认 `--annotate all`，每张会输出与 Python 相同命名的
`*_rectified_annotated.jpg` 和 `*_original_annotated.jpg`，使用同一套字段颜色、椭圆框、分数侧栏
与设备识别红色行；还可传 `--annotate flagged`（只标缺核心字段的图）或 `--annotate none`。

启用 OCR 时，每个检测字段会按 Python 相同的 8% 边距裁图，执行完整 DB 文本检测、方向分类、CTC
识别，并写入 `detections[].ocr` 与 `fields.time/amount/transfer_status/recipient/payment_method`；标注图侧栏
也会显示 OCR 文本。OCR 的 `det/cls/rec` Session 每批只创建一次。

目前仍未移植 OpenCV 透视矫正、单应矩阵回投。因此输入需要是已纠正的回单图/截图；两张兼容命名的 JPG
仍基于 EXIF 摆正后的输入坐标绘制，内容相同。照片类原图要获得与 Python 完全相同的几何结果，需要后续
移植矫正模块。

安装 [.NET 8 SDK](https://dotnet.microsoft.com/download/dotnet/8.0) 后，CPU 单图验证：

```powershell
cd D:\alipay-ai-data\alipay-ai-inference
dotnet restore .\dotnet\ReceiptMlNet.Cli\ReceiptMlNet.Cli.csproj
$ocrBundle = ".\artifacts\paddle_ocr_ppocrv4_v1_delivery"

dotnet build .\dotnet\ReceiptMlNet.Cli\ReceiptMlNet.Cli.csproj -p:OnnxRuntimeFlavor=cpu

dotnet run --no-build --project .\dotnet\ReceiptMlNet.Cli\ReceiptMlNet.Cli.csproj -- `
  --detector ".\artifacts\receipt_lrcnn_v1.onnx" `
  --device-model ".\artifacts\statusbar_device_v1.onnx" `
  --ocr onnx `
  --ocr-bundle $ocrBundle `
  --input "D:\download\TempFakeImages\s3_voucher_GWCZ2071991511234514944_20260701001815.png" `
  --output "D:\download\TempFakeResults_mlnet_onnx_ocr_cpu_1" `
  --device cpu `
  --annotate all `
  --require-complete
```

CUDA 机器构建 GPU Runtime 并强制 GPU 验证：

`Microsoft.ML.OnnxRuntime.Gpu` 自带 ONNX Runtime CUDA Provider，但不自带 NVIDIA
CUDA/cuDNN 的运行时 DLL。运行 .NET 命令前，Windows 需要在 `PATH` 中提供兼容的
CUDA 12.x 与 cuDNN 9.x。若当前 Python 虚拟环境使用 CUDA 12 的 PyTorch，可先在**同一个
PowerShell** 中执行下列命令，让 `dotnet` 子进程复用 PyTorch 的 DLL：

```powershell
$torchLib = python -c "import os, torch; print(os.path.join(os.path.dirname(torch.__file__), 'lib'))"
Test-Path "$torchLib\cublasLt64_12.dll"
Test-Path "$torchLib\cudnn64_9.dll"
$env:Path = "$torchLib;$env:Path"
```

前两个检查都应为 `True`；该 PATH 设置只对当前终端有效。正式的纯 .NET 交付应安装 Windows
x64 的 CUDA 12.x 与 cuDNN 9.x，并把两者的 `bin` 目录写入系统 PATH，不能只复制零散 DLL
到项目目录。

```powershell
dotnet build .\dotnet\ReceiptMlNet.Cli\ReceiptMlNet.Cli.csproj -p:OnnxRuntimeFlavor=gpu

dotnet run --no-build --project .\dotnet\ReceiptMlNet.Cli\ReceiptMlNet.Cli.csproj -p:OnnxRuntimeFlavor=gpu -- `
  --detector ".\artifacts\receipt_lrcnn_v1.onnx" `
  --device-model ".\artifacts\statusbar_device_v1.onnx" `
  --ocr onnx `
  --ocr-bundle $ocrBundle `
  --input "D:\download\TempFakeImages\s3_voucher_GWCZ2071991511234514944_20260701001815.png" `
  --output "D:\download\TempFakeResults_mlnet_onnx_ocr_gpu_1" `
  --device cuda:0 `
  --annotate all `
  --require-complete
```

`--device cuda:0` 是严格 GPU 验证：回单检测、设备识别和 OCR 的 det/cls/rec 三个 Session 都会请求
同一块 CUDA 设备；任一个 CUDA Session 不能创建时，该命令会失败。ONNX Runtime 可能将少量 shape/metadata
节点保留在 CPU，这是正常的，不影响神经网络计算优先走 GPU。`--device auto` 则允许整套 OCR 回退 CPU，不应用作
GPU 验收。

不要混用 CPU/GPU 的输出目录。下面命名中的最后一段是本次验证上限：`_1`、`_100`、`_10000`。

批量 100 张 CPU 验证：

```powershell
$batchInput = "D:\download\TempFakeImages"
$batchOutputCpu = "D:\download\TempFakeResults_mlnet_onnx_ocr_cpu_100"

dotnet build .\dotnet\ReceiptMlNet.Cli\ReceiptMlNet.Cli.csproj -p:OnnxRuntimeFlavor=cpu

dotnet run --no-build --project .\dotnet\ReceiptMlNet.Cli\ReceiptMlNet.Cli.csproj -p:OnnxRuntimeFlavor=cpu -- `
  --detector ".\artifacts\receipt_lrcnn_v1.onnx" `
  --device-model ".\artifacts\statusbar_device_v1.onnx" `
  --ocr onnx `
  --ocr-bundle $ocrBundle `
  --input $batchInput `
  --output $batchOutputCpu `
  --device cpu `
  --annotate all `
  --require-complete `
  --continue-on-error `
  --limit 100
```

批量 100 张 GPU 验证：

```powershell
dotnet run --no-build --project .\dotnet\ReceiptMlNet.Cli\ReceiptMlNet.Cli.csproj -p:OnnxRuntimeFlavor=gpu -- `
  --detector ".\artifacts\receipt_lrcnn_v1.onnx" `
  --device-model ".\artifacts\statusbar_device_v1.onnx" `
  --ocr onnx `
  --ocr-bundle $ocrBundle `
  --input "D:\download\TempFakeImages" `
  --output "D:\download\TempFakeResults_mlnet_onnx_ocr_gpu_100" `
  --device cuda:0 `
  --annotate all `
  --require-complete `
  --continue-on-error `
  --limit 100
```

10,000 张正式 CPU / GPU 回归只改输出目录与执行设备；每次先构建匹配的 Runtime flavor：

```powershell
$batchInput = "D:\download\TempFakeImages"
$batchOutputCpu = "D:\download\TempFakeResults_mlnet_onnx_ocr_cpu_10000"

dotnet build .\dotnet\ReceiptMlNet.Cli\ReceiptMlNet.Cli.csproj -p:OnnxRuntimeFlavor=cpu
dotnet run --no-build --project .\dotnet\ReceiptMlNet.Cli\ReceiptMlNet.Cli.csproj -p:OnnxRuntimeFlavor=cpu -- `
  --detector ".\artifacts\receipt_lrcnn_v1.onnx" `
  --device-model ".\artifacts\statusbar_device_v1.onnx" `
  --ocr onnx `
  --ocr-bundle $ocrBundle `
  --input $batchInput `
  --output $batchOutputCpu `
  --device cpu `
  --annotate all `
  --require-complete `
  --continue-on-error `
  --limit 10000
```

```powershell
$batchInput = "D:\download\TempFakeImages"
$batchOutputGpu = "D:\download\TempFakeResults_mlnet_onnx_ocr_gpu_10000"

dotnet build .\dotnet\ReceiptMlNet.Cli\ReceiptMlNet.Cli.csproj -p:OnnxRuntimeFlavor=gpu
dotnet run --no-build --project .\dotnet\ReceiptMlNet.Cli\ReceiptMlNet.Cli.csproj -p:OnnxRuntimeFlavor=gpu -- `
  --detector ".\artifacts\receipt_lrcnn_v1.onnx" `
  --device-model ".\artifacts\statusbar_device_v1.onnx" `
  --ocr onnx `
  --ocr-bundle $ocrBundle `
  --input $batchInput `
  --output $batchOutputGpu `
  --device cuda:0 `
  --annotate all `
  --require-complete `
  --continue-on-error `
  --limit 10000
```

中断续跑时，在对应命令末尾添加 `--skip-existing`；如果旧 JSON 没有 `fields`，在 `--ocr onnx` 模式下会
自动重跑而不会误跳过。CPU 与 GPU 输出的 `inference_manifest.json` / `inference_errors.jsonl` 可分别用于
统计两套验证的成功数、错误数和耗时。

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
