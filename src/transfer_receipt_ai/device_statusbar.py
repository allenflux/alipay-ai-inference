"""状态栏设备识别(安卓/苹果)—— 合并进本流水线,替代基于对号的 status_style 设备判定。

方法:分辨率规则(零解码,~78% 直接判)判不了,再看**状态栏**(顶部整条,极小 CNN);命中分辨率
也顺手核一眼状态栏,发现"把安卓图缩放到 iPhone 精确分辨率冒充苹果"的伪造(device_prior_conflict)。
看的是**原图 source_rgb**(状态栏在最顶上,不是矫正后的裁剪)。自包含,只依赖 torch/numpy/PIL。

用法:
    from .device_statusbar import StatusBarDeviceClassifier
    clf = StatusBarDeviceClassifier("checkpoints/statusbar_device_v1/best.pt", device="cuda")
    d = clf.classify(source_rgb)   # {platform, platform_cn, confidence, source, device_prior_conflict, ...}
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

import numpy as np
import torch
from PIL import Image
from torch import nn

# ---- 分辨率规则(与 platform-classifier 一致;出新机型要在这里更新,漂移监控会预警)----
IPHONE_RESOLUTIONS: Final[frozenset[tuple[int, int]]] = frozenset({
    (640, 960), (640, 1136), (750, 1334), (828, 1792), (960, 640), (1125, 2436),
    (1136, 640), (1170, 2532), (1179, 2556), (1206, 2622), (1242, 2208), (1242, 2688),
    (1284, 2778), (1290, 2796), (1320, 2868), (1334, 750), (1792, 828), (2208, 1242),
    (2436, 1125), (2532, 1170), (2556, 1179), (2622, 1206), (2688, 1242), (2778, 1284),
    (2796, 1290), (2868, 1320),
})
ANDROID_PANEL_WIDTHS: Final[frozenset[int]] = frozenset({720, 1080, 1440})


def resolution_platform(width: int, height: int) -> str:
    """仅凭分辨率给高精度判定:'ios' | 'android' | 'abstain'。"""
    if width <= 0 or height <= 0:
        return "abstain"
    if (width, height) in IPHONE_RESOLUTIONS:
        return "ios"
    if min(width, height) in ANDROID_PANEL_WIDTHS:
        return "android"
    return "abstain"


def device_prior_conflict(res_plat: str, cnn_dev: str, cnn_conf: float, *, min_conf: float = 0.8) -> bool:
    """分辨率给了明确平台、状态栏 CNN 却高置信判成另一个 -> 矛盾(疑似缩放伪造)。"""
    if res_plat not in ("ios", "android") or cnn_dev not in ("ios", "android"):
        return False
    return cnn_dev != res_plat and cnn_conf >= min_conf


# ---- 预处理(与训练逐字节一致:顶部 8% 整条 → 512x64 → ImageNet 归一化)----
_STRIP_FRACTION: Final[float] = 0.08
_CANVAS_W: Final[int] = 512
_CANVAS_H: Final[int] = 64
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _preprocess(rgb: np.ndarray) -> np.ndarray:
    h = rgb.shape[0]
    strip = rgb[: max(1, int(round(h * _STRIP_FRACTION)))]
    canvas = np.asarray(Image.fromarray(strip).convert("RGB").resize((_CANVAS_W, _CANVAS_H)))
    a = canvas.astype(np.float32) / 255.0
    a = (a - _MEAN) / _STD
    return np.ascontiguousarray(a.transpose(2, 0, 1))


# ---- 极小 CNN(深度可分离卷积;结构须与 statusbar checkpoint 完全一致)----
class _DWSep(nn.Module):
    def __init__(self, cin: int, cout: int, stride: int = 1) -> None:
        super().__init__()
        self.dw = nn.Conv2d(cin, cin, 3, stride=stride, padding=1, groups=cin, bias=False)
        self.pw = nn.Conv2d(cin, cout, 1, bias=False)
        self.bn = nn.BatchNorm2d(cout)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.pw(self.dw(x))))


class _StatusBarNet(nn.Module):
    def __init__(self, num_classes: int = 2, width: float = 1.0) -> None:
        super().__init__()

        def c(n: int) -> int:
            return max(8, int(round(n * width)))

        self.stem = nn.Sequential(
            nn.Conv2d(3, c(16), 3, stride=2, padding=1, bias=False), nn.BatchNorm2d(c(16)), nn.ReLU(inplace=True)
        )
        self.blocks = nn.Sequential(
            _DWSep(c(16), c(32), 2), _DWSep(c(32), c(48), 2), _DWSep(c(48), c(64), 2), _DWSep(c(64), c(96), 1)
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(0.1)
        self.head = nn.Linear(c(96), num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.blocks(x)
        x = self.pool(x).flatten(1)
        return self.head(self.dropout(x))


DEVICE_CN: Final[dict[str, str]] = {"ios": "苹果", "android": "安卓", "uncertain": "不确定", "unknown": "未知"}


def _resolve_device(device: str) -> str:
    if device in ("auto", ""):
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device.startswith("cuda") and not torch.cuda.is_available():
        return "cpu"
    return device


class StatusBarDeviceClassifier:
    """加载一次、多次 classify。看原图,返回设备判定 dict。"""

    def __init__(self, checkpoint: str | Path, *, device: str = "auto",
                 conf_uncertain: float = 0.75, crosscheck: bool = True) -> None:
        self.device = _resolve_device(device)
        payload = torch.load(checkpoint, map_location=self.device, weights_only=False)
        self.model = _StatusBarNet(2, width=float(payload.get("width", 1.0))).to(self.device)
        self.model.load_state_dict(payload["model_state"])
        self.model.eval()
        self.conf_uncertain = conf_uncertain
        self.crosscheck = crosscheck

    @torch.inference_mode()
    def _p_ios(self, rgb: np.ndarray) -> float:
        x = torch.from_numpy(_preprocess(rgb)).unsqueeze(0).to(self.device)
        return float(torch.softmax(self.model(x), 1)[0, 1])

    def classify(self, source_rgb: np.ndarray) -> dict:
        """source_rgb: 已 EXIF 摆正的原图 HxWx3 RGB(uint8)。返回设备判定 dict。"""
        h, w = source_rgb.shape[:2]
        plat = resolution_platform(w, h)
        if plat in ("ios", "android"):
            out = {"platform": plat, "platform_cn": DEVICE_CN[plat], "source": "resolution",
                   "confidence": 0.99, "device_prior_conflict": False}
            if self.crosscheck:
                pv = self._p_ios(source_rgb)
                cdev = "ios" if pv > 0.5 else "android"
                if device_prior_conflict(plat, cdev, max(pv, 1 - pv)):
                    out.update(confidence=0.5, device_prior_conflict=True, cnn_platform=cdev,
                               conflict_detail=f"分辨率判{plat}、状态栏判{cdev}(疑似缩放伪造)")
            return out
        pv = self._p_ios(source_rgb)
        conf = max(pv, 1 - pv)
        dev = "uncertain" if conf < self.conf_uncertain else ("ios" if pv > 0.5 else "android")
        return {"platform": dev, "platform_cn": DEVICE_CN[dev], "source": "cnn",
                "confidence": round(conf, 3), "p_ios": round(pv, 4), "device_prior_conflict": False}
