"""ONNX Runtime adapters for the deployable receipt detector models.

The receipt detector is exported with one fixed-size RGB image input.  This
module owns the corresponding preprocessing and coordinate restoration so an
ONNX invocation keeps the same ``predict(H×W×3 RGB)`` protocol as the
PyTorch-backed predictor used by the rest of the pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
from PIL import Image

from .labels import ID_TO_LABEL
from .model import Detection


ResizeMode = Literal["letterbox", "stretch"]


@dataclass(frozen=True)
class DetectorResize:
    """Mapping between a rectified source image and the ONNX input canvas."""

    source_width: int
    source_height: int
    scale_x: float
    scale_y: float
    offset_x: float = 0.0
    offset_y: float = 0.0

    def restore_boxes(self, boxes: np.ndarray) -> np.ndarray:
        """Map ``xyxy`` boxes from ONNX-canvas pixels back to source pixels."""

        restored = np.asarray(boxes, dtype=np.float32).reshape(-1, 4).copy()
        if not len(restored):
            return restored
        restored[:, (0, 2)] = (restored[:, (0, 2)] - self.offset_x) / self.scale_x
        restored[:, (1, 3)] = (restored[:, (1, 3)] - self.offset_y) / self.scale_y
        restored[:, (0, 2)] = np.clip(restored[:, (0, 2)], 0.0, float(self.source_width))
        restored[:, (1, 3)] = np.clip(restored[:, (1, 3)], 0.0, float(self.source_height))
        return restored


def _validate_rgb_image(image_rgb: np.ndarray) -> tuple[int, int]:
    if not isinstance(image_rgb, np.ndarray) or image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
        raise ValueError("expected an H×W×3 RGB numpy array")
    height, width = image_rgb.shape[:2]
    if height < 1 or width < 1:
        raise ValueError("image must not be empty")
    return height, width


def prepare_detector_input(
    image_rgb: np.ndarray,
    *,
    input_width: int,
    input_height: int,
    resize_mode: ResizeMode = "letterbox",
) -> tuple[np.ndarray, DetectorResize]:
    """Build a float32 ``[3,H,W]`` detector tensor and its inverse mapping.

    The tensor deliberately contains RGB values in ``0..1`` only.  The
    TorchVision Faster R-CNN graph retains its own ImageNet normalisation.
    """

    source_height, source_width = _validate_rgb_image(image_rgb)
    if input_width < 1 or input_height < 1:
        raise ValueError("ONNX input width and height must be positive")
    if resize_mode not in ("letterbox", "stretch"):
        raise ValueError("resize_mode must be 'letterbox' or 'stretch'")

    source = Image.fromarray(np.ascontiguousarray(image_rgb).astype(np.uint8, copy=False), mode="RGB")
    if resize_mode == "stretch":
        canvas = np.asarray(source.resize((input_width, input_height), Image.Resampling.BILINEAR))
        mapping = DetectorResize(
            source_width=source_width,
            source_height=source_height,
            scale_x=input_width / source_width,
            scale_y=input_height / source_height,
        )
    else:
        scale = min(input_width / source_width, input_height / source_height)
        resized_width = min(input_width, max(1, int(round(source_width * scale))))
        resized_height = min(input_height, max(1, int(round(source_height * scale))))
        resized = np.asarray(source.resize((resized_width, resized_height), Image.Resampling.BILINEAR))
        canvas = np.zeros((input_height, input_width, 3), dtype=np.uint8)
        left = (input_width - resized_width) // 2
        top = (input_height - resized_height) // 2
        canvas[top : top + resized_height, left : left + resized_width] = resized
        mapping = DetectorResize(
            source_width=source_width,
            source_height=source_height,
            scale_x=resized_width / source_width,
            scale_y=resized_height / source_height,
            offset_x=float(left),
            offset_y=float(top),
        )

    tensor = np.ascontiguousarray(canvas.transpose(2, 0, 1), dtype=np.float32) / 255.0
    return tensor, mapping


def _import_onnxruntime() -> Any:
    try:
        import onnxruntime as ort
    except ModuleNotFoundError as error:
        raise ImportError(
            "ONNX Runtime is required for ONNX inference. Install `onnxruntime` for CPU "
            "or the matching `onnxruntime-gpu` package for CUDA."
        ) from error
    return ort


def onnx_providers(device: str, ort: Any) -> list[Any]:
    """Choose an explicit ONNX Runtime provider chain for the requested device."""

    requested = device.lower()
    available = set(ort.get_available_providers())
    cpu = "CPUExecutionProvider"
    cuda = "CUDAExecutionProvider"
    if requested in ("", "auto"):
        return [cuda, cpu] if cuda in available else [cpu]
    if requested == "cpu":
        return [cpu]
    if requested == "cuda" or requested.startswith("cuda:"):
        if cuda not in available:
            installed = ", ".join(sorted(available)) or "none"
            raise RuntimeError(
                "CUDA was requested for ONNX Runtime but CUDAExecutionProvider is unavailable. "
                f"Available providers: {installed}. Install a CUDA-compatible onnxruntime-gpu package."
            )
        options: dict[str, str] = {}
        if requested.startswith("cuda:"):
            try:
                device_id = int(requested.split(":", 1)[1])
            except ValueError as error:
                raise ValueError("ONNX device must be auto, cpu, cuda, or cuda:<non-negative integer>") from error
            if device_id < 0:
                raise ValueError("ONNX device must be auto, cpu, cuda, or cuda:<non-negative integer>")
            options["device_id"] = str(device_id)
        return [(cuda, options), cpu]
    raise ValueError("ONNX device must be auto, cpu, cuda, or cuda:<non-negative integer>")


def _preload_cuda_dlls(ort: Any, providers: list[Any]) -> None:
    """Load CUDA/cuDNN DLLs before creating a Windows CUDA session when supported.

    Recent ``onnxruntime-gpu`` wheels provide ``preload_dlls``.  It discovers
    the DLLs bundled with PyTorch as well as NVIDIA pip packages, avoiding a
    fragile requirement for users to edit their system PATH.  Older ORT builds
    do not expose the helper and retain their native loader behaviour.
    """

    uses_cuda = any(
        (provider[0] if isinstance(provider, tuple) else provider) == "CUDAExecutionProvider"
        for provider in providers
    )
    preload_dlls = getattr(ort, "preload_dlls", None)
    if uses_cuda and callable(preload_dlls):
        preload_dlls()


def _static_image_shape(session: Any, input_name: str) -> tuple[bool, int, int]:
    """Return ``(has_batch_axis, height, width)`` for an exported image input."""

    input_meta = next((item for item in session.get_inputs() if item.name == input_name), None)
    if input_meta is None:
        names = ", ".join(item.name for item in session.get_inputs())
        raise ValueError(f"ONNX input {input_name!r} was not found; available inputs: {names}")
    shape = list(input_meta.shape)
    if len(shape) == 3:
        channels, height, width = shape
        has_batch_axis = False
    elif len(shape) == 4:
        batch, channels, height, width = shape
        if isinstance(batch, int) and batch != 1:
            raise ValueError(f"ONNX detector supports a batch of one; found input shape {shape}")
        has_batch_axis = True
    else:
        raise ValueError(f"ONNX detector input must be [3,H,W] or [1,3,H,W]; found {shape}")
    if channels != 3 or not isinstance(height, int) or not isinstance(width, int):
        raise ValueError(
            "ONNX detector requires a static RGB image input. "
            f"Expected [3,H,W] or [1,3,H,W] with integer H/W; found {shape}"
        )
    if height < 1 or width < 1:
        raise ValueError(f"ONNX detector input has invalid shape {shape}")
    return has_batch_axis, height, width


class OnnxLRCNNPredictor:
    """ONNX Runtime implementation of the existing ``LRCNNPredictor`` protocol."""

    def __init__(
        self,
        model_path: str | Path,
        *,
        device: str = "auto",
        score_threshold: float = 0.50,
        resize_mode: ResizeMode = "letterbox",
        input_name: str = "image",
        output_names: tuple[str, str, str] = ("boxes", "labels", "scores"),
    ) -> None:
        if not 0.0 <= score_threshold <= 1.0:
            raise ValueError("score_threshold must be between 0 and 1")
        if resize_mode not in ("letterbox", "stretch"):
            raise ValueError("resize_mode must be 'letterbox' or 'stretch'")
        self.model_path = Path(model_path).expanduser().resolve()
        if not self.model_path.is_file():
            raise FileNotFoundError(f"ONNX model not found: {self.model_path}")
        ort = _import_onnxruntime()
        self.providers = onnx_providers(device, ort)
        _preload_cuda_dlls(ort, self.providers)
        self.session = ort.InferenceSession(str(self.model_path), providers=self.providers)
        self.input_name = input_name
        self.has_batch_axis, self.input_height, self.input_width = _static_image_shape(self.session, input_name)
        available_outputs = {item.name for item in self.session.get_outputs()}
        missing = [name for name in output_names if name not in available_outputs]
        if missing:
            available = ", ".join(sorted(available_outputs))
            raise ValueError(
                f"ONNX detector is missing expected output(s): {', '.join(missing)}. "
                f"Available outputs: {available}"
            )
        self.output_names = output_names
        self.score_threshold = score_threshold
        self.resize_mode = resize_mode

    def predict(self, image_rgb: np.ndarray) -> list[Detection]:
        tensor, mapping = prepare_detector_input(
            image_rgb,
            input_width=self.input_width,
            input_height=self.input_height,
            resize_mode=self.resize_mode,
        )
        if self.has_batch_axis:
            tensor = tensor[np.newaxis, ...]
        raw_boxes, raw_labels, raw_scores = self.session.run(list(self.output_names), {self.input_name: tensor})
        boxes = mapping.restore_boxes(np.asarray(raw_boxes))
        labels = np.asarray(raw_labels).reshape(-1)
        scores = np.asarray(raw_scores, dtype=np.float32).reshape(-1)
        count = min(len(boxes), len(labels), len(scores))
        best_by_label: dict[str, Detection] = {}
        for bbox, class_id, score in zip(boxes[:count], labels[:count], scores[:count]):
            score = float(score)
            if score < self.score_threshold:
                continue
            label = ID_TO_LABEL.get(int(class_id))
            if label is None:
                continue
            detection = Detection(label, score, tuple(float(value) for value in bbox))
            if label not in best_by_label or score > best_by_label[label].score:
                best_by_label[label] = detection
        return sorted(best_by_label.values(), key=lambda item: item.label)


_IPHONE_RESOLUTIONS = frozenset({
    (640, 960), (640, 1136), (750, 1334), (828, 1792), (960, 640), (1125, 2436),
    (1136, 640), (1170, 2532), (1179, 2556), (1206, 2622), (1242, 2208), (1242, 2688),
    (1284, 2778), (1290, 2796), (1320, 2868), (1334, 750), (1792, 828), (2208, 1242),
    (2436, 1125), (2532, 1170), (2556, 1179), (2622, 1206), (2688, 1242), (2778, 1284),
    (2796, 1290), (2868, 1320),
})
_ANDROID_PANEL_WIDTHS = frozenset({720, 1080, 1440})
_DEVICE_CN = {"ios": "苹果", "android": "安卓", "uncertain": "不确定", "unknown": "未知"}
_STATUSBAR_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STATUSBAR_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _resolution_platform(width: int, height: int) -> str:
    if width <= 0 or height <= 0:
        return "abstain"
    if (width, height) in _IPHONE_RESOLUTIONS:
        return "ios"
    if min(width, height) in _ANDROID_PANEL_WIDTHS:
        return "android"
    return "abstain"


def _statusbar_tensor(source_rgb: np.ndarray) -> np.ndarray:
    height, _ = _validate_rgb_image(source_rgb)
    strip = source_rgb[: max(1, int(round(height * 0.08)))]
    # Keep Pillow's default resize selection to match the existing PyTorch
    # status-bar preprocessing byte for byte.
    canvas = np.asarray(Image.fromarray(strip).convert("RGB").resize((512, 64)))
    values = canvas.astype(np.float32) / 255.0
    values = (values - _STATUSBAR_MEAN) / _STATUSBAR_STD
    return np.ascontiguousarray(values.transpose(2, 0, 1))[np.newaxis, ...]


class OnnxStatusBarDeviceClassifier:
    """ONNX Runtime equivalent of ``StatusBarDeviceClassifier``.

    It intentionally preserves the resolution prior and conflict rule from the
    PyTorch implementation; only the CNN execution changes to ONNX Runtime.
    """

    def __init__(
        self,
        model_path: str | Path,
        *,
        device: str = "auto",
        conf_uncertain: float = 0.75,
        crosscheck: bool = True,
        input_name: str = "statusbar",
        output_name: str = "probabilities",
    ) -> None:
        if not 0.0 <= conf_uncertain <= 1.0:
            raise ValueError("conf_uncertain must be between 0 and 1")
        self.model_path = Path(model_path).expanduser().resolve()
        if not self.model_path.is_file():
            raise FileNotFoundError(f"ONNX model not found: {self.model_path}")
        ort = _import_onnxruntime()
        self.providers = onnx_providers(device, ort)
        _preload_cuda_dlls(ort, self.providers)
        self.session = ort.InferenceSession(str(self.model_path), providers=self.providers)
        self.input_name = input_name
        input_meta = next((item for item in self.session.get_inputs() if item.name == input_name), None)
        if input_meta is None:
            names = ", ".join(item.name for item in self.session.get_inputs())
            raise ValueError(f"ONNX status-bar input {input_name!r} was not found; available inputs: {names}")
        if list(input_meta.shape) != [1, 3, 64, 512]:
            raise ValueError(
                "ONNX status-bar model must use input [1,3,64,512]; "
                f"found {list(input_meta.shape)}"
            )
        if output_name not in {item.name for item in self.session.get_outputs()}:
            names = ", ".join(item.name for item in self.session.get_outputs())
            raise ValueError(f"ONNX status-bar output {output_name!r} was not found; available outputs: {names}")
        self.output_name = output_name
        self.conf_uncertain = conf_uncertain
        self.crosscheck = crosscheck

    def _p_ios(self, source_rgb: np.ndarray) -> float:
        probabilities = np.asarray(
            self.session.run([self.output_name], {self.input_name: _statusbar_tensor(source_rgb)})[0],
            dtype=np.float32,
        ).reshape(-1)
        if len(probabilities) != 2:
            raise ValueError(f"ONNX status-bar model must return two probabilities; found shape {probabilities.shape}")
        return float(probabilities[1])

    def classify(self, source_rgb: np.ndarray) -> dict[str, object]:
        height, width = _validate_rgb_image(source_rgb)
        resolution_platform = _resolution_platform(width, height)
        if resolution_platform in ("ios", "android"):
            result: dict[str, object] = {
                "platform": resolution_platform,
                "platform_cn": _DEVICE_CN[resolution_platform],
                "source": "resolution",
                "confidence": 0.99,
                "device_prior_conflict": False,
            }
            if self.crosscheck:
                p_ios = self._p_ios(source_rgb)
                cnn_platform = "ios" if p_ios > 0.5 else "android"
                cnn_confidence = max(p_ios, 1.0 - p_ios)
                if cnn_platform != resolution_platform and cnn_confidence >= 0.8:
                    result.update(
                        confidence=0.5,
                        device_prior_conflict=True,
                        cnn_platform=cnn_platform,
                        conflict_detail=f"分辨率判{resolution_platform}、状态栏判{cnn_platform}(疑似缩放伪造)",
                    )
            return result
        p_ios = self._p_ios(source_rgb)
        confidence = max(p_ios, 1.0 - p_ios)
        platform = "uncertain" if confidence < self.conf_uncertain else ("ios" if p_ios > 0.5 else "android")
        return {
            "platform": platform,
            "platform_cn": _DEVICE_CN[platform],
            "source": "cnn",
            "confidence": round(confidence, 3),
            "p_ios": round(p_ios, 4),
            "device_prior_conflict": False,
        }
