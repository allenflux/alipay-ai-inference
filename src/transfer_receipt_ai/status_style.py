"""Standalone classifier for the visual style of the success-status check.

This module deliberately stays separate from the five-field detector.  The
detector first locates the transfer-status region; this classifier can then be
run on that crop to distinguish the three presentation styles used by the
business rule.

The result is only a UI-style signal.  In particular, seeing a check mark does
not prove that a receipt is authentic; all authentication decisions need other
independent evidence as well.
"""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Final, Mapping, Sequence

import numpy as np


STATUS_STYLE_CLASSES: Final[tuple[str, ...]] = (
    "check_offset",
    "check_aligned",
    "check_absent",
)

UNKNOWN_STATUS_STYLE: Final[str] = "unknown"
STATUS_STYLE_SCHEMA_VERSION: Final[int] = 2
STATUS_STYLE_RULE_VERSION: Final[str] = "status-style-v2"

STATUS_STYLE_BUSINESS_TAGS: Final[dict[str, str]] = {
    "check_offset": "android",
    "check_aligned": "ios",
    "check_absent": "suspected_fake",
    UNKNOWN_STATUS_STYLE: "review",
}


@dataclass(frozen=True)
class StatusStyleConfig:
    """Architecture and preprocessing settings saved with each checkpoint.

    ``input_width`` and ``input_height`` describe the final canvas.  The crop
    is scaled uniformly and padded to that canvas; it is never stretched or
    centre-cropped, because either operation could change the alignment signal
    that this classifier is meant to learn.
    """

    input_width: int = 320
    input_height: int = 128
    pretrained: bool = False

    def __post_init__(self) -> None:
        if self.input_width < 1 or self.input_height < 1:
            raise ValueError("status-style input width and height must be positive")

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class StatusStylePrediction:
    """Thresholded classifier result and its auditable raw probabilities."""

    label: str
    confidence: float
    business_tag: str
    candidate_label: str
    probabilities: dict[str, float]

    def as_dict(self) -> dict[str, object]:
        return {
            "label": self.label,
            "confidence": round(self.confidence, 6),
            "business_tag": self.business_tag,
            "candidate_label": self.candidate_label,
            "probabilities": {
                label: round(float(self.probabilities[label]), 6)
                for label in STATUS_STYLE_CLASSES
            },
        }


def business_tag_for_status_style(label: str) -> str:
    """Map a thresholded visual-style label to the requested business tag.

    Unknown or unrecognised outputs are routed to manual review.  The
    ``suspected_fake`` tag is a review signal rather than a final authenticity
    verdict.
    """

    return STATUS_STYLE_BUSINESS_TAGS.get(label, "review")


def status_style_tags(prediction: Mapping[str, object]) -> dict[str, object]:
    """Build the auditable platform/risk tags shared by sidecar and inline inference."""

    label = prediction.get("label")
    if label == "check_offset":
        return {
            "platform": "android",
            "authenticity": "not_assessed",
            "review_tag": None,
            "requires_manual_review": False,
            "reason": "status_check_offset",
            "rule_version": STATUS_STYLE_RULE_VERSION,
        }
    if label == "check_aligned":
        return {
            "platform": "ios",
            "authenticity": "not_assessed",
            "review_tag": None,
            "requires_manual_review": False,
            "reason": "status_check_aligned",
            "rule_version": STATUS_STYLE_RULE_VERSION,
        }
    if label == "check_absent":
        return {
            "platform": None,
            "authenticity": "not_assessed",
            "review_tag": "suspected_fake",
            "requires_manual_review": True,
            "reason": "status_check_absent",
            "rule_version": STATUS_STYLE_RULE_VERSION,
        }
    return {
        "platform": None,
        "authenticity": "not_assessed",
        "review_tag": "review",
        "requires_manual_review": True,
        "reason": "status_style_low_confidence",
        "rule_version": STATUS_STYLE_RULE_VERSION,
    }


def status_style_checkpoint_signature(checkpoint_path: str | Path) -> dict[str, object]:
    """Return a stable identity for skip/resume and inline-result provenance."""

    checkpoint = Path(checkpoint_path).resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    digest = hashlib.sha256()
    with checkpoint.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return {
        "path": checkpoint.as_posix(),
        "sha256": digest.hexdigest(),
        "size_bytes": checkpoint.stat().st_size,
    }


def validate_status_style_checkpoint(payload: Any) -> None:
    """Reject checkpoints whose class order differs from this exact contract."""

    expected = list(STATUS_STYLE_CLASSES)
    found = payload.get("classes") if isinstance(payload, dict) else None
    if not isinstance(found, (list, tuple)) or list(found) != expected:
        raise ValueError(
            "Checkpoint labels do not match the status-style schema. "
            f"Expected {expected}; found {found!r}. Re-train with the exact class order."
        )


def build_status_style_model(
    config: StatusStyleConfig | None = None,
    *,
    load_pretrained_weights: bool | None = None,
):
    """Build a three-logit MobileNetV3-small image classifier."""

    config = config or StatusStyleConfig()
    if load_pretrained_weights is None:
        load_pretrained_weights = config.pretrained
    try:
        from torch import nn
        from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small
    except (ImportError, ModuleNotFoundError) as error:
        raise ImportError(
            "PyTorch and TorchVision are required for status-style classification. "
            "Install the project's training dependencies."
        ) from error

    weights = MobileNet_V3_Small_Weights.DEFAULT if load_pretrained_weights else None
    model = mobilenet_v3_small(weights=weights)
    final_layer = model.classifier[-1]
    if not isinstance(final_layer, nn.Linear):  # pragma: no cover - guards torchvision API drift
        raise RuntimeError("Unexpected MobileNetV3-small classifier layout")
    model.classifier[-1] = nn.Linear(final_layer.in_features, len(STATUS_STYLE_CLASSES))
    return model


def letterbox_status_crop(
    image_rgb: np.ndarray,
    target_size: tuple[int, int] = (320, 128),
    *,
    fill: tuple[int, int, int] = (0, 0, 0),
) -> np.ndarray:
    """Resize an RGB crop onto a ``(width, height)`` canvas without distortion.

    The whole source crop remains visible.  Scaling is uniform, and any unused
    canvas area is split as evenly as possible between opposite sides.
    """

    if not isinstance(image_rgb, np.ndarray) or image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
        raise ValueError("letterbox_status_crop expects an H×W×3 RGB numpy array")
    source_height, source_width = image_rgb.shape[:2]
    if source_width < 1 or source_height < 1:
        raise ValueError("status crop must not be empty")
    target_width, target_height = (int(target_size[0]), int(target_size[1]))
    if target_width < 1 or target_height < 1:
        raise ValueError("letterbox target width and height must be positive")
    if len(fill) != 3 or any(not 0 <= int(value) <= 255 for value in fill):
        raise ValueError("letterbox fill must contain three values in the range 0..255")

    # Rounding independently after calculating one shared scale changes either
    # dimension by at most one pixel; it never uses a second, distorting scale.
    scale = min(target_width / source_width, target_height / source_height)
    resized_width = min(target_width, max(1, int(round(source_width * scale))))
    resized_height = min(target_height, max(1, int(round(source_height * scale))))

    try:
        from PIL import Image
    except (ImportError, ModuleNotFoundError) as error:  # pragma: no cover - declared dependency
        raise ImportError("Pillow is required for status-style preprocessing") from error

    source = Image.fromarray(np.ascontiguousarray(image_rgb).astype(np.uint8, copy=False), mode="RGB")
    resized = np.asarray(source.resize((resized_width, resized_height), Image.Resampling.BILINEAR))
    canvas = np.empty((target_height, target_width, 3), dtype=np.uint8)
    canvas[...] = np.asarray(fill, dtype=np.uint8)
    left = (target_width - resized_width) // 2
    top = (target_height - resized_height) // 2
    canvas[top : top + resized_height, left : left + resized_width] = resized
    return canvas


def preprocess_status_crop(
    image_rgb: np.ndarray,
    config: StatusStyleConfig | None = None,
):
    """Letterbox and ImageNet-normalise one crop into a C×H×W float tensor."""

    config = config or StatusStyleConfig()
    try:
        import torch
    except (ImportError, ModuleNotFoundError) as error:
        raise ImportError("PyTorch is required for status-style preprocessing") from error

    canvas = letterbox_status_crop(
        image_rgb,
        (config.input_width, config.input_height),
    )
    tensor = torch.from_numpy(np.ascontiguousarray(canvas)).permute(2, 0, 1).float().div(255.0)
    mean = tensor.new_tensor((0.485, 0.456, 0.406)).view(3, 1, 1)
    std = tensor.new_tensor((0.229, 0.224, 0.225)).view(3, 1, 1)
    return (tensor - mean) / std


def _validate_thresholds(confidence_threshold: float, absent_confidence_threshold: float) -> None:
    if not 0.0 <= confidence_threshold <= 1.0:
        raise ValueError("confidence_threshold must be between 0 and 1")
    if not 0.0 <= absent_confidence_threshold <= 1.0:
        raise ValueError("absent_confidence_threshold must be between 0 and 1")
    if absent_confidence_threshold < confidence_threshold:
        raise ValueError("absent_confidence_threshold must be at least confidence_threshold")


def prediction_from_probabilities(
    probabilities: Mapping[str, float] | Sequence[float],
    *,
    confidence_threshold: float = 0.70,
    absent_confidence_threshold: float = 0.85,
) -> StatusStylePrediction:
    """Apply class-specific confidence gates to one probability vector.

    ``check_absent`` deliberately has its own, higher default gate because it
    maps to the more consequential ``suspected_fake`` review tag.
    """

    _validate_thresholds(confidence_threshold, absent_confidence_threshold)
    if isinstance(probabilities, Mapping):
        if set(probabilities) != set(STATUS_STYLE_CLASSES):
            raise ValueError(f"probabilities must contain exactly {list(STATUS_STYLE_CLASSES)}")
        values = [float(probabilities[label]) for label in STATUS_STYLE_CLASSES]
    else:
        values = [float(value) for value in probabilities]
        if len(values) != len(STATUS_STYLE_CLASSES):
            raise ValueError(f"expected {len(STATUS_STYLE_CLASSES)} probabilities; found {len(values)}")
    if any(not np.isfinite(value) or value < 0.0 or value > 1.0 for value in values):
        raise ValueError("probabilities must be finite values between 0 and 1")

    candidate_index = int(np.argmax(values))
    candidate_label = STATUS_STYLE_CLASSES[candidate_index]
    confidence = values[candidate_index]
    required_confidence = (
        absent_confidence_threshold
        if candidate_label == "check_absent"
        else confidence_threshold
    )
    label = candidate_label if confidence >= required_confidence else UNKNOWN_STATUS_STYLE
    probability_map = dict(zip(STATUS_STYLE_CLASSES, values))
    return StatusStylePrediction(
        label=label,
        confidence=confidence,
        business_tag=business_tag_for_status_style(label),
        candidate_label=candidate_label,
        probabilities=probability_map,
    )


class StatusStylePredictor:
    """Checkpoint-backed MobileNetV3-small predictor for one status crop."""

    def __init__(
        self,
        checkpoint_path: str | Path,
        *,
        device: str = "auto",
        confidence_threshold: float = 0.70,
        absent_confidence_threshold: float = 0.85,
        model_config: StatusStyleConfig | None = None,
    ) -> None:
        import torch

        # Imported lazily so the data/annotation tools do not require PyTorch.
        from .model import choose_device

        _validate_thresholds(confidence_threshold, absent_confidence_threshold)
        self.device = choose_device(device)
        payload = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        validate_status_style_checkpoint(payload)
        checkpoint_config = payload.get("model_config") if isinstance(payload, dict) else None
        if model_config is not None:
            config = model_config
        elif isinstance(checkpoint_config, dict):
            config = StatusStyleConfig(**checkpoint_config)
        else:
            config = StatusStyleConfig()
        state_dict = payload.get("model_state") if isinstance(payload, dict) else None
        if not isinstance(state_dict, dict):
            raise ValueError("Status-style checkpoint does not contain a model_state dictionary")

        self.model = build_status_style_model(config, load_pretrained_weights=False)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device).eval()
        self.config = config
        self.confidence_threshold = confidence_threshold
        self.absent_confidence_threshold = absent_confidence_threshold

    def predict(self, image_rgb: np.ndarray) -> StatusStylePrediction:
        import torch

        tensor = preprocess_status_crop(image_rgb, self.config).unsqueeze(0).to(self.device)
        with torch.inference_mode():
            logits = self.model(tensor)
            probabilities = torch.softmax(logits, dim=1)[0].detach().cpu().tolist()
        return prediction_from_probabilities(
            probabilities,
            confidence_threshold=self.confidence_threshold,
            absent_confidence_threshold=self.absent_confidence_threshold,
        )
