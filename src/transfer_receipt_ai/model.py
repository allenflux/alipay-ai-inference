"""Lightweight region-CNN detector used for transfer receipt field localisation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .labels import DETECTION_CLASSES, ID_TO_LABEL, NUM_MODEL_CLASSES


@dataclass(frozen=True)
class LRCNNConfig:
    """Configuration for the project's LRCNN (MobileNetV3-FPN Faster R-CNN)."""

    min_size: int = 768
    max_size: int = 1536
    trainable_backbone_layers: int = 3
    pretrained: bool = False

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class Detection:
    label: str
    score: float
    bbox_xyxy: tuple[float, float, float, float]

    def as_dict(self) -> dict[str, object]:
        return {
            "label": self.label,
            "score": round(self.score, 6),
            "bbox_rectified": [round(value, 3) for value in self.bbox_xyxy],
        }


def build_lrcnn(
    config: LRCNNConfig | None = None,
    *,
    load_pretrained_weights: bool | None = None,
):
    """Build the requested lightweight R-CNN detector.

    The architecture is MobileNetV3 + FPN + RPN + RoIAlign + Fast R-CNN heads.
    It is deliberately called ``LRCNN`` in this project (lightweight region CNN),
    while using TorchVision's tested Faster R-CNN implementation underneath.
    """
    config = config or LRCNNConfig()
    if load_pretrained_weights is None:
        load_pretrained_weights = config.pretrained
    try:
        from torch import nn
        from torchvision.models import mobilenet_v3_large
        from torchvision.models.detection import FasterRCNN
        from torchvision.models.detection import FasterRCNN_MobileNet_V3_Large_FPN_Weights
        from torchvision.models.detection.backbone_utils import _mobilenet_extractor
        from torchvision.models.detection.rpn import AnchorGenerator
        from torchvision.ops.misc import FrozenBatchNorm2d
    except ModuleNotFoundError as error:
        raise ImportError("PyTorch and TorchVision are required. Install `pip install -r requirements.txt`.") from error

    # Build the backbone explicitly instead of forwarding a custom anchor
    # generator through fasterrcnn_mobilenet_v3_large_fpn().  Recent TorchVision
    # versions create their own generator inside that factory, which otherwise
    # passes rpn_anchor_generator to FasterRCNN twice.
    #
    # Keep the normalization architecture tied to the checkpoint configuration,
    # not to whether weights are downloaded during this particular call.  This
    # lets resume and inference rebuild a pretrained checkpoint without changing
    # FrozenBatchNorm2d into BatchNorm2d.
    if config.pretrained:
        norm_layer = FrozenBatchNorm2d
        trainable_backbone_layers = config.trainable_backbone_layers
    else:
        norm_layer = nn.BatchNorm2d
        # TorchVision trains every backbone layer when starting without weights.
        trainable_backbone_layers = 6
    mobilenet = mobilenet_v3_large(weights=None, norm_layer=norm_layer)
    backbone = _mobilenet_extractor(mobilenet, True, trainable_backbone_layers)

    # The compact status region and long field rows need smaller and wider anchors
    # than generic COCO. MobileNetV3-FPN exposes three feature levels, each with
    # five scales and five aspect ratios.
    anchor_generator = AnchorGenerator(
        sizes=((8, 16, 32, 64, 128),) * 3,
        aspect_ratios=((0.25, 0.5, 1.0, 2.0, 4.0),) * 3,
    )
    model = FasterRCNN(
        backbone,
        num_classes=NUM_MODEL_CLASSES,
        min_size=config.min_size,
        max_size=config.max_size,
        rpn_anchor_generator=anchor_generator,
        rpn_score_thresh=0.05,
        box_detections_per_img=50,
    )
    if config.pretrained and load_pretrained_weights:
        # Reuse every matching COCO tensor (backbone/FPN/RoI features), while the
        # six-logit (background + five fields) final head and custom-anchor logits
        # remain freshly initialised.
        pretrained_state = FasterRCNN_MobileNet_V3_Large_FPN_Weights.DEFAULT.get_state_dict(
            progress=True,
            check_hash=True,
        )
        current_state = model.state_dict()
        compatible_state = {
            key: value
            for key, value in pretrained_state.items()
            if key in current_state and current_state[key].shape == value.shape
        }
        model.load_state_dict(compatible_state, strict=False)
    return model


def choose_device(requested: str = "auto") -> str:
    """Prefer service CUDA, then Apple MPS, then CPU."""
    import torch

    requested = requested.lower()
    if requested != "auto":
        if requested.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        if requested == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("MPS was requested but is unavailable")
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def validate_checkpoint_classes(payload: Any) -> None:
    """Reject checkpoints trained with a different semantic class order.

    This is stricter than tensor-shape validation: the old and new receipt
    schemas both have five foreground classes, so PyTorch would otherwise load
    old weights successfully while silently assigning each class a new meaning.
    """
    expected = list(DETECTION_CLASSES)
    found = payload.get("classes") if isinstance(payload, dict) else None
    if not isinstance(found, (list, tuple)) or list(found) != expected:
        raise ValueError(
            "Checkpoint labels do not match the current five-field schema. "
            f"Expected {expected}; found {found!r}. Re-train from newly labelled data."
        )


class LRCNNPredictor:
    """Checkpoint-backed detector with one best detection per required field."""

    def __init__(
        self,
        checkpoint_path: str | Path,
        *,
        device: str = "auto",
        score_threshold: float = 0.50,
        model_config: LRCNNConfig | None = None,
    ) -> None:
        import torch

        self.device = choose_device(device)
        payload = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        validate_checkpoint_classes(payload)
        checkpoint_config = payload.get("model_config") if isinstance(payload, dict) else None
        if model_config is not None:
            config = model_config
        elif isinstance(checkpoint_config, dict):
            config = LRCNNConfig(**checkpoint_config)
        else:
            config = LRCNNConfig()
        # Model weights are in the checkpoint, so rebuild the same architecture
        # without triggering another COCO download.
        self.model = build_lrcnn(config, load_pretrained_weights=False)
        state_dict = payload.get("model_state") if isinstance(payload, dict) else payload
        if not isinstance(state_dict, dict):
            raise ValueError("Checkpoint does not contain a model_state dictionary")
        self.model.load_state_dict(state_dict)
        self.model.to(self.device).eval()
        self.score_threshold = score_threshold

    def predict(self, image_rgb: np.ndarray) -> list[Detection]:
        import torch

        if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
            raise ValueError("predict expects an H×W×3 RGB image")
        tensor = torch.from_numpy(np.ascontiguousarray(image_rgb)).permute(2, 0, 1).float().div(255.0).to(self.device)
        with torch.inference_mode():
            result = self.model([tensor])[0]
        boxes = result["boxes"].detach().cpu().numpy()
        labels = result["labels"].detach().cpu().numpy()
        scores = result["scores"].detach().cpu().numpy()
        best_by_label: dict[str, Detection] = {}
        for bbox, class_id, score in zip(boxes, labels, scores):
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
