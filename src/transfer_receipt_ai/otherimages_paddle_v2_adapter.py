"""Pinned PaddleOCR 2.10.0 DB+CLS+REC adapter for Windows capture.

Importing this module does not import Paddle.  ``create_adapter`` performs the
runtime import and exact-version checks only when the capture CLI calls it.
The engine's raw v2 layout result is retained before any drop-score filtering,
so every detected DB box is represented in the capture evidence.
"""

from __future__ import annotations

import gc
import hashlib
import math
import os
from importlib import metadata
from pathlib import Path

import cv2
import numpy as np

from .ocr import PaddleOCRReader
from .otherimages_paddle_capture import PaddleCaptureBatch, PaddleCapturedLine, PaddleViewContract
from .otherimages_paddle_teacher import (
    ADAPTER_EVIDENCE_KIND,
    PADDLE_EFFECTIVE_ARG_KEYS,
    PINNED_ADAPTER_IMPLEMENTATION,
    canonical_paddle_color_contract,
    _canonical_sha256,
    _is_reparse,
    _require_no_reparse_ancestors,
)


PINNED_PADDLEOCR_VERSION = "2.10.0"
PINNED_ALBUMENTATIONS_VERSION = "1.4.10"
PINNED_ALBUCORE_VERSION = "0.0.13"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _bind_path(path: Path) -> dict[str, object]:
    _require_no_reparse_ancestors(path)
    resolved = path.expanduser().resolve(strict=True)
    if resolved.is_file():
        files = [resolved]
        root = resolved.parent
    elif resolved.is_dir():
        root = resolved
        files = []
        stack = [resolved]
        while stack:
            directory = stack.pop()
            with os.scandir(directory) as entries:
                for entry in entries:
                    item = Path(entry.path)
                    if entry.is_symlink() or _is_reparse(item):
                        raise RuntimeError(f"Paddle model asset traverses a symlink/junction/reparse point: {item}")
                    if entry.is_dir(follow_symlinks=False):
                        stack.append(item)
                    elif entry.is_file(follow_symlinks=False):
                        files.append(item)
                    else:
                        raise RuntimeError(f"Paddle model asset contains a non-regular entry: {item}")
        files.sort(key=lambda item: item.relative_to(root).as_posix().encode("utf-8"))
    else:
        raise FileNotFoundError(resolved)
    bindings = [
        {
            "path": item.relative_to(root).as_posix(),
            "sha256": _sha256(item),
            "size_bytes": item.stat().st_size,
        }
        for item in files
    ]
    return {
        "path": str(resolved),
        "files": bindings,
        "closure_sha256": _canonical_sha256(bindings),
        "size_bytes": sum(int(item["size_bytes"]) for item in bindings),
    }


def _engine_args(engine: object) -> dict[str, object]:
    args = getattr(engine, "args", None)
    if isinstance(args, dict):
        return dict(args)
    try:
        return vars(args) if args is not None else {}
    except TypeError:
        return {}


def _point(value: object, *, line_index: int, point_index: int) -> tuple[float, float]:
    if not isinstance(value, (list, tuple, np.ndarray)) or len(value) != 2:
        raise ValueError(f"Paddle line {line_index} point {point_index} must contain x,y")
    x, y = float(value[0]), float(value[1])
    if not math.isfinite(x) or not math.isfinite(y):
        raise ValueError(f"Paddle line {line_index} point {point_index} is non-finite")
    return x, y


def _capture_with_engine(
    engine: object,
    transformed_rgb: np.ndarray,
    *,
    cropper: object,
    use_angle_cls: bool,
) -> PaddleCaptureBatch:
    """Call the raw PaddleOCR v2 stages without its public drop-score filter."""
    if transformed_rgb.dtype != np.uint8 or transformed_rgb.ndim != 3 or transformed_rgb.shape[2] != 3:
        raise ValueError("Paddle adapter expects HxWx3 uint8 transformed RGB")
    if not callable(cropper):
        raise TypeError("Paddle cropper must be callable")
    text_detector = getattr(engine, "text_detector", None)
    text_classifier = getattr(engine, "text_classifier", None)
    text_recognizer = getattr(engine, "text_recognizer", None)
    if not callable(text_detector) or not callable(text_recognizer):
        raise TypeError("PaddleOCR v2 engine must expose raw text_detector/text_recognizer stages")

    height, width = transformed_rgb.shape[:2]
    # The repository's frozen native/ONNX parity contract feeds Pillow-decoded
    # RGB byte planes straight into PaddleOCR v2, and the .NET CPU adapter does
    # the same.  Make a private writable copy so raw Paddle stages cannot
    # mutate the capture core's immutable pixels, but never exchange R and B.
    source = np.array(transformed_rgb, dtype=np.uint8, order="C", copy=True)
    dt_boxes, _detector_elapsed = text_detector(source)
    if dt_boxes is None or len(dt_boxes) == 0:
        return PaddleCaptureBatch((), 0, 0, 0)
    boxes = [np.asarray(box, dtype=np.float32) for box in dt_boxes]
    for line_index, box in enumerate(boxes):
        if box.shape != (4, 2) or not np.isfinite(box).all():
            raise ValueError(f"Paddle DB line {line_index} must be a finite 4x2 quad")
    boxes.sort(key=lambda box: (float(np.mean(box[:, 1])), float(np.mean(box[:, 0]))))
    crops = [cropper(source, box.copy()) for box in boxes]
    orientations = [0] * len(crops)
    if use_angle_cls:
        if not callable(text_classifier):
            raise TypeError("PaddleOCR v2 angle classifier stage is unavailable")
        crops, angles, _classifier_elapsed = text_classifier(crops)
        if not isinstance(angles, (list, tuple)) or len(angles) != len(crops):
            raise ValueError("Paddle angle classifier must return one result per DB crop")
        args = _engine_args(engine)
        threshold_value = args.get("cls_thresh")
        if isinstance(threshold_value, bool):
            raise ValueError("Paddle angle classifier threshold must be numeric")
        try:
            threshold = float(threshold_value)
        except (TypeError, ValueError):
            raise ValueError("Paddle angle classifier threshold must be numeric") from None
        if not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
            raise ValueError("Paddle angle classifier threshold must be finite and in [0,1]")
        for index, result in enumerate(angles):
            if not isinstance(result, (list, tuple)) or len(result) < 2:
                raise ValueError(f"Paddle angle result {index} must contain label/confidence")
            label = result[0]
            score_value = result[1]
            if label not in {"0", "180"} or isinstance(score_value, bool):
                raise ValueError(f"Paddle angle result {index} must have label 0/180 and numeric confidence")
            try:
                score = float(score_value)
            except (TypeError, ValueError):
                raise ValueError(f"Paddle angle result {index} confidence must be numeric") from None
            if not math.isfinite(score) or not 0.0 <= score <= 1.0:
                raise ValueError(f"Paddle angle result {index} confidence must be finite and in [0,1]")
            # Paddle's TextClassifier rotates only this exact decision.  Record
            # the applied transform rather than the unthresholded class label.
            orientations[index] = 180 if label == "180" and score > threshold else 0
    recognitions, _recognizer_elapsed = text_recognizer(crops)
    if not isinstance(recognitions, (list, tuple)):
        raise TypeError("PaddleOCR v2 recognizer result must be a list")
    if len(recognitions) > len(boxes):
        raise ValueError("PaddleOCR v2 recognizer returned more results than raw DB boxes")

    rejected_count = max(0, len(boxes) - len(recognitions))
    captured: list[PaddleCapturedLine] = []
    for line_index, quad_raw in enumerate(boxes):
        if line_index >= len(recognitions):
            text = ""
            confidence = 0.0
        else:
            recognition = recognitions[line_index]
            if not isinstance(recognition, (list, tuple)) or len(recognition) < 2 or not isinstance(recognition[0], str):
                raise ValueError(f"Paddle line {line_index} recognition must contain text/confidence")
            text = recognition[0]
            confidence = float(recognition[1])
            if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
                raise ValueError(f"Paddle line {line_index} confidence is invalid")
        normalized_quad: list[tuple[float, float]] = []
        transformed_quad_pixels: list[tuple[float, float]] = []
        for point_index, raw_point in enumerate(quad_raw):
            x, y = _point(raw_point, line_index=line_index, point_index=point_index)
            x_normalized = x / max(1, width - 1)
            y_normalized = y / max(1, height - 1)
            if not 0.0 <= x_normalized <= 1.0 or not 0.0 <= y_normalized <= 1.0:
                raise ValueError(f"Paddle line {line_index} quad is outside transformed image")
            normalized_quad.append((x_normalized, y_normalized))
            transformed_quad_pixels.append((x, y))
        captured.append(
            PaddleCapturedLine(
                text=text,
                confidence=confidence,
                orientation_degrees=orientations[line_index],
                transformed_quad_pixels=tuple(transformed_quad_pixels),  # type: ignore[arg-type]
                quad_normalized=tuple(normalized_quad),  # type: ignore[arg-type]
            )
        )
    captured.sort(
        key=lambda line: (
            sum(point[1] for point in line.quad_normalized) / 4.0,
            sum(point[0] for point in line.quad_normalized) / 4.0,
            line.text,
        )
    )
    return PaddleCaptureBatch(
        lines=tuple(captured),
        raw_detected_line_count=len(boxes),
        recognition_attempted_line_count=len(crops),
        recognition_rejected_line_count=rejected_count,
    )


class PinnedPaddleOcrV2Adapter:
    def __init__(self, *, device: str = "auto") -> None:
        for package, expected in (
            ("paddleocr", PINNED_PADDLEOCR_VERSION),
            ("albumentations", PINNED_ALBUMENTATIONS_VERSION),
            ("albucore", PINNED_ALBUCORE_VERSION),
        ):
            observed = metadata.version(package)
            if observed != expected:
                raise RuntimeError(f"{package} must be pinned to {expected}; observed {observed}")
        requested = device.lower()
        if requested == "gpu":
            requested = "cuda"
        if requested not in {"auto", "cpu", "cuda"} and not (
            requested.startswith("cuda:") and requested[5:].isdigit()
        ):
            raise ValueError("OTHERIMAGES_PADDLE_DEVICE must be auto, cpu, cuda, or cuda:N")
        # First initialise only to materialise/download the exact v2 assets and
        # discover Paddle's effective paths.  This engine is deliberately not
        # used for OCR: the disk assets are bound before a fresh execution
        # engine is constructed, closing the old "engine A / evidence B" gap.
        bootstrap_reader = PaddleOCRReader(
            language="ch",
            use_angle_cls=True,
            device=requested,
            require_v2=True,
        )
        bootstrap_args = _engine_args(bootstrap_reader._engine)
        required = {
            "det": bootstrap_args.get("det_model_dir"),
            "cls": bootstrap_args.get("cls_model_dir"),
            "rec": bootstrap_args.get("rec_model_dir"),
            "dictionary": bootstrap_args.get("rec_char_dict_path"),
        }
        if any(not isinstance(value, str) or not value for value in required.values()):
            raise RuntimeError("Pinned PaddleOCR engine did not expose det/cls/rec/dictionary paths")
        self._assets = {name: _bind_path(Path(str(value))) for name, value in required.items()}
        del bootstrap_reader
        gc.collect()

        execution_reader = PaddleOCRReader(
            language="ch",
            use_angle_cls=True,
            device=requested,
            require_v2=True,
        )
        self._engine = execution_reader._engine
        args = _engine_args(self._engine)
        execution_required = {
            "det": args.get("det_model_dir"),
            "cls": args.get("cls_model_dir"),
            "rec": args.get("rec_model_dir"),
            "dictionary": args.get("rec_char_dict_path"),
        }
        if any(not isinstance(value, str) or not value for value in execution_required.values()):
            raise RuntimeError("Fresh PaddleOCR execution engine omitted det/cls/rec/dictionary paths")
        expected_paths = {
            role: str(Path(str(path)).expanduser().resolve(strict=True))
            for role, path in required.items()
        }
        observed_paths = {
            role: str(Path(str(path)).expanduser().resolve(strict=True))
            for role, path in execution_required.items()
        }
        if observed_paths != expected_paths:
            raise RuntimeError("Fresh PaddleOCR execution engine changed model/dictionary paths")
        rebound_assets = {name: _bind_path(Path(str(value))) for name, value in execution_required.items()}
        if rebound_assets != self._assets:
            raise RuntimeError("Paddle model/dictionary bytes changed while the fresh execution engine loaded")

        import paddle

        self._use_angle_cls = True
        self._device = str(paddle.get_device())
        self._drop_score = float(args.get("drop_score", 0.5))
        if not math.isfinite(self._drop_score) or not 0.0 <= self._drop_score <= 1.0:
            raise RuntimeError("Pinned PaddleOCR engine exposed invalid drop_score")
        self._runtime_versions = {
            "paddleocr": PINNED_PADDLEOCR_VERSION,
            "paddlepaddle": str(getattr(paddle, "__version__", "unknown")),
            "albumentations": PINNED_ALBUMENTATIONS_VERSION,
            "albucore": PINNED_ALBUCORE_VERSION,
            "opencv": str(cv2.__version__),
            "numpy": str(np.__version__),
            "pillow": metadata.version("Pillow"),
        }
        missing_effective_args = [name for name in PADDLE_EFFECTIVE_ARG_KEYS if name not in args]
        if missing_effective_args:
            raise RuntimeError(f"Pinned PaddleOCR engine omitted effective arguments: {missing_effective_args}")
        self._effective_args = {name: args[name] for name in PADDLE_EFFECTIVE_ARG_KEYS}
        model_identity_payload = {
            "adapter_implementation": PINNED_ADAPTER_IMPLEMENTATION,
            "paddleocr_version": PINNED_PADDLEOCR_VERSION,
            "runtime_versions": self._runtime_versions,
            "effective_paddle_args": self._effective_args,
            "device": self._device,
            "drop_score": self._drop_score,
            "assets": self._assets,
            "paddle_color_contract": canonical_paddle_color_contract(),
        }
        self._model_contract_sha256 = _canonical_sha256(model_identity_payload)

    def evidence(self) -> dict[str, object]:
        return {
            "kind": ADAPTER_EVIDENCE_KIND,
            "adapter_implementation": PINNED_ADAPTER_IMPLEMENTATION,
            "paddle_version": PINNED_PADDLEOCR_VERSION,
            "model_contract_sha256": self._model_contract_sha256,
            "drop_score": self._drop_score,
            "stages": {"db": True, "cls": True, "rec": True},
            "execution_device": self._device,
            "runtime_versions": self._runtime_versions,
            "effective_paddle_args": self._effective_args,
            "model_assets": self._assets,
            "raw_db_lines_preserved_before_drop_filter": True,
            "paddle_color_contract": canonical_paddle_color_contract(),
        }

    def capture(self, transformed_rgb: np.ndarray, view: PaddleViewContract) -> PaddleCaptureBatch:
        del view
        from paddleocr.tools.infer.utility import get_rotate_crop_image
        return _capture_with_engine(
            self._engine,
            transformed_rgb,
            cropper=get_rotate_crop_image,
            use_angle_cls=self._use_angle_cls,
        )


def create_adapter() -> PinnedPaddleOcrV2Adapter:
    """Default built-in Windows factory used by the capture checkout script."""
    return PinnedPaddleOcrV2Adapter(device=os.environ.get("OTHERIMAGES_PADDLE_DEVICE", "auto"))
