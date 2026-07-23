"""Export the deployable PyTorch receipt models as standard ONNX artifacts.

The exporter deliberately creates fixed-shape graphs.  TorchVision documents
Faster R-CNN ONNX support for a fixed batch and fixed image dimensions; the
companion ONNX runtime adapter owns the matching fixed-canvas preprocessing.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Literal

import numpy as np

from .labels import DETECTION_CLASSES
from .model import LRCNNPredictor
from .onnx_runtime import prepare_detector_input


ExportKind = Literal["detector", "statusbar"]
DETECTOR_INPUT_NAME = "image"
DETECTOR_OUTPUT_NAMES = ("boxes", "labels", "scores")
STATUSBAR_INPUT_NAME = "statusbar"
STATUSBAR_OUTPUT_NAME = "probabilities"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: dict[str, object]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def contract_path_for(model_path: str | Path) -> Path:
    """Return the portable sidecar contract path for an ONNX model."""

    return Path(model_path).with_suffix(".contract.json")


def _import_onnx() -> Any:
    try:
        import onnx
    except ModuleNotFoundError as error:
        raise ImportError("ONNX export requires the `onnx` package. Install `pip install -r requirements-export.txt`.") from error
    return onnx


def _export_runtime_versions() -> dict[str, str | None]:
    """Capture the conversion/runtime versions that materially affect an artifact."""

    onnx = _import_onnx()
    try:
        import onnxruntime
    except ModuleNotFoundError:
        onnxruntime_version: str | None = None
    else:
        onnxruntime_version = onnxruntime.__version__
    return {"onnx_version": onnx.__version__, "onnxruntime_version": onnxruntime_version}


def _load_sample_tensor(
    sample_image: Path,
    *,
    input_width: int,
    input_height: int,
    resize_mode: str,
) -> np.ndarray:
    from .geometry import load_upright_rgb

    if not sample_image.is_file():
        raise FileNotFoundError(f"Sample image not found: {sample_image}")
    source_rgb = load_upright_rgb(sample_image)
    tensor, _ = prepare_detector_input(
        source_rgb,
        input_width=input_width,
        input_height=input_height,
        resize_mode=resize_mode,  # type: ignore[arg-type]
    )
    return tensor


def _temporary_onnx_path(output_path: Path) -> Path:
    descriptor, raw_path = tempfile.mkstemp(
        prefix=f".{output_path.stem}.", suffix=".onnx.tmp", dir=str(output_path.parent)
    )
    os.close(descriptor)
    return Path(raw_path)


def _legacy_onnx_export(torch: Any, module: Any, sample: Any, output_path: Path, *, input_names: list[str], output_names: list[str], opset_version: int) -> None:
    """Call the legacy exporter on both new and older supported PyTorch releases."""

    kwargs: dict[str, object] = {
        "export_params": True,
        "opset_version": opset_version,
        "do_constant_folding": True,
        "input_names": input_names,
        "output_names": output_names,
    }
    # ``dynamo`` was added after older PyTorch versions already used in some
    # training environments. Pass it only when available, while still pinning
    # current releases to TorchVision's legacy export path.
    if "dynamo" in inspect.signature(torch.onnx.export).parameters:
        kwargs["dynamo"] = False
    torch.onnx.export(module, sample, str(output_path), **kwargs)


def _verify_runtime_graph(
    model_path: Path,
    *,
    input_name: str,
    sample: np.ndarray,
    output_names: tuple[str, ...],
    expected_outputs: tuple[np.ndarray, ...],
) -> None:
    """Compare the fixed PyTorch wrapper and fresh ONNX Runtime graph on one input."""

    try:
        import onnxruntime as ort
    except ModuleNotFoundError as error:
        raise ImportError(
            "Runtime verification requires `onnxruntime`. Install `pip install -r requirements-export.txt` "
            "or pass --skip-runtime-verify."
        ) from error
    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    outputs = session.run(list(output_names), {input_name: np.ascontiguousarray(sample)})
    if len(outputs) != len(output_names):  # pragma: no cover - defensive provider guard
        raise RuntimeError(f"ONNX Runtime returned {len(outputs)} outputs; expected {len(output_names)}")
    if any(not np.all(np.isfinite(np.asarray(output, dtype=np.float64))) for output in outputs):
        raise RuntimeError("ONNX Runtime produced non-finite output values")
    for name, expected, actual in zip(output_names, expected_outputs, outputs):
        expected_array = np.asarray(expected)
        actual_array = np.asarray(actual)
        if actual_array.shape != expected_array.shape:
            raise RuntimeError(
                f"ONNX Runtime output {name!r} shape {list(actual_array.shape)} does not match "
                f"the PyTorch wrapper shape {list(expected_array.shape)}"
            )
        if expected_array.dtype.kind in "iu":
            if not np.array_equal(actual_array, expected_array):
                raise RuntimeError(f"ONNX Runtime output {name!r} does not match the PyTorch wrapper")
        elif not np.allclose(actual_array, expected_array, rtol=1e-3, atol=1e-3):
            maximum_error = float(np.max(np.abs(actual_array - expected_array), initial=0.0))
            raise RuntimeError(
                f"ONNX Runtime output {name!r} diverges from the PyTorch wrapper; "
                f"maximum absolute error={maximum_error:.6g}"
            )


def _export_detector(
    checkpoint: Path,
    output_path: Path,
    *,
    sample_tensor: np.ndarray,
    opset_version: int,
    verify_runtime: bool,
) -> dict[str, object]:
    import torch
    import torchvision

    predictor = LRCNNPredictor(checkpoint, device="cpu", score_threshold=0.0)
    detector = predictor.model.eval()

    class ReceiptDetectorOnnxWrapper(torch.nn.Module):
        def __init__(self, model: torch.nn.Module) -> None:
            super().__init__()
            self.model = model

        def forward(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            prediction = self.model([image])[0]
            return prediction["boxes"], prediction["labels"], prediction["scores"]

    wrapper = ReceiptDetectorOnnxWrapper(detector).eval()
    sample = torch.from_numpy(np.ascontiguousarray(sample_tensor)).to(dtype=torch.float32)
    with torch.inference_mode():
        expected_outputs = tuple(item.detach().cpu().numpy() for item in wrapper(sample))
    temporary_path = _temporary_onnx_path(output_path)
    try:
        # TorchVision's detection-specific ONNX branches are exercised by the
        # legacy exporter. The newer Dynamo exporter is intentionally not used
        # until it is proven against this model's RPN/RoIAlign/NMS graph.
        _legacy_onnx_export(
            torch,
            wrapper,
            sample,
            temporary_path,
            input_names=[DETECTOR_INPUT_NAME],
            output_names=list(DETECTOR_OUTPUT_NAMES),
            opset_version=opset_version,
        )
        onnx = _import_onnx()
        onnx.checker.check_model(str(temporary_path))
        if verify_runtime:
            _verify_runtime_graph(
                temporary_path,
                input_name=DETECTOR_INPUT_NAME,
                sample=sample_tensor,
                output_names=DETECTOR_OUTPUT_NAMES,
                expected_outputs=expected_outputs,
            )
        temporary_path.replace(output_path)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise

    return {
        "kind": "receipt_lrcnn_v1",
        "input": {
            "name": DETECTOR_INPUT_NAME,
            "shape": [int(value) for value in sample.shape],
            "dtype": "float32",
            "layout": "CHW",
            "color_space": "RGB",
            "value_range": [0.0, 1.0],
            "normalization": "TorchVision FasterRCNN transform inside ONNX graph",
        },
        "outputs": [
            {"name": "boxes", "dtype": "float32", "shape": ["N", 4], "coordinate_format": "xyxy"},
            {"name": "labels", "dtype": "int64", "shape": ["N"]},
            {"name": "scores", "dtype": "float32", "shape": ["N"]},
        ],
        "classes": {str(index): label for index, label in enumerate(DETECTION_CLASSES, start=1)},
        "postprocess": {
            "score_threshold": 0.50,
            "select_highest_score_per_class": True,
            "max_detections": 50,
        },
        "torch": {"version": torch.__version__, "torchvision_version": torchvision.__version__},
    }


def _export_statusbar(
    checkpoint: Path,
    output_path: Path,
    *,
    opset_version: int,
    verify_runtime: bool,
) -> dict[str, object]:
    import torch

    from .device_statusbar import StatusBarDeviceClassifier

    predictor = StatusBarDeviceClassifier(checkpoint, device="cpu")
    model = predictor.model.eval()

    class StatusBarOnnxWrapper(torch.nn.Module):
        def __init__(self, classifier: torch.nn.Module) -> None:
            super().__init__()
            self.classifier = classifier

        def forward(self, statusbar: torch.Tensor) -> torch.Tensor:
            return torch.softmax(self.classifier(statusbar), dim=1)

    wrapper = StatusBarOnnxWrapper(model).eval()
    sample = torch.zeros((1, 3, 64, 512), dtype=torch.float32)
    with torch.inference_mode():
        expected_outputs = tuple(item.detach().cpu().numpy() for item in (wrapper(sample),))
    temporary_path = _temporary_onnx_path(output_path)
    try:
        _legacy_onnx_export(
            torch,
            wrapper,
            sample,
            temporary_path,
            input_names=[STATUSBAR_INPUT_NAME],
            output_names=[STATUSBAR_OUTPUT_NAME],
            opset_version=opset_version,
        )
        onnx = _import_onnx()
        onnx.checker.check_model(str(temporary_path))
        if verify_runtime:
            _verify_runtime_graph(
                temporary_path,
                input_name=STATUSBAR_INPUT_NAME,
                sample=sample.numpy(),
                output_names=(STATUSBAR_OUTPUT_NAME,),
                expected_outputs=expected_outputs,
            )
        temporary_path.replace(output_path)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise

    return {
        "kind": "statusbar_device_v1",
        "input": {
            "name": STATUSBAR_INPUT_NAME,
            "shape": [1, 3, 64, 512],
            "dtype": "float32",
            "layout": "NCHW",
            "color_space": "RGB",
            "preprocess": "top 8% of source image, resize 512x64, /255, ImageNet mean/std normalize",
        },
        "outputs": [
            {"name": STATUSBAR_OUTPUT_NAME, "dtype": "float32", "shape": [1, 2], "labels": ["android", "ios"]},
        ],
        "postprocess": {
            "resolution_prior": True,
            "cnn_uncertain_threshold": 0.75,
            "crosscheck_conflict_threshold": 0.8,
        },
        "torch": {"version": torch.__version__},
    }


def export_onnx(
    *,
    kind: ExportKind,
    checkpoint: str | Path,
    output: str | Path,
    input_width: int = 864,
    input_height: int = 1536,
    resize_mode: str = "letterbox",
    sample_image: str | Path | None = None,
    opset_version: int = 17,
    verify_runtime: bool = True,
    overwrite: bool = False,
) -> tuple[Path, Path]:
    """Export one trusted checkpoint and return ``(onnx_path, contract_path)``."""

    if opset_version < 11:
        raise ValueError("opset_version must be at least 11 for Faster R-CNN export")
    if kind not in ("detector", "statusbar"):
        raise ValueError(f"Unsupported ONNX export kind: {kind}")
    checkpoint_path = Path(checkpoint).expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    output_path = Path(output).expanduser().resolve()
    if output_path.suffix.lower() != ".onnx":
        raise ValueError("ONNX output must have a .onnx extension")
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"ONNX output already exists: {output_path}. Pass --overwrite to replace it.")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if kind == "detector":
        if sample_image is None:
            raise ValueError("Detector export requires --sample-image with a representative receipt image")
        if input_width < 1 or input_height < 1:
            raise ValueError("input_width and input_height must be positive")
        sample_tensor = _load_sample_tensor(
            Path(sample_image).expanduser().resolve(),
            input_width=input_width,
            input_height=input_height,
            resize_mode=resize_mode,
        )
        details = _export_detector(
            checkpoint_path,
            output_path,
            sample_tensor=sample_tensor,
            opset_version=opset_version,
            verify_runtime=verify_runtime,
        )
        details["preprocess"] = {
            "resize_mode": resize_mode,
            "canvas_width": input_width,
            "canvas_height": input_height,
            "padding": "black" if resize_mode == "letterbox" else None,
            "restore_boxes": "remove padding and inverse resize to the rectified source image",
        }
    else:
        details = _export_statusbar(
            checkpoint_path,
            output_path,
            opset_version=opset_version,
            verify_runtime=verify_runtime,
        )

    contract_path = contract_path_for(output_path)
    contract = {
        "schema_version": 1,
        "source_checkpoint": {
            "filename": checkpoint_path.name,
            "sha256": _sha256(checkpoint_path),
            "size_bytes": checkpoint_path.stat().st_size,
        },
        "onnx": {
            "filename": output_path.name,
            "sha256": _sha256(output_path),
            "opset_version": opset_version,
        },
        "export_runtime": _export_runtime_versions(),
        **details,
    }
    _atomic_json(contract_path, contract)
    return output_path, contract_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export trusted receipt checkpoints to standard ONNX")
    parser.add_argument("--kind", choices=("detector", "statusbar"), default="detector")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Trusted .pt checkpoint to export")
    parser.add_argument("--output", type=Path, required=True, help="Destination .onnx path")
    parser.add_argument(
        "--sample-image",
        type=Path,
        help="Representative image used to trace the detector graph; required for --kind detector",
    )
    parser.add_argument("--input-width", type=int, default=864, help="Fixed detector input canvas width")
    parser.add_argument("--input-height", type=int, default=1536, help="Fixed detector input canvas height")
    parser.add_argument("--resize-mode", choices=("letterbox", "stretch"), default="letterbox")
    parser.add_argument("--opset", type=int, default=17, help="Target ONNX opset version (default: 17)")
    parser.add_argument(
        "--skip-runtime-verify",
        action="store_true",
        help="Skip the ONNX Runtime CPU load-and-run smoke check after export",
    )
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing ONNX artifact")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    try:
        onnx_path, metadata_path = export_onnx(
            kind=args.kind,
            checkpoint=args.checkpoint,
            output=args.output,
            sample_image=args.sample_image,
            input_width=args.input_width,
            input_height=args.input_height,
            resize_mode=args.resize_mode,
            opset_version=args.opset,
            verify_runtime=not args.skip_runtime_verify,
            overwrite=args.overwrite,
        )
    except (FileNotFoundError, FileExistsError, ImportError, RuntimeError, ValueError) as error:
        raise SystemExit(f"ONNX export failed: {error}") from None
    print(f"Wrote ONNX model: {onnx_path}")
    print(f"Wrote ONNX contract: {metadata_path}")


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
