"""Command-line runner for the first production receipt detector."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from transfer_receipt_ai.prepare import parse_max_side

from .models import MODEL_SPECS, resolve_checkpoint


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a registered receipt model; receipt_lrcnn_v1 is the current default"
    )
    parser.add_argument("--model", choices=sorted(MODEL_SPECS), default="receipt_lrcnn_v1")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        help="Optional checkpoint override; normally just copy best.pt into the model directory",
    )
    parser.add_argument(
        "--onnx-model",
        type=Path,
        help="Use a fixed-shape ONNX detector instead of the registered PyTorch checkpoint",
    )
    platform_group = parser.add_mutually_exclusive_group()
    platform_group.add_argument(
        "--platform-checkpoint",
        type=Path,
        help="Optional status-bar device (Android/iOS) classifier checkpoint; adds a 'device' field",
    )
    platform_group.add_argument(
        "--platform-onnx-model",
        type=Path,
        help="Optional ONNX status-bar device classifier; adds a 'device' field",
    )
    parser.add_argument("--input", type=Path, required=True, help="Input image or directory")
    parser.add_argument("--output", type=Path, required=True, help="Result directory")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:0, or mps")
    parser.add_argument("--score-threshold", type=float, default=0.50)
    parser.add_argument(
        "--onnx-resize-mode",
        choices=("letterbox", "stretch"),
        default="letterbox",
        help="Fixed-canvas preprocessing for --onnx-model; must match its .contract.json",
    )
    parser.add_argument("--ocr", choices=("paddle", "none"), default="paddle")
    parser.add_argument(
        "--annotate",
        choices=("flagged", "all", "none"),
        default="all",
        help="标注哪些图:all=每张(默认,兼容旧行为) / flagged=只标缺核心字段、需复核的(更快、省盘) / none=都不标",
    )
    parser.add_argument("--require-complete", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--max-side",
        type=parse_max_side,
        default=1600,
        help="Maximum long edge after correction; use 0 to keep original resolution",
    )
    return parser


def _run_child(command: list[str]) -> None:
    """Run one framework-isolated worker and preserve its exit status."""
    completed = subprocess.run(command)
    if completed.returncode:
        raise SystemExit(completed.returncode)


def _detector_command(args: argparse.Namespace, model_path: Path, stage_output: Path) -> list[str]:
    """Build the detector-only child command using this exact virtualenv."""
    model_argument = "--onnx-model" if args.onnx_model is not None else "--checkpoint"
    command = [
        sys.executable,
        "-m",
        "transfer_receipt_ai.infer",
        model_argument,
        str(model_path),
        "--input",
        str(args.input),
        "--output",
        str(stage_output),
        "--device",
        str(args.device),
        "--score-threshold",
        str(args.score_threshold),
        "--onnx-resize-mode",
        str(args.onnx_resize_mode),
        "--ocr",
        "none",
        "--max-side",
        str(args.max_side),
        "--_write-ocr-stage-artifacts",
        "--_quiet",
    ]
    if args.platform_checkpoint is not None:
        command.extend(("--platform-checkpoint", str(args.platform_checkpoint)))
    if args.platform_onnx_model is not None:
        command.extend(("--platform-onnx-model", str(args.platform_onnx_model)))
    if args.require_complete:
        command.append("--require-complete")
    if args.continue_on_error:
        command.append("--continue-on-error")
    if args.skip_existing:
        command.extend(
            (
                "--skip-existing",
                "--_skip-existing-output",
                str(args.output),
                "--_require-ocr-complete-for-skip",
            )
        )
    if args.limit is not None:
        command.extend(("--limit", str(args.limit)))
    return command


def _ocr_command(args: argparse.Namespace, stage_output: Path) -> list[str]:
    """Build the Paddle-only child command using this exact virtualenv."""
    command = [
        sys.executable,
        "-m",
        "transfer_receipt_ai.ocr_enrich",
        "--stage-output",
        str(stage_output),
        "--output",
        str(args.output),
        "--device",
        str(args.device),
        "--_quiet",
    ]
    command.extend(("--annotate", str(args.annotate)))
    if args.continue_on_error:
        command.append("--continue-on-error")
    if args.skip_existing:
        command.append("--skip-existing")
    return command


def _read_final_manifest(output_dir: Path) -> list[dict[str, str]]:
    manifest_path = output_dir / "inference_manifest.json"
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"OCR worker did not write a readable manifest: {manifest_path}") from error
    if not isinstance(payload, list):
        raise RuntimeError(f"OCR worker wrote an invalid manifest: {manifest_path}")
    return payload


def _run_paddle_two_stage(args: argparse.Namespace, model_path: Path) -> list[dict[str, str]]:
    """Run detector and Paddle sequentially, never in the same Windows GPU process."""
    # Keep staged PNGs outside the input/output trees.  Otherwise a directory
    # input that contains its output folder could recursively ingest temporary
    # detector artifacts on the next scan.
    stage_output = Path(tempfile.mkdtemp(prefix="receipt_inference_stage_"))
    completed = False
    try:
        detector_name = "ONNX" if args.onnx_model is not None else "Torch"
        print(f"[1/2] Running {detector_name} detector...")
        _run_child(_detector_command(args, model_path, stage_output))
        print("[2/2] Running PaddleOCR...")
        _run_child(_ocr_command(args, stage_output))
        outputs = _read_final_manifest(args.output)
        completed = True
        return outputs
    finally:
        if completed:
            try:
                shutil.rmtree(stage_output)
            except OSError as error:
                print(f"WARNING: Could not remove temporary detector artifacts: {stage_output} ({error})", file=sys.stderr)
        else:
            print(f"Temporary detector artifacts retained for debugging: {stage_output}", file=sys.stderr)


def _needs_windows_gpu_split(args: argparse.Namespace) -> bool:
    """Only Windows CUDA needs DLL isolation; retain the existing other paths."""
    requested = str(args.device).lower()
    return os.name == "nt" and (requested == "auto" or requested == "cuda" or requested.startswith("cuda:"))


def _run_single_process(args: argparse.Namespace, model_path: Path) -> list[dict[str, str]]:
    """Preserve the existing non-Windows/CPU execution path."""
    from transfer_receipt_ai.infer import run_inference

    return run_inference(
        checkpoint=None if args.onnx_model is not None else model_path,
        onnx_model=model_path if args.onnx_model is not None else None,
        input_path=args.input,
        output_dir=args.output,
        device=args.device,
        score_threshold=args.score_threshold,
        use_ocr=args.ocr == "paddle",
        require_complete=args.require_complete,
        continue_on_error=args.continue_on_error,
        skip_existing=args.skip_existing,
        limit=args.limit,
        max_side=args.max_side,
        status_style_checkpoint=None,
        platform_checkpoint=args.platform_checkpoint,
        platform_onnx_model=args.platform_onnx_model,
        onnx_resize_mode=args.onnx_resize_mode,
        annotate=args.annotate,
    )


def _resolve_model_path(args: argparse.Namespace) -> Path:
    """Resolve exactly one detector artifact selected by the public CLI."""

    if args.onnx_model is None:
        return resolve_checkpoint(args.model, args.checkpoint)
    if args.checkpoint is not None:
        raise ValueError("--checkpoint and --onnx-model cannot be used together")
    model_path = args.onnx_model.expanduser()
    if not model_path.is_file():
        raise FileNotFoundError(f"ONNX model not found: {model_path}")
    return model_path.resolve()


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if not 0.0 <= args.score_threshold <= 1.0:
        raise SystemExit("--score-threshold must be between 0 and 1")
    if args.limit is not None and args.limit <= 0:
        raise SystemExit("--limit must be positive")

    try:
        model_path = _resolve_model_path(args)
    except (ValueError, FileNotFoundError) as error:
        raise SystemExit(str(error)) from error

    outputs = _run_paddle_two_stage(args, model_path) if (
        args.ocr == "paddle" and _needs_windows_gpu_split(args)
    ) else _run_single_process(args, model_path)
    print(f"Wrote {len(outputs)} inference result bundle(s) to {args.output}")


if __name__ == "__main__":  # pragma: no cover
    main()
