"""Command-line runner for the first production receipt detector."""

from __future__ import annotations

import argparse
from pathlib import Path

from transfer_receipt_ai.infer import run_inference
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
    parser.add_argument("--input", type=Path, required=True, help="Input image or directory")
    parser.add_argument("--output", type=Path, required=True, help="Result directory")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:0, or mps")
    parser.add_argument("--score-threshold", type=float, default=0.50)
    parser.add_argument("--ocr", choices=("paddle", "none"), default="paddle")
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


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if not 0.0 <= args.score_threshold <= 1.0:
        raise SystemExit("--score-threshold must be between 0 and 1")
    if args.limit is not None and args.limit <= 0:
        raise SystemExit("--limit must be positive")

    try:
        checkpoint = resolve_checkpoint(args.model, args.checkpoint)
    except (ValueError, FileNotFoundError) as error:
        raise SystemExit(str(error)) from error

    outputs = run_inference(
        checkpoint=checkpoint,
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
    )
    print(f"Wrote {len(outputs)} inference result bundle(s) to {args.output}")


if __name__ == "__main__":  # pragma: no cover
    main()

