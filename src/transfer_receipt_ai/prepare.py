"""CLI for normalising image direction and rectifying photographed screens."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

from tqdm import tqdm

from .geometry import RectificationOptions, load_upright_rgb, rectify_receipt, save_rgb

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


def iter_image_paths(input_path: Path) -> Iterable[Path]:
    if input_path.is_file():
        if input_path.suffix.lower() not in IMAGE_SUFFIXES:
            raise ValueError(f"Unsupported image type: {input_path}")
        return [input_path]
    if not input_path.is_dir():
        raise FileNotFoundError(input_path)
    return sorted(path for path in input_path.rglob("*") if path.suffix.lower() in IMAGE_SUFFIXES)


def load_corrections(path: Path | None) -> dict[str, dict[str, Any]]:
    """Load an optional per-file override JSON object.

    Example::

        {
          "IMG_0001.jpg": {"orientation_degrees": 90},
          "batch/a.jpg": {"screen_quad": [[12, 18], [701, 9], [710, 1270], [7, 1264]]}
        }
    """
    if path is None:
        return {}
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("Correction file must be a JSON object keyed by image path")
    invalid = [key for key, value in raw.items() if not isinstance(key, str) or not isinstance(value, dict)]
    if invalid:
        raise ValueError("Every correction must map an image path to an object")
    return raw


def _find_override(corrections: dict[str, dict[str, Any]], relative_path: Path, source_path: Path) -> dict[str, Any]:
    return corrections.get(relative_path.as_posix(), corrections.get(source_path.name, {}))


def _build_orientation_scorer(enabled: bool):
    if not enabled:
        return None
    try:
        from .ocr import PaddleOCRReader

        reader = PaddleOCRReader()
    except ImportError as error:
        raise RuntimeError(
            "--ocr-orientation needs PaddleOCR. Install PaddlePaddle for your platform, then `pip install -r requirements-ocr.txt`."
        ) from error
    return reader.orientation_score


def prepare_images(
    input_path: Path,
    output_dir: Path,
    *,
    corrections: dict[str, dict[str, Any]] | None = None,
    orientation_degrees: int | None = None,
    prefer_portrait: bool = True,
    auto_screen: bool = True,
    max_side: int = 1600,
    use_ocr_orientation: bool = False,
    overwrite: bool = False,
) -> list[dict[str, Any]]:
    """Prepare every image and return the rectification manifest records."""
    corrections = corrections or {}
    image_paths = list(iter_image_paths(input_path))
    if not image_paths:
        raise ValueError(f"No supported images found under {input_path}")
    output_images = output_dir / "images"
    output_images.mkdir(parents=True, exist_ok=True)
    scorer = _build_orientation_scorer(use_ocr_orientation)
    root = input_path.parent if input_path.is_file() else input_path
    records: list[dict[str, Any]] = []

    for source_path in tqdm(image_paths, desc="Rectifying images"):
        relative_path = source_path.relative_to(root)
        override = _find_override(corrections, relative_path, source_path)
        if override.get("skip") is True:
            continue
        effective_orientation = override.get("orientation_degrees", orientation_degrees)
        options = RectificationOptions(
            orientation_degrees=effective_orientation,
            prefer_portrait=bool(override.get("prefer_portrait", prefer_portrait)),
            auto_screen=bool(override.get("auto_screen", auto_screen)),
            screen_quad=override.get("screen_quad"),
            max_side=int(override.get("max_side", max_side)),
            orientation_scorer=scorer,
        )
        source_rgb = load_upright_rgb(source_path)
        result = rectify_receipt(source_rgb, options)
        output_path = (output_images / relative_path).with_suffix(".jpg")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if output_path.exists() and not overwrite:
            raise FileExistsError(f"{output_path} already exists; pass --overwrite to replace it")
        save_rgb(output_path, result.rectified_rgb)
        record = {
            "source": source_path.resolve().as_posix(),
            "source_relative": relative_path.as_posix(),
            "rectified_image": output_path.relative_to(output_dir).as_posix(),
            **result.manifest(),
        }
        records.append(record)

    manifest_path = output_dir / "rectification_manifest.jsonl"
    with manifest_path.open("w", encoding="utf-8") as manifest_file:
        for record in records:
            manifest_file.write(json.dumps(record, ensure_ascii=False) + "\n")
    return records


def _parse_orientation(value: str) -> int | None:
    if value == "auto":
        return None
    degrees = int(value)
    if degrees not in {0, 90, 180, 270}:
        raise argparse.ArgumentTypeError("orientation must be auto, 0, 90, 180 or 270")
    return degrees


def parse_max_side(value: str) -> int:
    """Parse 0 as no resizing, otherwise require a usable pixel dimension."""
    max_side = int(value)
    if max_side < 0 or max_side == 1:
        raise argparse.ArgumentTypeError("max-side must be 0 (keep original resolution) or at least 2")
    return max_side


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="EXIF-correct, rotate and perspective-rectify receipt screenshots/photos."
    )
    parser.add_argument("--input", type=Path, required=True, help="One image or a folder of raw images")
    parser.add_argument("--output", type=Path, required=True, help="Prepared dataset output folder")
    parser.add_argument("--corrections", type=Path, help="Optional per-image correction JSON")
    parser.add_argument("--orientation", type=_parse_orientation, default=None, help="auto (default), 0, 90, 180, or 270")
    parser.add_argument("--landscape-ok", action="store_true", help="Do not rotate wide images to portrait by geometry")
    parser.add_argument("--no-auto-screen", action="store_true", help="Do not attempt automatic screen quadrilateral detection")
    parser.add_argument("--ocr-orientation", action="store_true", help="Use PaddleOCR to score all four text orientations")
    parser.add_argument(
        "--max-side",
        type=parse_max_side,
        default=1600,
        help="Maximum long edge after correction; use 0 to keep original resolution",
    )
    parser.add_argument("--overwrite", action="store_true", help="Replace existing prepared images")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    records = prepare_images(
        args.input,
        args.output,
        corrections=load_corrections(args.corrections),
        orientation_degrees=args.orientation,
        prefer_portrait=not args.landscape_ok,
        auto_screen=not args.no_auto_screen,
        max_side=args.max_side,
        use_ocr_orientation=args.ocr_orientation,
        overwrite=args.overwrite,
    )
    print(f"Prepared {len(records)} image(s) in {args.output}")


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
