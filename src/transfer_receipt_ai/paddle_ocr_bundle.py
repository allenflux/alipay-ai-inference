"""Freeze and export the exact PaddleOCR assets used by this project.

The public receipt CLI deliberately lets PaddleOCR 2.x resolve its default
Chinese models on first use.  That is convenient for development, but it is
not a reproducible .NET delivery boundary: a later install can resolve a
different detector, recognizer, classifier, or character dictionary.

This module turns the *currently effective* PaddleOCR runtime into an audited
bundle before converting it to ONNX:

``snapshot``
    copies the three Paddle static-graph models plus the exact recognition
    dictionary, and records all effective PaddleOCR arguments, package
    versions, sizes and SHA-256 values.

``export-onnx``
    invokes Paddle2ONNX without an input-shape override.  OCR graphs must keep
    their dynamic dimensions for parity with the current PaddleOCR runtime.

``verify``
    rechecks every recorded hash before a bundle is handed to the .NET port.

The source Paddle files are kept for audit and reproducible conversion only.
The eventual deployment package contains ``onnx/``, ``charset/`` and the
contract; it must not require Python, Paddle, or PaddleOCR at runtime.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Mapping, Sequence
from importlib import metadata
from pathlib import Path
from time import perf_counter
from typing import Any


BUNDLE_SCHEMA_VERSION = 1
BUNDLE_KIND = "paddle_ocr_v2_bundle"
DELIVERY_KIND = "paddle_ocr_v2_delivery"
CONTRACT_FILENAME = "paddle_ocr_bundle.contract.json"
DELIVERY_CONTRACT_FILENAME = "paddle_ocr_delivery.contract.json"
MODEL_ROLES = ("det", "rec", "cls")
REQUIRED_MODEL_FILENAMES = ("inference.pdmodel", "inference.pdiparams")
NATIVE_IDENTITY_KIND = "paddle_ocr_native_asset_identity_v1"
ADAPTER_VERSION = "paddle_ocr_dotnet_adapter_v1"


class PaddleOcrBundleError(ValueError):
    """Raised when a Paddle OCR bundle is incomplete or has been modified."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _file_record(path: Path, *, relative_to: Path) -> dict[str, object]:
    return {
        "path": path.relative_to(relative_to).as_posix(),
        "sha256": _sha256(path),
        "size_bytes": path.stat().st_size,
    }


def _all_file_records(directory: Path, *, relative_to: Path) -> list[dict[str, object]]:
    return [_file_record(path, relative_to=relative_to) for path in sorted(directory.rglob("*")) if path.is_file()]


def _canonical_json_sha256(value: Mapping[str, object]) -> str:
    payload = json.dumps(dict(value), ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _native_asset_identity(
    assets: Mapping[str, object], dictionary: Mapping[str, object]
) -> dict[str, object]:
    """Return a path-independent identity for the frozen native OCR bytes.

    Source cache paths are intentionally excluded.  The identity travels into
    evaluation and delivery evidence, so the exact det/rec/cls/dictionary
    bytes remain comparable after the native Paddle files are omitted from the
    lean runtime package.
    """

    files: list[dict[str, object]] = []
    for role in MODEL_ROLES:
        asset = assets.get(role)
        records = asset.get("files") if isinstance(asset, Mapping) else None
        if not isinstance(records, list) or not records:
            raise PaddleOcrBundleError(f"Cannot identify missing/invalid frozen {role} files")
        for record in records:
            if not isinstance(record, Mapping):
                raise PaddleOcrBundleError(f"Cannot identify invalid frozen {role} file record")
            path = record.get("path")
            sha256 = record.get("sha256")
            size = record.get("size_bytes")
            if not isinstance(path, str) or not isinstance(sha256, str) or not isinstance(size, int):
                raise PaddleOcrBundleError(f"Cannot identify invalid frozen {role} file record")
            files.append({"role": role, "path": path, "sha256": sha256.lower(), "size_bytes": size})
    dictionary_path = dictionary.get("path")
    dictionary_hash = dictionary.get("sha256")
    dictionary_size = dictionary.get("size_bytes")
    if (
        not isinstance(dictionary_path, str)
        or not isinstance(dictionary_hash, str)
        or not isinstance(dictionary_size, int)
    ):
        raise PaddleOcrBundleError("Cannot identify invalid frozen character dictionary")
    files.append(
        {
            "role": "dictionary",
            "path": dictionary_path,
            "sha256": dictionary_hash.lower(),
            "size_bytes": dictionary_size,
        }
    )
    ordered_files = sorted(files, key=lambda record: (str(record["role"]), str(record["path"])))
    components: dict[str, str] = {}
    for role in (*MODEL_ROLES, "dictionary"):
        role_files = [record for record in ordered_files if record["role"] == role]
        components[role] = _canonical_json_sha256({"role": role, "files": role_files})
    unsigned: dict[str, object] = {
        "schema_version": 1,
        "kind": NATIVE_IDENTITY_KIND,
        "components": components,
        "files": ordered_files,
    }
    return {**unsigned, "sha256": _canonical_json_sha256(unsigned)}


def _verify_native_asset_identity(contract: Mapping[str, Any]) -> dict[str, object]:
    assets = contract.get("assets")
    dictionary = contract.get("dictionary")
    claimed = contract.get("native_asset_identity")
    if not isinstance(assets, Mapping) or not isinstance(dictionary, Mapping) or not isinstance(claimed, Mapping):
        raise PaddleOcrBundleError("Bundle contract has no native asset identity")
    expected = _native_asset_identity(assets, dictionary)
    if dict(claimed) != expected:
        raise PaddleOcrBundleError("Bundle native asset identity differs from its frozen file records")
    return expected


def _verify_carried_native_asset_identity(value: object) -> dict[str, object]:
    """Validate an identity embedded in a lean package without native files."""

    if not isinstance(value, Mapping):
        raise PaddleOcrBundleError("Delivery contract has no native asset identity")
    schema = value.get("schema_version")
    kind = value.get("kind")
    files = value.get("files")
    components = value.get("components")
    claimed_hash = value.get("sha256")
    if (
        schema != 1
        or kind != NATIVE_IDENTITY_KIND
        or not isinstance(files, list)
        or not files
        or not isinstance(components, Mapping)
    ):
        raise PaddleOcrBundleError("Delivery contract has an invalid native asset identity")
    normalized: list[dict[str, object]] = []
    for record in files:
        if not isinstance(record, Mapping):
            raise PaddleOcrBundleError("Delivery native asset identity has an invalid file record")
        role = record.get("role")
        path = record.get("path")
        sha256 = record.get("sha256")
        size = record.get("size_bytes")
        if (
            role not in {*MODEL_ROLES, "dictionary"}
            or not isinstance(path, str)
            or not isinstance(sha256, str)
            or len(sha256) != 64
            or any(character not in "0123456789abcdef" for character in sha256)
            or not isinstance(size, int)
            or size < 0
        ):
            raise PaddleOcrBundleError("Delivery native asset identity has an invalid file record")
        normalized.append({"role": role, "path": path, "sha256": sha256, "size_bytes": size})
    if normalized != sorted(normalized, key=lambda record: (str(record["role"]), str(record["path"]))):
        raise PaddleOcrBundleError("Delivery native asset identity files are not canonical")
    expected_components: dict[str, str] = {}
    for role in (*MODEL_ROLES, "dictionary"):
        role_files = [record for record in normalized if record["role"] == role]
        expected_components[role] = _canonical_json_sha256({"role": role, "files": role_files})
    if dict(components) != expected_components:
        raise PaddleOcrBundleError("Delivery native asset identity component hashes are invalid")
    unsigned: dict[str, object] = {
        "schema_version": 1,
        "kind": NATIVE_IDENTITY_KIND,
        "components": expected_components,
        "files": normalized,
    }
    if not isinstance(claimed_hash, str) or claimed_hash != _canonical_json_sha256(unsigned):
        raise PaddleOcrBundleError("Delivery native asset identity SHA-256 is invalid")
    return {**unsigned, "sha256": claimed_hash}


def _adapter_contract() -> dict[str, object]:
    return {
        "adapter_version": ADAPTER_VERSION,
        # The current Python pipeline supplies an RGB ndarray from
        # Pillow/OpenCV geometry code directly to PaddleOCR v2.  This wording
        # prevents an unverified RGB/BGR swap in the C# adapter.
        "input_color_order": "RGB_passthrough_to_paddle_v2",
        "line_aggregation": {
            "text": "clean each non-empty line then join with one ASCII space",
            "confidence": "arithmetic mean of all Paddle line confidences",
        },
        "preprocessing": {
            "detector_normalization": {
                "scale": 0.00392156862745098,
                "mean": [0.485, 0.456, 0.406],
                "std": [0.229, 0.224, 0.225],
            },
            "classifier_recognizer_normalization": {
                "scale": 0.00392156862745098,
                "mean": [0.5, 0.5, 0.5],
                "std": [0.5, 0.5, 0.5],
            },
            "classifier_recognizer_right_padding": "float_zero_after_normalization",
        },
        "hardware_note": "Snapshot initialisation may run on CPU; device-selection args are not OCR behavior parity settings.",
        "required_components": ["text_detection", "angle_classification", "text_recognition"],
    }


def _verify_adapter_contract(value: object, *, description: str) -> dict[str, object]:
    expected = _adapter_contract()
    if not isinstance(value, Mapping) or dict(value) != expected:
        raise PaddleOcrBundleError(f"{description} has an unsupported Paddle OCR adapter contract")
    return expected


def _atomic_json(path: Path, value: Mapping[str, object]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(dict(value), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _atomic_jsonl(path: Path, values: Sequence[Mapping[str, object]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        for value in values:
            stream.write(json.dumps(dict(value), ensure_ascii=False) + "\n")
    temporary.replace(path)


def _json_safe(value: Any) -> Any:
    """Convert Paddle's argparse values into stable, serialisable JSON."""

    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    return str(value)


def _package_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def _require_file(path: Path, *, description: str) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise PaddleOcrBundleError(f"{description} does not exist: {resolved}")
    return resolved


def _require_model_directory(role: str, path: Path) -> Path:
    directory = path.expanduser().resolve()
    if not directory.is_dir():
        raise PaddleOcrBundleError(f"{role} model directory does not exist: {directory}")
    missing = [name for name in REQUIRED_MODEL_FILENAMES if not (directory / name).is_file()]
    if missing:
        raise PaddleOcrBundleError(f"{role} model directory is missing {', '.join(missing)}: {directory}")
    return directory


def _ensure_output_does_not_overlap_sources(output_dir: Path, sources: Sequence[Path]) -> None:
    output = output_dir.expanduser().resolve()
    for source in sources:
        resolved_source = source.expanduser().resolve()
        try:
            output.relative_to(resolved_source)
        except ValueError:
            pass
        else:
            raise PaddleOcrBundleError(
                f"Bundle output must not be inside a source asset directory: {output} is within {resolved_source}"
            )


def _preflight_default_v2_assets(*, allow_model_download: bool) -> None:
    """Avoid a snapshot command silently fetching a newer default model set."""

    if allow_model_download:
        return
    base = Path(os.environ.get("PADDLE_OCR_BASE_DIR", str(Path.home() / ".paddleocr"))).expanduser()
    expected = {
        "det": base / "whl" / "det" / "ch" / "ch_PP-OCRv4_det_infer",
        "rec": base / "whl" / "rec" / "ch" / "ch_PP-OCRv4_rec_infer",
        "cls": base / "whl" / "cls" / "ch_ppocr_mobile_v2.0_cls_infer",
    }
    missing = [
        f"{role}:{directory}"
        for role, directory in expected.items()
        if any(not (directory / filename).is_file() for filename in REQUIRED_MODEL_FILENAMES)
    ]
    if missing:
        raise PaddleOcrBundleError(
            "Refusing to initialise PaddleOCR because its expected v2.10 assets are not already cached. "
            "A snapshot must not silently download a possibly different model version. Missing: "
            + "; ".join(missing)
            + ". Pass all explicit model paths, or use --allow-model-download only when a new download is intentional."
        )


def _effective_runtime_assets(
    *, device: str, allow_model_download: bool
) -> tuple[dict[str, Path], Path, dict[str, Any], dict[str, object]]:
    """Initialise the project's real reader and capture its resolved v2 assets."""

    _preflight_default_v2_assets(allow_model_download=allow_model_download)

    from .ocr import PaddleOCRReader

    reader = PaddleOCRReader(device=device, require_v2=True)
    if getattr(reader, "_api_version", None) != 2:
        raise PaddleOcrBundleError("Only the current PaddleOCR 2.x runtime may be frozen for this delivery")
    engine = reader._engine  # PaddleOCR v2 public operation is backed by this stable args Namespace.
    raw_args = getattr(engine, "args", None)
    if raw_args is None:
        raise PaddleOcrBundleError("PaddleOCR did not expose effective v2 runtime arguments")
    args = vars(raw_args)
    required_paths = {
        "det": "det_model_dir",
        "rec": "rec_model_dir",
        "cls": "cls_model_dir",
    }
    model_dirs: dict[str, Path] = {}
    for role, key in required_paths.items():
        value = args.get(key)
        if not isinstance(value, str) or not value:
            raise PaddleOcrBundleError(f"Effective PaddleOCR args have no {key}")
        model_dirs[role] = _require_model_directory(role, Path(value))
    charset_raw = args.get("rec_char_dict_path")
    if not isinstance(charset_raw, str) or not charset_raw:
        raise PaddleOcrBundleError("Effective PaddleOCR args have no rec_char_dict_path")
    charset = _require_file(Path(charset_raw), description="PaddleOCR recognition character dictionary")

    try:
        import paddle
    except ModuleNotFoundError:  # pragma: no cover - PaddleOCRReader already imports Paddle.
        paddle_version: str | None = None
    else:
        paddle_version = str(getattr(paddle, "__version__", "")) or None
    runtime = {
        "paddleocr_version": _package_version("paddleocr"),
        "paddle_version": paddle_version,
        "python_version": sys.version.split()[0],
        "platform": sys.platform,
        "resolved_by": "transfer_receipt_ai.ocr.PaddleOCRReader(require_v2=True)",
        "snapshot_initialization_device": device,
        "paddle_ocr_base_dir": os.environ.get("PADDLE_OCR_BASE_DIR", str(Path.home() / ".paddleocr")),
    }
    return model_dirs, charset, {str(key): _json_safe(value) for key, value in args.items()}, runtime


def snapshot_bundle(
    *,
    output_dir: Path,
    model_dirs: Mapping[str, Path],
    charset_path: Path,
    effective_args: Mapping[str, Any],
    runtime: Mapping[str, object],
) -> Path:
    """Copy immutable Paddle assets into ``output_dir`` and create a contract."""

    missing_roles = sorted(set(MODEL_ROLES) - set(model_dirs))
    extra_roles = sorted(set(model_dirs) - set(MODEL_ROLES))
    if missing_roles or extra_roles:
        raise PaddleOcrBundleError(f"Model roles must be {MODEL_ROLES}; missing={missing_roles}, extra={extra_roles}")
    validated_models = {role: _require_model_directory(role, Path(model_dirs[role])) for role in MODEL_ROLES}
    charset = _require_file(Path(charset_path), description="PaddleOCR recognition character dictionary")
    output = output_dir.expanduser().resolve()
    _ensure_output_does_not_overlap_sources(output, [*validated_models.values(), charset])
    if output.exists():
        raise PaddleOcrBundleError(
            f"Refusing to overwrite existing bundle output: {output}. Choose a new directory so the audit snapshot stays immutable."
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    try:
        assets: dict[str, object] = {}
        for role, source in validated_models.items():
            destination = stage / "paddle" / role
            shutil.copytree(source, destination)
            files = _all_file_records(destination, relative_to=stage)
            assets[role] = {
                "source_directory": str(source),
                "bundle_directory": destination.relative_to(stage).as_posix(),
                "files": files,
                "size_bytes": sum(int(record["size_bytes"]) for record in files),
            }
        dictionary_destination = stage / "charset" / "ppocr_keys_v1.txt"
        dictionary_destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(charset, dictionary_destination)
        dictionary = _file_record(dictionary_destination, relative_to=stage)
        dictionary["source_path"] = str(charset)
        native_identity = _native_asset_identity(assets, dictionary)
        contract: dict[str, object] = {
            "schema_version": BUNDLE_SCHEMA_VERSION,
            "kind": BUNDLE_KIND,
            "runtime": dict(runtime),
            "effective_paddleocr_args": {str(key): _json_safe(value) for key, value in effective_args.items()},
            "hardware_dependent_paddleocr_args": [
                "use_gpu",
                "gpu_id",
                "gpu_mem",
                "cpu_threads",
                "enable_mkldnn",
                "use_tensorrt",
                "precision",
                "ir_optim",
            ],
            "assets": assets,
            "dictionary": dictionary,
            "native_asset_identity": native_identity,
            "onnx": {},
            "adapter_contract": _adapter_contract(),
            "delivery_layout": {
                "generated_by": "package-delivery",
                "required": [
                    "onnx/paddle_ocr_det.onnx",
                    "onnx/paddle_ocr_rec.onnx",
                    "onnx/paddle_ocr_cls.onnx",
                    dictionary["path"],
                    DELIVERY_CONTRACT_FILENAME,
                ],
                "exclude_from_delivery": ["paddle/"],
            },
        }
        _atomic_json(stage / CONTRACT_FILENAME, contract)
        stage.replace(output)
    except Exception:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    return output


def _load_contract(bundle_dir: Path) -> tuple[Path, dict[str, Any]]:
    bundle = bundle_dir.expanduser().resolve()
    contract_path = bundle / CONTRACT_FILENAME
    if not contract_path.is_file():
        raise PaddleOcrBundleError(f"Paddle OCR bundle contract does not exist: {contract_path}")
    try:
        value = json.loads(contract_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise PaddleOcrBundleError(f"Invalid Paddle OCR bundle contract: {error}") from None
    if not isinstance(value, dict):
        raise PaddleOcrBundleError("Paddle OCR bundle contract must be a JSON object")
    if value.get("schema_version") != BUNDLE_SCHEMA_VERSION or value.get("kind") != BUNDLE_KIND:
        raise PaddleOcrBundleError("Unsupported Paddle OCR bundle contract schema")
    return bundle, value


def _verify_file_record(bundle: Path, record: Mapping[str, Any]) -> None:
    relative = record.get("path")
    expected_hash = record.get("sha256")
    expected_size = record.get("size_bytes")
    if not isinstance(relative, str) or not isinstance(expected_hash, str) or not isinstance(expected_size, int):
        raise PaddleOcrBundleError("Bundle contract contains an invalid file record")
    candidate = (bundle / relative).resolve()
    try:
        candidate.relative_to(bundle)
    except ValueError:
        raise PaddleOcrBundleError(f"Bundle contract attempts to read outside bundle: {relative}") from None
    if not candidate.is_file():
        raise PaddleOcrBundleError(f"Bundle file is missing: {candidate}")
    if candidate.stat().st_size != expected_size:
        raise PaddleOcrBundleError(f"Bundle file size differs from contract: {candidate}")
    if _sha256(candidate) != expected_hash:
        raise PaddleOcrBundleError(f"Bundle file SHA-256 differs from contract: {candidate}")


def verify_bundle(bundle_dir: Path, *, require_onnx: bool = False) -> dict[str, Any]:
    """Validate all recorded source and ONNX artifact hashes in a snapshot."""

    bundle, contract = _load_contract(bundle_dir)
    assets = contract.get("assets")
    if not isinstance(assets, Mapping):
        raise PaddleOcrBundleError("Bundle contract has no model assets")
    for role in MODEL_ROLES:
        asset = assets.get(role)
        if not isinstance(asset, Mapping):
            raise PaddleOcrBundleError(f"Bundle contract has no {role} asset")
        files = asset.get("files")
        if not isinstance(files, list):
            raise PaddleOcrBundleError(f"Bundle contract has invalid {role} files")
        for record in files:
            if not isinstance(record, Mapping):
                raise PaddleOcrBundleError(f"Bundle contract has invalid {role} file record")
            _verify_file_record(bundle, record)
    dictionary = contract.get("dictionary")
    if not isinstance(dictionary, Mapping):
        raise PaddleOcrBundleError("Bundle contract has no character dictionary")
    _verify_file_record(bundle, dictionary)
    _verify_native_asset_identity(contract)
    _verify_adapter_contract(contract.get("adapter_contract"), description="Bundle contract")
    onnx = contract.get("onnx")
    if not isinstance(onnx, Mapping):
        raise PaddleOcrBundleError("Bundle contract has invalid ONNX records")
    if require_onnx and set(onnx) != set(MODEL_ROLES):
        raise PaddleOcrBundleError("Bundle has not exported all det, rec and cls ONNX models")
    for role, record in onnx.items():
        if role not in MODEL_ROLES or not isinstance(record, Mapping):
            raise PaddleOcrBundleError("Bundle contract has an invalid ONNX model record")
        _verify_file_record(bundle, record)
    return contract


def package_delivery_bundle(*, bundle_dir: Path, output_dir: Path) -> Path:
    """Create a Python/Paddle-free delivery folder from a verified audit bundle.

    The audit bundle intentionally retains ``paddle/`` so it remains
    reproducible.  This separate command is the only supported way to produce
    the lean deployment folder, preventing users from deleting source files
    and invalidating the audit contract in place.
    """

    audit_contract = verify_bundle(bundle_dir, require_onnx=True)
    bundle, _ = _load_contract(bundle_dir)
    output = output_dir.expanduser().resolve()
    _ensure_output_does_not_overlap_sources(output, [bundle])
    if output.exists():
        raise PaddleOcrBundleError(f"Refusing to overwrite existing delivery output: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    try:
        raw_onnx = audit_contract.get("onnx")
        raw_dictionary = audit_contract.get("dictionary")
        if not isinstance(raw_onnx, Mapping) or not isinstance(raw_dictionary, Mapping):  # defensive after verify_bundle
            raise PaddleOcrBundleError("Verified audit bundle has no ONNX or dictionary records")
        models: dict[str, dict[str, object]] = {}
        for role in MODEL_ROLES:
            raw_record = raw_onnx[role]
            if not isinstance(raw_record, Mapping):  # defensive after verify_bundle
                raise PaddleOcrBundleError(f"Verified audit bundle has no {role} ONNX record")
            source_relative = raw_record.get("path")
            if not isinstance(source_relative, str):
                raise PaddleOcrBundleError(f"Invalid {role} ONNX path in audit contract")
            source = bundle / source_relative
            destination = stage / "onnx" / source.name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
            record = _file_record(destination, relative_to=stage)
            for key in ("io", "dynamic_shape_validation"):
                if key in raw_record:
                    record[key] = raw_record[key]
            models[role] = record
        dictionary_relative = raw_dictionary.get("path")
        if not isinstance(dictionary_relative, str):
            raise PaddleOcrBundleError("Invalid character dictionary path in audit contract")
        dictionary_source = bundle / dictionary_relative
        dictionary_destination = stage / "charset" / dictionary_source.name
        dictionary_destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(dictionary_source, dictionary_destination)
        dictionary = _file_record(dictionary_destination, relative_to=stage)

        raw_args = audit_contract.get("effective_paddleocr_args")
        normalized_args = dict(raw_args) if isinstance(raw_args, Mapping) else {}
        normalized_args.update(
            {
                "det_model_dir": models["det"]["path"],
                "rec_model_dir": models["rec"]["path"],
                "cls_model_dir": models["cls"]["path"],
                "rec_char_dict_path": dictionary["path"],
                "use_onnx": True,
                "use_angle_cls": True,
            }
        )
        for key in audit_contract.get("hardware_dependent_paddleocr_args", []):
            if isinstance(key, str):
                normalized_args.pop(key, None)
        raw_runtime = audit_contract.get("runtime")
        delivery_runtime = (
            {
                key: value
                for key, value in raw_runtime.items()
                if key not in {"paddle_ocr_base_dir", "snapshot_initialization_device"}
            }
            if isinstance(raw_runtime, Mapping)
            else {}
        )
        delivery_contract: dict[str, object] = {
            "schema_version": BUNDLE_SCHEMA_VERSION,
            "kind": DELIVERY_KIND,
            "source_audit_contract_sha256": _sha256(bundle / CONTRACT_FILENAME),
            "native_asset_identity": audit_contract["native_asset_identity"],
            "runtime": delivery_runtime,
            "effective_paddleocr_args": normalized_args,
            "adapter_contract": audit_contract.get("adapter_contract", {}),
            "models": models,
            "dictionary": dictionary,
            "package_size_bytes": sum(int(record["size_bytes"]) for record in models.values()) + int(dictionary["size_bytes"]),
            "runtime_dependencies": ["ONNX Runtime", "OpenCV-compatible image processing for the OCR adapter"],
            "forbidden_runtime_dependencies": ["Python", "PaddlePaddle", "PaddleOCR", "paddle static graph files"],
        }
        _atomic_json(stage / DELIVERY_CONTRACT_FILENAME, delivery_contract)
        stage.replace(output)
    except Exception:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    return output


def verify_delivery_bundle(delivery_dir: Path) -> dict[str, Any]:
    """Verify a lean delivery package without requiring its audit source files."""

    delivery = delivery_dir.expanduser().resolve()
    contract_path = delivery / DELIVERY_CONTRACT_FILENAME
    if not contract_path.is_file():
        raise PaddleOcrBundleError(f"Paddle OCR delivery contract does not exist: {contract_path}")
    try:
        contract = json.loads(contract_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise PaddleOcrBundleError(f"Invalid Paddle OCR delivery contract: {error}") from None
    if not isinstance(contract, dict) or contract.get("schema_version") != BUNDLE_SCHEMA_VERSION:
        raise PaddleOcrBundleError("Unsupported Paddle OCR delivery contract schema")
    if contract.get("kind") != DELIVERY_KIND:
        raise PaddleOcrBundleError("Not a Paddle OCR delivery contract")
    _verify_carried_native_asset_identity(contract.get("native_asset_identity"))
    _verify_adapter_contract(contract.get("adapter_contract"), description="Delivery contract")
    models = contract.get("models")
    if not isinstance(models, Mapping) or set(models) != set(MODEL_ROLES):
        raise PaddleOcrBundleError("Delivery contract must contain det, rec and cls ONNX models")
    for role in MODEL_ROLES:
        record = models[role]
        if not isinstance(record, Mapping):
            raise PaddleOcrBundleError(f"Delivery contract has invalid {role} model record")
        _verify_file_record(delivery, record)
    dictionary = contract.get("dictionary")
    if not isinstance(dictionary, Mapping):
        raise PaddleOcrBundleError("Delivery contract has no character dictionary")
    _verify_file_record(delivery, dictionary)
    forbidden = (delivery / "paddle").exists()
    if forbidden:
        raise PaddleOcrBundleError("Delivery package must not contain the audit paddle/ source directory")
    return contract


def _paddle_onnx_options(bundle: Path, contract: Mapping[str, Any]) -> dict[str, object]:
    """Construct the v2 ``use_onnx`` reader from frozen effective arguments."""

    raw_args = contract.get("effective_paddleocr_args")
    if not isinstance(raw_args, Mapping):
        raise PaddleOcrBundleError("Bundle contract has no effective PaddleOCR arguments")
    raw_onnx = contract.get("onnx")
    raw_dictionary = contract.get("dictionary")
    if not isinstance(raw_onnx, Mapping) or not isinstance(raw_dictionary, Mapping):
        raise PaddleOcrBundleError("Bundle contract has no ONNX model or dictionary records")
    selected_keys = (
        "lang",
        "ocr_version",
        "det_algorithm",
        "det_limit_side_len",
        "det_limit_type",
        "det_box_type",
        "det_db_thresh",
        "det_db_box_thresh",
        "det_db_unclip_ratio",
        "det_db_score_mode",
        "use_dilation",
        "max_batch_size",
        "rec_algorithm",
        "rec_image_shape",
        "rec_batch_num",
        "max_text_length",
        "use_space_char",
        "drop_score",
        "cls_image_shape",
        "cls_batch_num",
        "cls_thresh",
    )
    options: dict[str, object] = {key: raw_args[key] for key in selected_keys if key in raw_args}
    options["lang"] = str(options.get("lang", "ch"))
    options["ocr_version"] = str(options.get("ocr_version", "PP-OCRv4"))
    options.update(
        {
            "use_angle_cls": True,
            "use_gpu": False,
            "gpu_id": 0,
            "show_log": False,
            "use_onnx": True,
            "onnx_providers": ["CPUExecutionProvider"],
        }
    )
    for role, parameter in (("det", "det_model_dir"), ("rec", "rec_model_dir"), ("cls", "cls_model_dir")):
        record = raw_onnx.get(role)
        if not isinstance(record, Mapping) or not isinstance(record.get("path"), str):
            raise PaddleOcrBundleError(f"Bundle contract has no {role} ONNX model record")
        options[parameter] = str(bundle / str(record["path"]))
    dictionary_path = raw_dictionary.get("path")
    if not isinstance(dictionary_path, str):
        raise PaddleOcrBundleError("Bundle contract has an invalid dictionary record")
    options["rec_char_dict_path"] = str(bundle / dictionary_path)
    return options


def _paddle_native_options(bundle: Path, contract: Mapping[str, Any]) -> dict[str, object]:
    """Construct a CPU reader from the source bytes frozen in ``bundle``.

    Converter parity must never compare the exported ONNX files with whatever
    model happens to be in PaddleOCR's mutable user cache.  Start from the
    same behavior arguments as the ONNX reader, then bind det/rec/cls and the
    dictionary to the snapshot itself.
    """

    options = _paddle_onnx_options(bundle, contract)
    raw_assets = contract.get("assets")
    raw_dictionary = contract.get("dictionary")
    if not isinstance(raw_assets, Mapping) or not isinstance(raw_dictionary, Mapping):
        raise PaddleOcrBundleError("Bundle contract has no native model or dictionary records")
    options["use_onnx"] = False
    options.pop("onnx_providers", None)
    for role, parameter in (("det", "det_model_dir"), ("rec", "rec_model_dir"), ("cls", "cls_model_dir")):
        record = raw_assets.get(role)
        directory = record.get("bundle_directory") if isinstance(record, Mapping) else None
        if not isinstance(directory, str):
            raise PaddleOcrBundleError(f"Bundle contract has no frozen {role} model directory")
        options[parameter] = str(bundle / directory)
    dictionary_path = raw_dictionary.get("path")
    if not isinstance(dictionary_path, str):
        raise PaddleOcrBundleError("Bundle contract has an invalid dictionary record")
    options["rec_char_dict_path"] = str(bundle / dictionary_path)
    return options


def _create_paddle_native_reader(bundle_dir: Path) -> Any:
    """Create the CPU reference reader from immutable snapshot contents."""

    bundle, contract = _load_contract(bundle_dir)
    verify_bundle(bundle, require_onnx=True)
    if _package_version("paddleocr") != "2.10.0":
        raise PaddleOcrBundleError("ONNX converter parity validation requires paddleocr==2.10.0")
    try:
        import paddle
        from paddleocr import PaddleOCR
    except ModuleNotFoundError as error:
        raise PaddleOcrBundleError("ONNX converter parity validation requires paddlepaddle and paddleocr==2.10.0") from error
    paddle.set_device("cpu")
    return PaddleOCR(**_paddle_native_options(bundle, contract))


def _create_paddle_onnx_reader(bundle_dir: Path) -> Any:
    """Create a CPU v2 PaddleOCR wrapper using the just-exported ONNX files.

    This exists only for converter parity validation.  The final delivery does
    not retain this PaddleOCR dependency; the C# adapter will use the same
    files directly through ONNX Runtime.
    """

    bundle, contract = _load_contract(bundle_dir)
    verify_bundle(bundle, require_onnx=True)
    if _package_version("paddleocr") != "2.10.0":
        raise PaddleOcrBundleError("ONNX converter parity validation requires paddleocr==2.10.0")
    try:
        import paddle
        from paddleocr import PaddleOCR
    except ModuleNotFoundError as error:
        raise PaddleOcrBundleError("ONNX converter parity validation requires paddlepaddle and paddleocr==2.10.0") from error
    paddle.set_device("cpu")
    return PaddleOCR(**_paddle_onnx_options(bundle, contract))


def _ocr_result_from_paddle_payload(payload: Any) -> tuple[str, float | None, list[dict[str, object]]]:
    # Import the production parser rather than recreating a subtly different
    # interpretation of PaddleOCR v2's nested line result.
    from .ocr import _extract_paddle_lines, clean_text

    lines = _extract_paddle_lines(payload)
    line_records = [{"text": clean_text(text), "confidence": confidence} for text, confidence in lines if clean_text(text)]
    if not line_records:
        return "", None, []
    text = " ".join(str(line["text"]) for line in line_records)
    confidence = sum(float(line["confidence"]) for line in line_records) / len(line_records)
    return text, confidence, line_records


_IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


def _iter_ocr_validation_images(input_path: Path, *, limit: int | None) -> list[Path]:
    source = input_path.expanduser().resolve()
    if source.is_file():
        candidates = [source]
    elif source.is_dir():
        candidates = [path for path in sorted(source.rglob("*")) if path.is_file() and path.suffix.lower() in _IMAGE_SUFFIXES]
    else:
        raise PaddleOcrBundleError(f"OCR validation input does not exist: {source}")
    if not candidates:
        raise PaddleOcrBundleError(f"No supported images found under: {source}")
    if limit is not None:
        if limit <= 0:
            raise PaddleOcrBundleError("--limit must be positive")
        candidates = candidates[:limit]
    return candidates


def _percentile(values: Sequence[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = round((len(ordered) - 1) * percentile)
    return ordered[index]


def validate_onnx_conversion(
    *,
    bundle_dir: Path,
    input_path: Path,
    output_dir: Path,
    limit: int | None = None,
    min_text_exact_match: float = 1.0,
    max_confidence_delta: float = 0.01,
) -> tuple[dict[str, object], bool]:
    """Compare frozen native PaddleOCR and its ONNX export on identical RGB crops."""

    if not 0.0 <= min_text_exact_match <= 1.0:
        raise PaddleOcrBundleError("min_text_exact_match must be between 0 and 1")
    if max_confidence_delta < 0.0:
        raise PaddleOcrBundleError("max_confidence_delta must be non-negative")
    output = output_dir.expanduser().resolve()
    if output.exists():
        raise PaddleOcrBundleError(f"Refusing to overwrite existing validation output: {output}")
    bundle = bundle_dir.expanduser().resolve()
    _ensure_output_does_not_overlap_sources(output, [bundle])
    audit_contract = verify_bundle(bundle, require_onnx=True)
    audit_contract_sha256 = _sha256(bundle / CONTRACT_FILENAME)
    images = _iter_ocr_validation_images(input_path, limit=limit)
    try:
        import numpy as np
        from PIL import Image
    except ModuleNotFoundError as error:
        raise PaddleOcrBundleError("ONNX converter parity validation requires numpy and Pillow") from error
    # Both readers receive the same RGB ndarray and are bound to the same
    # immutable snapshot; no mutable Paddle cache or BGR conversion is allowed.
    baseline = _create_paddle_native_reader(bundle)
    candidate = _create_paddle_onnx_reader(bundle)
    comparisons: list[dict[str, object]] = []
    baseline_seconds: list[float] = []
    onnx_seconds: list[float] = []
    try:
        for image_path in images:
            with Image.open(image_path) as source_image:
                image_rgb = np.asarray(source_image.convert("RGB")).copy()
            baseline_start = perf_counter()
            baseline_raw = baseline.ocr(image_rgb, cls=True)
            baseline_elapsed = perf_counter() - baseline_start
            baseline_text, baseline_confidence, baseline_lines = _ocr_result_from_paddle_payload(baseline_raw)
            onnx_start = perf_counter()
            candidate_raw = candidate.ocr(image_rgb, cls=True)
            onnx_elapsed = perf_counter() - onnx_start
            candidate_text, candidate_confidence, candidate_lines = _ocr_result_from_paddle_payload(candidate_raw)
            confidence_delta = (
                None
                if baseline_confidence is None or candidate_confidence is None
                else abs(float(baseline_confidence) - candidate_confidence)
            )
            comparison = {
                "source": str(image_path),
                "baseline": {
                    "text": baseline_text,
                    "confidence": baseline_confidence,
                    "lines": baseline_lines,
                    "elapsed_ms": baseline_elapsed * 1000.0,
                },
                "onnx": {
                    "text": candidate_text,
                    "confidence": candidate_confidence,
                    "lines": candidate_lines,
                    "elapsed_ms": onnx_elapsed * 1000.0,
                },
                "text_exact_match": baseline_text == candidate_text,
                "line_text_exact_match": [str(line["text"]) for line in baseline_lines]
                == [str(line["text"]) for line in candidate_lines],
                "confidence_absolute_delta": confidence_delta,
            }
            comparisons.append(comparison)
            baseline_seconds.append(baseline_elapsed)
            onnx_seconds.append(onnx_elapsed)
    finally:
        # PaddleOCR has no explicit close method.  Releasing references avoids
        # keeping its ORT/Paddle sessions alive during report writing.
        del candidate
        del baseline
    text_matches = sum(bool(record["text_exact_match"]) for record in comparisons)
    line_matches = sum(bool(record["line_text_exact_match"]) for record in comparisons)
    confidence_deltas = [
        float(record["confidence_absolute_delta"])
        for record in comparisons
        if record["confidence_absolute_delta"] is not None
    ]
    output.mkdir(parents=True)
    comparisons_path = output / "comparisons.jsonl"
    _atomic_jsonl(comparisons_path, comparisons)
    summary: dict[str, object] = {
        "schema_version": 1,
        "kind": "paddle_ocr_onnx_conversion_parity_v1",
        "bundle": str(bundle),
        "bundle_contract_sha256": audit_contract_sha256,
        "native_asset_identity_sha256": audit_contract["native_asset_identity"]["sha256"],
        "input": str(input_path.expanduser().resolve()),
        "records": len(comparisons),
        "comparisons_sha256": _sha256(comparisons_path),
        "text_exact_match": text_matches / len(comparisons),
        "line_text_exact_match": line_matches / len(comparisons),
        "max_confidence_absolute_delta": max(confidence_deltas, default=None),
        "baseline_latency_ms": {"p50": _percentile([value * 1000.0 for value in baseline_seconds], 0.50), "p95": _percentile([value * 1000.0 for value in baseline_seconds], 0.95)},
        "onnx_latency_ms": {"p50": _percentile([value * 1000.0 for value in onnx_seconds], 0.50), "p95": _percentile([value * 1000.0 for value in onnx_seconds], 0.95)},
        "acceptance": {
            "min_text_exact_match": min_text_exact_match,
            "max_confidence_delta": max_confidence_delta,
        },
    }
    accepted = (
        float(summary["text_exact_match"]) >= min_text_exact_match
        and (summary["max_confidence_absolute_delta"] is None or float(summary["max_confidence_absolute_delta"]) <= max_confidence_delta)
    )
    summary["accepted"] = accepted
    _atomic_json(output / "summary.json", summary)
    return summary, accepted


def _paddle2onnx_executable(value: str | None) -> str:
    if value:
        executable = Path(value).expanduser()
        if not executable.is_file():
            raise PaddleOcrBundleError(f"paddle2onnx executable does not exist: {executable}")
        return str(executable)
    local_name = "paddle2onnx.exe" if os.name == "nt" else "paddle2onnx"
    local = Path(sys.executable).resolve().parent / local_name
    if local.is_file():
        return str(local)
    discovered = shutil.which("paddle2onnx")
    if discovered:
        return discovered
    raise PaddleOcrBundleError(
        "paddle2onnx was not found. Install the compatible converter with "
        "`python -m pip install --no-deps paddle2onnx==1.3.0`, then pass "
        "`--paddle2onnx $env:VIRTUAL_ENV\\Scripts\\paddle2onnx.exe` on Windows."
    )


def _onnx_metadata(path: Path) -> dict[str, object]:
    """Run a second checker plus ORT load and preserve useful .NET handoff data."""

    try:
        import onnx
        import onnxruntime
    except ModuleNotFoundError as error:
        raise PaddleOcrBundleError(
            "ONNX export verification requires both onnx and onnxruntime. Install ONNX without replacing "
            "the server's GPU runtime, then retry the export."
        ) from error
    onnx.checker.check_model(str(path))
    session = onnxruntime.InferenceSession(str(path), providers=["CPUExecutionProvider"])

    def describe(node: Any) -> dict[str, object]:
        shape: list[int | str | None] = []
        for dimension in node.shape:
            if isinstance(dimension, int):
                shape.append(dimension)
            elif isinstance(dimension, str):
                shape.append(dimension)
            else:
                shape.append(None)
        return {"name": str(node.name), "shape": shape, "type": str(node.type)}

    return {
        "inputs": [describe(node) for node in session.get_inputs()],
        "outputs": [describe(node) for node in session.get_outputs()],
        "onnxruntime_cpu_provider_loaded": "CPUExecutionProvider" in session.get_providers(),
    }


def _require_dynamic_ocr_shapes(role: str, metadata_value: Mapping[str, object]) -> dict[str, object]:
    """Reject a static OCR export before it can be mistaken for a parity build."""

    inputs = metadata_value.get("inputs")
    if not isinstance(inputs, list) or len(inputs) != 1 or not isinstance(inputs[0], Mapping):
        raise PaddleOcrBundleError(f"{role} ONNX must expose exactly one input for this OCR adapter")
    shape = inputs[0].get("shape")
    if not isinstance(shape, list) or len(shape) != 4:
        raise PaddleOcrBundleError(f"{role} ONNX input must be rank-4 NCHW, got {shape!r}")

    def dynamic(value: object) -> bool:
        return value is None or isinstance(value, str)

    dynamic_axes: list[int]
    if role == "det":
        dynamic_axes = [2, 3]
    elif role == "rec":
        dynamic_axes = [3]
    else:
        # The v2 mobile classifier is commonly fixed at 3x48x192.  We record
        # its shape but do not reject it merely for not having a dynamic width.
        dynamic_axes = []
    static_axes = [axis for axis in dynamic_axes if not dynamic(shape[axis])]
    if static_axes:
        raise PaddleOcrBundleError(
            f"{role} ONNX lost required dynamic axis/axes {static_axes}; do not use --input_shape_dict or static OCR export"
        )
    return {"input_shape": shape, "required_dynamic_axes": dynamic_axes}


def export_bundle_onnx(
    *,
    bundle_dir: Path,
    paddle2onnx_executable: str | None = None,
    opset_version: int = 11,
) -> Path:
    """Convert all three frozen Paddle static graphs to dynamic-shape ONNX."""

    if opset_version < 11:
        raise PaddleOcrBundleError("Paddle OCR ONNX export requires opset version 11 or later")
    contract = verify_bundle(bundle_dir)
    bundle, _ = _load_contract(bundle_dir)
    converter = _paddle2onnx_executable(paddle2onnx_executable)
    output_dir = bundle / "onnx"
    if output_dir.exists():
        raise PaddleOcrBundleError(
            f"Refusing to overwrite ONNX output: {output_dir}. Use a fresh snapshot bundle for a new conversion."
        )
    stage = Path(tempfile.mkdtemp(prefix=".onnx.", dir=bundle))
    try:
        records: dict[str, dict[str, object]] = {}
        for role in MODEL_ROLES:
            model_dir = bundle / "paddle" / role
            temporary_output = stage / f"paddle_ocr_{role}.onnx"
            command = [
                converter,
                "--model_dir",
                str(model_dir),
                "--model_filename",
                "inference.pdmodel",
                "--params_filename",
                "inference.pdiparams",
                "--save_file",
                str(temporary_output),
                "--opset_version",
                str(opset_version),
                "--enable_onnx_checker",
                "True",
            ]
            subprocess.run(command, check=True)
            if not temporary_output.is_file() or temporary_output.stat().st_size == 0:
                raise PaddleOcrBundleError(f"paddle2onnx did not create a valid {role} ONNX file")
            record = _file_record(temporary_output, relative_to=stage)
            metadata_value = _onnx_metadata(temporary_output)
            record["io"] = metadata_value
            record["dynamic_shape_validation"] = _require_dynamic_ocr_shapes(role, metadata_value)
            records[role] = record
        # Move all three files as one directory.  A failed conversion therefore
        # never leaves a partially-created ONNX package that looks deployable.
        stage.replace(output_dir)
        for role, record in records.items():
            destination = output_dir / Path(str(record["path"])).name
            durable_record = _file_record(destination, relative_to=bundle)
            durable_record["io"] = record["io"]
            durable_record["dynamic_shape_validation"] = record["dynamic_shape_validation"]
            records[role] = durable_record
        updated = dict(contract)
        updated["onnx"] = records
        updated["onnx_export"] = {
            "converter": "paddle2onnx",
            "paddle2onnx_version": _package_version("paddle2onnx"),
            "opset_version": opset_version,
            "dynamic_shapes": "verified for detector H/W and recognizer width",
            "input_shape_override": None,
            "checker_enabled": True,
            "converter_executable": str(Path(converter).resolve()),
        }
        _atomic_json(bundle / CONTRACT_FILENAME, updated)
    except Exception:
        if output_dir.exists():
            shutil.rmtree(output_dir, ignore_errors=True)
        raise
    finally:
        shutil.rmtree(stage, ignore_errors=True)
    verify_bundle(bundle, require_onnx=True)
    return output_dir


def _parse_device(value: str) -> str:
    value = value.strip().lower()
    if value == "auto" or value == "cpu" or value == "cuda" or (value.startswith("cuda:") and value[5:].isdigit()):
        return value
    raise argparse.ArgumentTypeError("device must be auto, cpu, cuda, or cuda:N")


def _snapshot_main(args: argparse.Namespace) -> int:
    supplied = (args.det_model_dir, args.rec_model_dir, args.cls_model_dir, args.charset)
    if any(value is not None for value in supplied):
        if not all(value is not None for value in supplied):
            raise PaddleOcrBundleError(
                "Provide all --det-model-dir, --rec-model-dir, --cls-model-dir and --charset, or provide none "
                "to snapshot the currently effective PaddleOCR runtime."
            )
        model_dirs = {
            "det": Path(args.det_model_dir),
            "rec": Path(args.rec_model_dir),
            "cls": Path(args.cls_model_dir),
        }
        charset = Path(args.charset)
        effective_args: dict[str, Any] = {
            "det_model_dir": str(model_dirs["det"]),
            "rec_model_dir": str(model_dirs["rec"]),
            "cls_model_dir": str(model_dirs["cls"]),
            "rec_char_dict_path": str(charset),
            "source": "explicit CLI paths; use runtime snapshot for full effective PaddleOCR args",
        }
        runtime: dict[str, object] = {
            "paddleocr_version": _package_version("paddleocr"),
            "python_version": sys.version.split()[0],
            "platform": sys.platform,
            "resolved_by": "explicit CLI paths",
            "snapshot_initialization_device": args.device,
        }
    else:
        model_dirs, charset, effective_args, runtime = _effective_runtime_assets(
            device=args.device,
            allow_model_download=args.allow_model_download,
        )
    bundle = snapshot_bundle(
        output_dir=Path(args.output),
        model_dirs=model_dirs,
        charset_path=charset,
        effective_args=effective_args,
        runtime=runtime,
    )
    print(f"Frozen PaddleOCR bundle: {bundle}")
    print(f"Contract: {bundle / CONTRACT_FILENAME}")
    return 0


def _export_main(args: argparse.Namespace) -> int:
    output = export_bundle_onnx(
        bundle_dir=Path(args.bundle),
        paddle2onnx_executable=args.paddle2onnx,
        opset_version=args.opset_version,
    )
    print(f"Exported dynamic-shape Paddle OCR ONNX models: {output}")
    print(f"Verified contract: {Path(args.bundle).expanduser().resolve() / CONTRACT_FILENAME}")
    return 0


def _verify_main(args: argparse.Namespace) -> int:
    contract = verify_bundle(Path(args.bundle), require_onnx=args.require_onnx)
    onnx = contract.get("onnx", {})
    print(
        f"Paddle OCR bundle verified: {Path(args.bundle).expanduser().resolve()} "
        f"(onnx_roles={','.join(sorted(onnx)) or 'none'})"
    )
    return 0


def _package_delivery_main(args: argparse.Namespace) -> int:
    output = package_delivery_bundle(bundle_dir=Path(args.bundle), output_dir=Path(args.output))
    contract = verify_delivery_bundle(output)
    print(f"Created Paddle-free OCR delivery package: {output}")
    print(f"Verified delivery contract: {output / DELIVERY_CONTRACT_FILENAME} ({contract['package_size_bytes']} bytes)")
    return 0


def _verify_delivery_main(args: argparse.Namespace) -> int:
    contract = verify_delivery_bundle(Path(args.delivery))
    print(
        f"Paddle OCR delivery package verified: {Path(args.delivery).expanduser().resolve()} "
        f"({contract['package_size_bytes']} bytes)"
    )
    return 0


def _validate_onnx_main(args: argparse.Namespace) -> int:
    summary, accepted = validate_onnx_conversion(
        bundle_dir=Path(args.bundle),
        input_path=Path(args.input),
        output_dir=Path(args.output),
        limit=args.limit,
        min_text_exact_match=args.min_text_exact_match,
        max_confidence_delta=args.max_confidence_delta,
    )
    print(
        f"Wrote {summary['records']} native-Paddle/ONNX comparison(s) to {Path(args.output).expanduser().resolve()} "
        f"(text_exact_match={float(summary['text_exact_match']):.2%}, accepted={accepted})"
    )
    return 0 if accepted else 3


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Freeze, export and verify the exact PaddleOCR v2 assets in use.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    snapshot = subparsers.add_parser("snapshot", help="Copy current effective PaddleOCR models, dictionary and contract")
    snapshot.add_argument("--output", required=True, help="New immutable bundle directory")
    snapshot.add_argument("--device", type=_parse_device, default="cpu", help="Paddle device used only to initialise runtime assets")
    snapshot.add_argument(
        "--allow-model-download",
        action="store_true",
        help="Allow PaddleOCR to download missing defaults (normally unsafe for a reproducible snapshot)",
    )
    snapshot.add_argument("--det-model-dir")
    snapshot.add_argument("--rec-model-dir")
    snapshot.add_argument("--cls-model-dir")
    snapshot.add_argument("--charset")
    snapshot.set_defaults(handler=_snapshot_main)

    export = subparsers.add_parser("export-onnx", help="Convert frozen det/rec/cls models to dynamic-shape ONNX")
    export.add_argument("--bundle", required=True)
    export.add_argument(
        "--paddle2onnx",
        help="Path to paddle2onnx executable; defaults to paddle2onnx on PATH",
    )
    export.add_argument("--opset-version", type=int, default=11)
    export.set_defaults(handler=_export_main)

    verify = subparsers.add_parser("verify", help="Validate every SHA-256 recorded in a bundle contract")
    verify.add_argument("--bundle", required=True)
    verify.add_argument("--require-onnx", action="store_true")
    verify.set_defaults(handler=_verify_main)

    package_delivery = subparsers.add_parser(
        "package-delivery",
        help="Copy verified ONNX + character dictionary into a lean Python/Paddle-free delivery directory",
    )
    package_delivery.add_argument("--bundle", required=True)
    package_delivery.add_argument("--output", required=True)
    package_delivery.set_defaults(handler=_package_delivery_main)

    verify_delivery = subparsers.add_parser(
        "verify-delivery",
        help="Validate the hashes in a lean deployment package",
    )
    verify_delivery.add_argument("--delivery", required=True)
    verify_delivery.set_defaults(handler=_verify_delivery_main)

    validate = subparsers.add_parser(
        "validate-onnx",
        help="Compare native PaddleOCR 2.10 with the exported ONNX files on identical RGB crop images",
    )
    validate.add_argument("--bundle", required=True)
    validate.add_argument("--input", required=True, help="One crop image or a recursively scanned crop directory")
    validate.add_argument("--output", required=True)
    validate.add_argument("--limit", type=int)
    validate.add_argument("--min-text-exact-match", type=float, default=1.0)
    validate.add_argument("--max-confidence-delta", type=float, default=0.01)
    validate.set_defaults(handler=_validate_onnx_main)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return int(args.handler(args))
    except (OSError, PaddleOcrBundleError, subprocess.CalledProcessError) as error:
        print(f"Paddle OCR bundle error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":  # pragma: no cover - module execution convenience.
    raise SystemExit(main())
